import os
import re
import sys
import json
import time
import random
import string
import requests
import numpy as np
import pyarrow.parquet as pq
from collections import Counter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from raptor import build_raptor_tree, load_tree, count_tokens, ollama_generate, write_json

# ---- config ----
N_QUESTIONS = 20          # smoke test; `python benchmark.py 200` for the full run
SEED = 42
BUDGET_TOKENS = 1500      # same context budget for flat RAG and RAPTOR when answering
RECALL_BUDGETS = (750, 1500, 3000)  # retrieval-only sweep (no LLM calls needed)
MODEL = "qwen3:8b"        # local model for summarization and answering
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same encoder for both systems
MAX_LEVELS = 3            # RAPTOR depth cap

DATA_DIR = "data"
RESULTS_DIR = "results"

# The official CMU mirror (hotpot_dev_distractor_v1.json) is often unreachable,
# so we use the Hugging Face parquet export of the same dev distractor split.
HOTPOT_URL = (
    "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/"
    "parquet/distractor/validation/0.parquet"
)
HOTPOT_PATH = os.path.join(DATA_DIR, "hotpot_dev_distractor.parquet")


# ---- data ----
def download_hotpot(path=HOTPOT_PATH, url=HOTPOT_URL):
    """Download the HotpotQA dev distractor split once and cache it."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading HotpotQA dev distractor to {path}...")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def load_hotpot(path=HOTPOT_PATH):
    """Load HotpotQA rows as plain dicts with paragraph text joined from sentences."""
    rows = pq.read_table(path).to_pylist()
    questions = []
    for r in rows:
        ctx = r["context"]
        questions.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "type": r["type"],
            "level": r["level"],
            # Gold is paragraph-level: we only care which titles hold
            # supporting facts, not which sentences.
            "gold_titles": sorted(set(r["supporting_facts"]["title"])),
            # HotpotQA sentences carry their own leading spaces, so "".join
            # reconstructs the original paragraph.
            "paragraphs": [
                {"title": t, "text": "".join(s)}
                for t, s in zip(ctx["title"], ctx["sentences"])
            ],
        })
    return questions


def sample_questions(questions, n, seed=SEED):
    """Take the first n of a seeded shuffle, so a smaller N is a subset of a larger one."""
    # Shuffling once and slicing (instead of random.sample(n)) means the N=20
    # smoke test questions are exactly the first 20 of the N=200 run.
    order = list(range(len(questions)))
    random.Random(seed).shuffle(order)
    return [questions[i] for i in order[:n]]


def build_corpus(questions):
    """Pool every question's 10 paragraphs into one corpus, deduplicated by title."""
    # One shared corpus means each retriever must find a question's gold
    # paragraphs among distractors from *all* sampled questions, not just its
    # own 8. Each paragraph is a single leaf (no further splitting), so a
    # retrieved leaf maps exactly onto a gold title.
    corpus, seen = [], set()
    for q in questions:
        for p in q["paragraphs"]:
            if p["title"] in seen:
                continue
            seen.add(p["title"])
            corpus.append({"title": p["title"], "text": p["text"]})
    return corpus


def leaf_text(paragraph):
    """Text a leaf is embedded and shown to the model as: title line + paragraph."""
    # Prepending the title is standard for HotpotQA retrieval: many paragraphs
    # refer to their subject only by pronoun after the title. Both systems get
    # the same leaf text, so this does not favor either.
    return f"{paragraph['title']}\n{paragraph['text']}"


def prepare_data(n=N_QUESTIONS, seed=SEED):
    """Return (questions, corpus) for n sampled questions, cached under data/."""
    path = os.path.join(DATA_DIR, f"hotpot_n{n}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        return cached["questions"], cached["corpus"]

    download_hotpot()
    questions = sample_questions(load_hotpot(), n, seed)
    corpus = build_corpus(questions)
    # Paragraph text now lives in the corpus; keep questions lean.
    for q in questions:
        q.pop("paragraphs")

    with open(path, "w") as f:
        json.dump({"questions": questions, "corpus": corpus}, f, indent=1)
    return questions, corpus


# ---- tree ----
def tree_path(n=N_QUESTIONS, seed=SEED, model=MODEL):
    """Where the RAPTOR tree for this corpus/model is checkpointed."""
    return os.path.join(DATA_DIR, f"tree_n{n}_seed{seed}_{model.replace(':', '-')}.json")


def build_tree(corpus, embeddings, n=N_QUESTIONS, seed=SEED):
    """Build (or resume, or load) the RAPTOR tree over the corpus leaves."""
    leaves = [Document(page_content=leaf_text(p), metadata={"title": p["title"]}) for p in corpus]
    return build_raptor_tree(leaves, embeddings, max_levels=MAX_LEVELS, model=MODEL,
                             tree_path=tree_path(n, seed))


# ---- retrieval ----
def build_index(nodes, embeddings):
    """FAISS index over nodes, with each node's token count stored for budgeting."""
    # all-MiniLM-L6-v2 outputs unit-length vectors, so FAISS's default L2
    # ranking is the same as cosine ranking.
    for node in nodes:
        node.metadata["tokens"] = count_tokens(node.page_content)
    return FAISS.from_documents(nodes, embeddings)


def rank_all(index, query_vector):
    """Every node in the index, most similar first."""
    return index.similarity_search_by_vector(query_vector, k=index.index.ntotal)


def fill_budget(ranked, budget):
    """Take nodes in similarity order until the token budget is full."""
    # Both systems get the same budget in tokens, not the same k: summaries are
    # longer than leaves, so equal k would hand RAPTOR more context. A node
    # that would overflow is skipped (not a hard stop), so one long node near
    # the top can't leave most of either system's budget unused.
    chosen, used = [], 0
    for node in ranked:
        cost = node.metadata["tokens"] + 2  # +2 for the "\n\n" separator
        if used + cost <= budget:
            chosen.append(node)
            used += cost
    return chosen, used


def recall_metrics(chosen, gold_titles):
    """Strict (gold retrieved as a leaf) and lenient (leaf or inside a summary) recall."""
    # A leaf's leaf_titles is just its own title, so for flat RAG lenient
    # equals strict; only RAPTOR summaries can make them differ.
    as_leaf = {n.metadata["title"] for n in chosen if n.metadata["level"] == 0}
    covered = {t for n in chosen for t in n.metadata["leaf_titles"]}
    strict = [t in as_leaf for t in gold_titles]
    lenient = [t in covered for t in gold_titles]
    return {
        "strict_recall": sum(strict) / len(strict),
        "lenient_recall": sum(lenient) / len(lenient),
        "both_strict": all(strict),
        "both_lenient": all(lenient),
    }


# ---- generation and answer metrics ----
def answer_question(question, nodes):
    """Answer from the given context only, as a short phrase."""
    context = "\n\n".join(n.page_content for n in nodes)
    # HotpotQA answers are short spans (or yes/no) scored by EM/F1, so a full
    # sentence would be marked wrong even when correct; ask for the phrase only.
    prompt = (
        "Answer the question using only the context below. Reply with a short "
        "phrase only (for example a name, a date, a number, or yes/no), with no "
        f"explanation and no full sentence.\n\nContext:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return ollama_generate(prompt, MODEL, num_predict=64)


def normalize_answer(s):
    """Official HotpotQA normalization: lowercase, drop punctuation, articles, extra spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction, gold):
    """Token-level F1, as in the official hotpot_evaluate_v1.py."""
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    # yes/no/noanswer get no partial credit: overlapping on "no" is not a
    # half-right answer to a yes/no question.
    if pred in ("yes", "no", "noanswer") and pred != ref:
        return 0.0
    if ref in ("yes", "no", "noanswer") and pred != ref:
        return 0.0
    common = Counter(pred.split()) & Counter(ref.split())
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred.split()), overlap / len(ref.split())
    return 2 * precision * recall / (precision + recall)


def evaluate_question(q, indexes, embeddings):
    """Retrieve at every budget for both systems, then answer at BUDGET_TOKENS."""
    record = {k: q[k] for k in ("id", "question", "answer", "type", "gold_titles")}
    query_vector = embeddings.embed_query(q["question"])
    for name, index in indexes.items():
        ranked = rank_all(index, query_vector)
        result = {"retrieval": {}}
        for budget in RECALL_BUDGETS:
            chosen, used = fill_budget(ranked, budget)
            result["retrieval"][str(budget)] = {
                "ids": [n.metadata["id"] for n in chosen],
                "levels": [n.metadata["level"] for n in chosen],
                "tokens": used,
                **recall_metrics(chosen, q["gold_titles"]),
            }
        context, _ = fill_budget(ranked, BUDGET_TOKENS)
        t0 = time.time()
        prediction = answer_question(q["question"], context)
        result.update(
            # Ids of the exact nodes passed to the model, so the saved results
            # show which system's context produced each answer.
            context_ids=[n.metadata["id"] for n in context],
            prediction=prediction,
            em=exact_match(prediction, q["answer"]),
            f1=f1_score(prediction, q["answer"]),
            answer_seconds=round(time.time() - t0, 2),
        )
        record[name] = result
    return record


# ---- report ----
def paired_bootstrap(diffs, n_resamples=1000, seed=SEED):
    """Mean of per-question differences and its 95% bootstrap confidence interval."""
    # Paired: each resample draws whole questions, keeping a question's flat
    # and RAPTOR scores together. Both systems answer the same questions, so
    # per-question differences are far less noisy than comparing two means.
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return diffs.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pct(x):
    return f"{100 * x:.1f}"


def summary_markdown(records, build_stats, nodes):
    """Markdown tables for retrieval, answers, question types, RAPTOR levels and build cost."""
    systems = ("flat", "raptor")
    lines = [f"## Flat RAG vs RAPTOR on HotpotQA (N={len(records)}, model={MODEL})", ""]

    lines += ["### Retrieval recall (%)", "",
              "| Budget | System | Strict recall | Both found (strict) | Lenient recall | Both found (lenient) | Avg nodes | Avg tokens |",
              "|---|---|---|---|---|---|---|---|"]
    for budget in RECALL_BUDGETS:
        for name in systems:
            rs = [r[name]["retrieval"][str(budget)] for r in records]
            lenient = (pct(mean(x["lenient_recall"] for x in rs)), pct(mean(x["both_lenient"] for x in rs))) \
                if name == "raptor" else ("-", "-")
            lines.append(f"| {budget} | {name} | {pct(mean(x['strict_recall'] for x in rs))} | "
                         f"{pct(mean(x['both_strict'] for x in rs))} | {lenient[0]} | {lenient[1]} | "
                         f"{mean(len(x['ids']) for x in rs):.1f} | {mean(x['tokens'] for x in rs):.0f} |")

    b = str(BUDGET_TOKENS)
    lines += ["", f"### Answers and recall by question type (budget {BUDGET_TOKENS} tokens, %)", "",
              "| System | Type | n | EM | F1 | Strict recall | Both found (strict) | Lenient recall | Avg answer s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name in systems:
        for qtype in ("all", "bridge", "comparison"):
            rs = [r for r in records if qtype == "all" or r["type"] == qtype]
            if not rs:
                continue
            ret = [r[name]["retrieval"][b] for r in rs]
            lenient = pct(mean(x["lenient_recall"] for x in ret)) if name == "raptor" else "-"
            lines.append(f"| {name} | {qtype} | {len(rs)} | {pct(mean(r[name]['em'] for r in rs))} | "
                         f"{pct(mean(r[name]['f1'] for r in rs))} | {pct(mean(x['strict_recall'] for x in ret))} | "
                         f"{pct(mean(x['both_strict'] for x in ret))} | {lenient} | "
                         f"{mean(r[name]['answer_seconds'] for r in rs):.1f} |")

    n_ctx = sum(r["flat"]["context_ids"] != r["raptor"]["context_ids"] for r in records)
    n_ans = sum(r["flat"]["prediction"] != r["raptor"]["prediction"] for r in records)
    lines += ["", f"### RAPTOR minus flat, paired bootstrap (budget {BUDGET_TOKENS}, "
                  f"1000 resamples, seed {SEED}, percentage points)", "",
              f"Contexts differ on {n_ctx}/{len(records)} questions; "
              f"answer strings differ on {n_ans}/{len(records)}.", "",
              "| Metric | Flat | RAPTOR | Difference | 95% CI |",
              "|---|---|---|---|---|"]
    per_question = {
        "EM": lambda r, s: r[s]["em"],
        "F1": lambda r, s: r[s]["f1"],
        "Strict recall": lambda r, s: r[s]["retrieval"][b]["strict_recall"],
    }
    for metric, get in per_question.items():
        diff, lo, hi = paired_bootstrap([get(r, "raptor") - get(r, "flat") for r in records])
        lines.append(f"| {metric} | {pct(mean(get(r, 'flat') for r in records))} | "
                     f"{pct(mean(get(r, 'raptor') for r in records))} | {100 * diff:+.1f} | "
                     f"[{100 * lo:+.1f}, {100 * hi:+.1f}] |")

    max_level = max(n.metadata["level"] for n in nodes)
    lines += ["", "### RAPTOR retrieved node levels (% of retrieved nodes)", "",
              "| Budget | " + " | ".join(f"L{l}" for l in range(max_level + 1)) + " |",
              "|---|" + "---|" * (max_level + 1)]
    for budget in RECALL_BUDGETS:
        counts = Counter(l for r in records for l in r["raptor"]["retrieval"][str(budget)]["levels"])
        total = sum(counts.values())
        lines.append(f"| {budget} | " + " | ".join(pct(counts[l] / total) for l in range(max_level + 1)) + " |")

    per_level = Counter(n.metadata["level"] for n in nodes)
    calls = build_stats["summary_seconds"]
    lines += ["", "### RAPTOR build cost", "",
              f"- Nodes per level: {dict(sorted(per_level.items()))}",
              f"- Summarization calls: {build_stats['summary_calls']}",
              f"- Tree build time: {build_stats['build_seconds'] / 60:.1f} min "
              f"(summarization: {sum(calls) / 60:.1f} min, {mean(calls):.1f} s/call)"]
    return "\n".join(lines)


def run(n):
    """Build everything for n questions, evaluate both systems, save and print results."""
    questions, corpus = prepare_data(n)
    print(f"{len(questions)} questions, {len(corpus)} unique paragraphs")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    nodes = build_tree(corpus, embeddings, n)
    _, build_stats = load_tree(tree_path(n))

    # Flat RAG indexes exactly the tree's level-0 nodes, so both systems
    # retrieve the same leaf texts with the same ids; only the summaries differ.
    # Built in memory each run (seconds), separate from faiss_index/ and raptor_index/.
    indexes = {
        "flat": build_index([x for x in nodes if x.metadata["level"] == 0], embeddings),
        "raptor": build_index(nodes, embeddings),
    }

    records, t0 = [], time.time()
    for i, q in enumerate(questions, 1):
        r = evaluate_question(q, indexes, embeddings)
        records.append(r)
        print(f"[{i}/{n}] {q['type']:<10} gold={q['answer']!r} | "
              f"flat={r['flat']['prediction']!r} | raptor={r['raptor']['prediction']!r}")
    eval_seconds = time.time() - t0

    summary = summary_markdown(records, build_stats, nodes)
    summary += f"\n- Evaluation time: {eval_seconds / 60:.1f} min for {n} questions"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"benchmark_n{n}_seed{SEED}_{MODEL.replace(':', '-')}")
    write_json({
        "config": {"n": n, "seed": SEED, "model": MODEL, "embed_model": EMBED_MODEL,
                   "budget_tokens": BUDGET_TOKENS, "recall_budgets": RECALL_BUDGETS,
                   "max_levels": MAX_LEVELS},
        "build_stats": build_stats,
        "eval_seconds": eval_seconds,
        "records": records,
    }, out + ".json")
    with open(out + ".md", "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nSaved {out}.json and {out}.md")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else N_QUESTIONS)
