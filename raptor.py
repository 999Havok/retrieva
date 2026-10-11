import os
import re
import json
import time
import hashlib
import numpy as np
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import umap
from sklearn.mixture import GaussianMixture
import requests

load_dotenv()

# Ollama's default context window is small and silently drops the start of
# longer prompts, so we set it explicitly and keep every summarization input
# under MAX_SUMMARY_INPUT_TOKENS. The gap leaves room for the instructions, the
# output (NUM_PREDICT is only a runaway guard; summaries should end well before
# it), and tokenizer mismatch: we count with the Qwen3 tokenizer, which is
# close to but not identical to llama3.2's.
NUM_CTX = 8192
NUM_PREDICT = 1024
MAX_SUMMARY_INPUT_TOKENS = 6000
# ~120 words matches the RAPTOR paper's short summaries (~130 tokens on average).
SUMMARY_WORDS = 120
SUMMARY_SYSTEM = (
    f"You write summaries. A summary is never longer than {SUMMARY_WORDS} words, "
    "no matter how much input you get; when the input covers many topics, you "
    "mention only the most salient facts."
)

_tokenizer = None


def count_tokens(text: str) -> int:
    """Count tokens with the Qwen3 tokenizer (loaded once, on first use)."""
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-8B")
    return len(_tokenizer.encode(text, add_special_tokens=False).ids)


def strip_think(text: str) -> str:
    """Remove any <think>...</think> block a reasoning model emits despite think=false."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ollama_generate(prompt: str, model: str, system: str = None,
                    num_predict: int = NUM_PREDICT) -> str:
    """Call the local Ollama server deterministically with thinking disabled."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
            "temperature": 0,
        },
    }
    if system:
        body["system"] = system
    response = requests.post("http://127.0.0.1:11434/api/generate", json=body, timeout=600)
    response.raise_for_status()
    result = response.json()
    if result.get("done_reason") == "length":
        print(f"  warning: output hit num_predict={num_predict} and was cut off")
    return strip_think(result["response"])


def summarize_with_ollama(text: str, model: str = "llama3.2:3b") -> str:
    """Call the local Ollama server to summarize text."""
    # The word cap keeps summaries short enough that a whole cluster of them
    # still fits the context window when the next level summarizes them.
    prompt = (
        f"Write a dense summary of the following text in at most {SUMMARY_WORDS} "
        "words. Preserve key facts, names, and entities. Do NOT begin with phrases "
        "like 'Here is a summary' or 'The text describes'. Output only the summary "
        f"itself, nothing else.\n\n{text}\n\nSummary:"
    )
    # The length rule also lives in the system prompt because qwen3 ignores a
    # word cap in the user prompt when a cluster mixes unrelated topics (it
    # lists every fact; ~600 words in testing).
    return ollama_generate(prompt, model, system=SUMMARY_SYSTEM)


# clustering
def cluster_embeddings(embeddings, max_clusters=10, prob_threshold=0.1):
    """Soft-cluster embeddings via UMAP dim-reduction + BIC-selected Gaussian Mixture."""
    embeddings = np.asarray(embeddings)

    # RAPTOR clusters in a reduced space: GMMs struggle in 384 dims (distances
    # concentrate, covariances get unstable), so we compress to ~10 dims first.
    # n_neighbors is capped below the sample count because UMAP needs a local
    # neighborhood smaller than the dataset, and cosine matches how the
    # sentence-transformer embeddings are compared at retrieval time.
    # random_state=42 makes the reduction reproducible (cost is negligible here).
    # n_components must stay well below n_samples: UMAP's spectral initializer
    # fails when the target dimension approaches the sample count, so we cap at
    # n_samples - 2 and floor at 2 to give the spectral solver room to work.
    n_samples = len(embeddings)
    n_components = max(2, min(10, n_samples - 2))
    n_neighbors = min(15, n_samples - 1)
    try:
        reduced = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings)
    except (TypeError, ValueError):
        # Small-N fallback: init="random" skips the spectral solver entirely,
        # which is the documented workaround when spectral init can't converge.
        reduced = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            init="random",
            random_state=42,
        ).fit_transform(embeddings)

    # Pick the cluster count that minimizes BIC. BIC rewards fit but penalizes
    # extra components, so it naturally avoids over-splitting small levels
    # instead of us hard-coding k.
    max_clusters = min(max_clusters, len(embeddings))
    best_gmm, best_bic, best_k = None, np.inf, 1
    for k in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(reduced)
        bic = gmm.bic(reduced)
        if bic < best_bic:
            best_gmm, best_bic, best_k = gmm, bic, k

    # Soft assignment: a point belongs to every cluster where the GMM's
    # posterior probability exceeds prob_threshold. This matches RAPTOR's
    # overlapping-cluster design, so a chunk spanning two topics can inform
    # both summaries. Fall back to the argmax cluster if nothing clears the
    # threshold, so no point is ever orphaned.
    proba = best_gmm.predict_proba(reduced)
    assignments = []
    for row in proba:
        clusters = [c for c, p in enumerate(row) if p > prob_threshold]
        if not clusters:
            clusters = [int(np.argmax(row))]
        assignments.append(clusters)

    return assignments, best_k


# Small models sometimes refuse or hallucinate an empty input instead of
# summarizing; these markers let us drop such nodes before they pollute the index.
_REFUSAL_MARKERS = (
    "i don't see",
    "please provide",
    "no text",
    "there is no text",
    "i'm not sure what",
)


def is_bad_summary(text: str) -> bool:
    """Return True if a summary looks like a refusal or is too short to be useful."""
    if len(text) < 40:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


# fitting clusters into the context window
def _fits(idx, token_counts, max_tokens):
    """True if the texts at idx, joined with blank lines, fit max_tokens."""
    # +2 per text approximates the "\n\n" separator.
    return sum(token_counts[i] + 2 for i in idx) <= max_tokens


def pack_to_fit(idx, embeddings, token_counts, max_tokens):
    """Deterministically split a group into consecutive chunks that each fit max_tokens."""
    # Order along the group's first principal component so each chunk holds
    # neighboring (similar) texts rather than an arbitrary slice of the group.
    vecs = embeddings[idx]
    centered = vecs - vecs.mean(axis=0)
    pc = np.linalg.svd(centered, full_matrices=False)[2][0]
    ordered = [idx[i] for i in np.argsort(centered @ pc, kind="stable")]

    chunks, current = [], []
    for i in ordered:
        # Never truncate: a text that can't fit on its own is a hard error.
        if not _fits([i], token_counts, max_tokens):
            raise ValueError(
                f"A single text has {token_counts[i]} tokens, over the "
                f"{max_tokens}-token summarization budget"
            )
        if current and not _fits(current + [i], token_counts, max_tokens):
            chunks.append(current)
            current = []
        current.append(i)
    if current:
        chunks.append(current)
    return chunks


def split_to_fit(idx, embeddings, token_counts, max_tokens):
    """Re-cluster an oversized group until every piece fits; pack what clustering can't split."""
    # As in the RAPTOR paper, an oversized cluster is clustered again on its
    # own. This also keeps the BIC sweep cheap: each call sweeps K <= 10, and
    # big levels are handled by recursing (10 clusters of 10 of ...) instead of
    # sweeping K up to n/10 on the whole level. Recursion only continues on
    # strictly smaller groups, so it terminates. If BIC sees no sub-structure
    # (K*=1), or soft assignment puts the whole group in one sub-cluster,
    # clustering can't help and we fall back to deterministic packing.
    # Below 5 points UMAP + GMM is meaningless, so pack directly.
    if len(idx) >= 5:
        assignments, k = cluster_embeddings(embeddings[idx])
        if k > 1:
            groups = []
            for c in range(k):
                sub = [idx[i] for i, cs in enumerate(assignments) if c in cs]
                if not sub:
                    continue
                if _fits(sub, token_counts, max_tokens):
                    groups.append(sub)
                elif len(sub) < len(idx):
                    groups.extend(split_to_fit(sub, embeddings, token_counts, max_tokens))
                else:
                    groups.extend(pack_to_fit(sub, embeddings, token_counts, max_tokens))
            return groups
    return pack_to_fit(idx, embeddings, token_counts, max_tokens)


# tree persistence
def write_json(data, path):
    """Write JSON via write-then-rename, so a crash mid-write never leaves a corrupt file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def save_tree(nodes, stats, path):
    """Write all nodes built so far plus build stats to JSON."""
    data = {"stats": stats, "nodes": [{"text": n.page_content, **n.metadata} for n in nodes]}
    write_json(data, path)


def load_tree(path):
    """Load (nodes, stats) written by save_tree."""
    with open(path) as f:
        data = json.load(f)
    nodes = [Document(page_content=n.pop("text"), metadata=n) for n in data["nodes"]]
    return nodes, data["stats"]


# tree building
def build_raptor_tree(chunks, embeddings_model, max_levels=3, model="llama3.2:3b",
                      tree_path=None, max_tokens=MAX_SUMMARY_INPUT_TOKENS):
    """Recursively cluster + summarize chunks into a flat 'collapsed tree' of Documents.

    Every node gets an id ("<level>-<index>"). Summary nodes store their
    children ids and the sorted titles of all leaves they descend from (a leaf's
    title is its chunk's metadata["title"], or its id if it has none). If
    tree_path is given, the tree is checkpointed there after every level, each
    summary is cached as soon as it returns, and a rerun resumes where it stopped.
    """
    stats = {"levels_done": 0, "complete": False, "summary_calls": 0,
             "summary_seconds": [], "build_seconds": 0.0}
    if tree_path and os.path.exists(tree_path):
        all_nodes, stats = load_tree(tree_path)
        if stats["complete"]:
            print(f"Loaded finished tree from {tree_path} ({len(all_nodes)} nodes)")
            return all_nodes
        print(f"Resuming {tree_path} after level {stats['levels_done']}")
    else:
        # Level 0 is the raw leaf chunks. all_nodes accumulates every node across
        # every level; this flat list *is* the collapsed tree we retrieve over later.
        all_nodes = []
        for i, c in enumerate(chunks):
            node_id = f"0-{i}"
            title = c.metadata.get("title")
            all_nodes.append(Document(page_content=c.page_content, metadata={
                "id": node_id, "level": 0, "cluster": None, "title": title,
                "children": [], "leaf_titles": [title or node_id],
            }))
        print(f"Level 0: {len(all_nodes)} leaf chunks")

    def checkpoint():
        stats["build_seconds"] = prior_seconds + time.time() - run_start
        if tree_path:
            save_tree(all_nodes, stats, tree_path)

    # Per-call summary cache, saved after every call, so a crash mid-level
    # loses at most the one call in flight (level checkpoints alone would lose
    # the whole level).
    cache_path = os.path.splitext(tree_path)[0] + "_summaries.json" if tree_path else None
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    prior_seconds, run_start = stats["build_seconds"], time.time()
    current = [n for n in all_nodes if n.metadata["level"] == stats["levels_done"]]

    for level in range(stats["levels_done"] + 1, max_levels + 1):
        # A single node can't be clustered; nothing left to abstract.
        if len(current) < 2:
            break

        texts = [n.page_content for n in current]
        embeddings = np.asarray(embeddings_model.embed_documents(texts))
        token_counts = [count_tokens(t) for t in texts]
        assignments, n_clusters = cluster_embeddings(embeddings)

        # BIC-based stop: when the best fit is a single cluster, the data is
        # telling us this level is one coherent topic with no further useful
        # sub-structure. Summarizing it would just collapse everything into one
        # node, so we stop here and let max_levels act only as a safety cap.
        if n_clusters == 1:
            print(f"Level {level}: BIC chose 1 cluster, stopping")
            break

        # Degenerate stop: if every node landed in its own cluster, no actual
        # grouping happened and summaries would just be 1:1 rewrites, so stop.
        if n_clusters == len(current):
            print(f"Level {level}: BIC chose K={n_clusters} clusters, no grouping happening, stopping.")
            break

        # Invert the per-point assignments into per-cluster index groups. A
        # point assigned to multiple clusters (soft membership) goes into each,
        # matching RAPTOR's overlapping-cluster summarization. Groups too big
        # for the summarizer's context are re-clustered rather than truncated.
        groups, n_resplit = [], 0
        for c in range(n_clusters):
            idx = [i for i, cs in enumerate(assignments) if c in cs]
            if not idx:
                continue
            if _fits(idx, token_counts, max_tokens):
                groups.append(idx)
            else:
                n_resplit += 1
                groups.extend(split_to_fit(idx, embeddings, token_counts, max_tokens))
        print(f"Level {level}: {len(current)} nodes -> {n_clusters} clusters "
              f"({n_resplit} re-split) -> {len(groups)} groups to summarize")

        # Summarize each group into one abstractive node for the next level up.
        next_nodes = []
        for g, idx in enumerate(groups):
            members = [current[i] for i in idx]
            combined = "\n\n".join(m.page_content for m in members)
            # Keyed by the exact summarizer input (plus model and length rule),
            # so a resumed level reuses finished calls even if group order
            # shifts, and a changed prompt setting never reuses stale text.
            key = hashlib.sha1(f"{model}\n{SUMMARY_SYSTEM}\n{combined}".encode()).hexdigest()
            if key not in cache:
                t0 = time.time()
                summary = summarize_with_ollama(combined, model=model)
                cache[key] = {"summary": summary, "seconds": round(time.time() - t0, 2)}
                if cache_path:
                    write_json(cache, cache_path)
            summary = cache[key]["summary"]
            # Cached calls still count, with their original time, so build cost
            # reflects the whole tree no matter how many runs it took.
            stats["summary_calls"] += 1
            stats["summary_seconds"].append(cache[key]["seconds"])
            # Drop refusals/empties so they never enter the index or feed the
            # next level; a bad summary here would corrupt everything above it.
            if is_bad_summary(summary):
                print(f"Level {level}: dropped bad summary for group {g}")
                continue
            leaf_titles = sorted({t for m in members for t in m.metadata["leaf_titles"]})
            next_nodes.append(Document(page_content=summary, metadata={
                "id": f"{level}-{len(next_nodes)}", "level": level, "cluster": g,
                "children": [m.metadata["id"] for m in members],
                "leaf_titles": leaf_titles,
            }))

        # The summaries become the leaves for the next round of clustering.
        all_nodes.extend(next_nodes)
        current = next_nodes
        stats["levels_done"] = level
        checkpoint()
        print(f"Level {level}: wrote {len(next_nodes)} summary nodes "
              f"({stats['summary_calls']} calls so far)")

    stats["complete"] = True
    checkpoint()
    return all_nodes


# collapsed-tree index
def save_raptor_index(nodes, embeddings_model, path="raptor_index"):
    """Build a FAISS index over the flat all-level node list and save it."""
    # One index over every level: at query time a leaf chunk and a high-level
    # summary compete on equal footing, which is the point of the collapsed tree.
    vectorstore = FAISS.from_documents(nodes, embeddings_model)
    vectorstore.save_local(path)
    print(f"RAPTOR index saved to {path}/ ({len(nodes)} nodes)")
    return vectorstore


def load_raptor_index(path="raptor_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the collapsed-tree FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


# query
def raptor_query(vectorstore, question, k=5):
    """Retrieve top-k collapsed-tree nodes (leaves + summaries) and answer with Gemini."""
    # similarity_search (not RetrievalQA) so we keep the raw nodes and can report
    # the tree level each retrieved node came from — the interesting bit here is
    # seeing leaves and summaries mix in a single top-k result set.
    nodes = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join(node.page_content for node in nodes)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    answer = llm.invoke(prompt).content

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")
    print(f"\nRetrieved {len(nodes)} nodes:")
    for i, node in enumerate(nodes):
        level = node.metadata.get("level")
        kind = "leaf" if level == 0 else "summary"
        print(f"\n--- Node {i} | level {level} ({kind}) | cluster {node.metadata.get('cluster')} ---")
        print(node.page_content[:300])

    return answer, nodes


if __name__ == "__main__":
    from ingest import load_document, split_documents

    # Reuse the same loader/splitter as the traditional RAG pipeline so the
    # only variable is the RAPTOR tree on top. sample_doc.txt is a multi-topic
    # document (water cycle, bees, jazz, volcanoes, economics, printing,
    # neural nets) chosen so clustering finds clearly distinct topic groups.
    docs = load_document("sample_doc.txt")
    chunks = split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\n=== Building RAPTOR tree ===")
    nodes = build_raptor_tree(chunks, embeddings)
    save_raptor_index(nodes, embeddings)

    # Load once, reuse across queries.
    vectorstore = load_raptor_index()

    print("\n=== Running queries ===")
    # Broad question: should favor high-level summary nodes that span a topic.
    raptor_query(vectorstore, "How does jazz improvisation relate to its African and European roots?")
    # Specific question: should favor a leaf chunk with the concrete detail.
    raptor_query(vectorstore, "What does the waggle dance tell other bees?")
    # Cross-topic question: forces retrieval to mix nodes from different clusters.
    raptor_query(vectorstore, "What role does heat or energy from the sun play across these topics?")

