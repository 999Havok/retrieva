import os
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

def summarize_with_ollama(text: str, model: str = "llama3.2:3b") -> str:
    """Call the local Ollama server to summarize text."""
    prompt = (
        "Write a dense summary of the following text. Preserve key facts, names, "
        "and entities. Do NOT begin with phrases like 'Here is a summary' or 'The "
        "text describes'. Output only the summary itself, nothing else.\n\n"
        f"{text}\n\nSummary:"
    )
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"].strip()


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


# tree building
def build_raptor_tree(chunks, embeddings_model, max_levels=3):
    """Recursively cluster + summarize chunks into a flat 'collapsed tree' of Documents."""
    # Level 0 is the raw leaf chunks. all_nodes accumulates every node across
    # every level; this flat list *is* the collapsed tree we retrieve over later.
    all_nodes = [
        Document(page_content=c.page_content, metadata={"level": 0, "cluster": None})
        for c in chunks
    ]

    current_texts = [c.page_content for c in chunks]
    current_embeddings = embeddings_model.embed_documents(current_texts)
    print(f"Level 0: {len(current_texts)} leaf chunks")

    for level in range(1, max_levels + 1):
        # A single node can't be clustered; nothing left to abstract.
        if len(current_texts) < 2:
            break

        assignments, n_clusters = cluster_embeddings(current_embeddings)

        # BIC-based stop: when the best fit is a single cluster, the data is
        # telling us this level is one coherent topic with no further useful
        # sub-structure. Summarizing it would just collapse everything into one
        # node, so we stop here and let max_levels act only as a safety cap.
        if n_clusters == 1:
            print(f"Level {level}: BIC chose 1 cluster, stopping")
            break

        # Degenerate stop: if every node landed in its own cluster, no actual
        # grouping happened and summaries would just be 1:1 rewrites, so stop.
        if n_clusters == len(current_texts):
            print(f"Level {level}: BIC chose K={n_clusters} clusters, no grouping happening, stopping.")
            break

        print(f"Level {level}: clustering {len(current_texts)} nodes into {n_clusters} clusters...")

        # Invert the per-point assignments into per-cluster text groups. A point
        # assigned to multiple clusters (soft membership) is concatenated into
        # each, matching RAPTOR's overlapping-cluster summarization.
        cluster_texts = {c: [] for c in range(n_clusters)}
        for text, clusters in zip(current_texts, assignments):
            for c in clusters:
                cluster_texts[c].append(text)

        # Summarize each non-empty cluster into one abstractive node for the
        # next level up. Empty clusters (no point cleared the threshold) skipped.
        next_texts = []
        for c in range(n_clusters):
            if not cluster_texts[c]:
                continue
            combined = "\n\n".join(cluster_texts[c])
            summary = summarize_with_ollama(combined)
            # Drop refusals/empties so they never enter the index or feed the
            # next level; a bad summary here would corrupt everything above it.
            if is_bad_summary(summary):
                print(f"Level {level}: dropped bad summary for cluster {c}")
                continue
            all_nodes.append(
                Document(page_content=summary, metadata={"level": level, "cluster": c})
            )
            next_texts.append(summary)

        # The summaries become the leaves for the next round of clustering.
        current_texts = next_texts
        current_embeddings = embeddings_model.embed_documents(current_texts)
        print(f"Level {level}: wrote {len(next_texts)} summary nodes")

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

