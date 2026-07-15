# retrieva

Traditional RAG and RAPTOR (hierarchical RAG) implementations from scratch, using LangChain, FAISS, sentence-transformers, Ollama, and Gemini.

## What and why

Retrieval-Augmented Generation (RAG) grounds LLM answers in your own documents by retrieving relevant chunks via vector similarity and passing them as context to a generation model. Traditional RAG treats every chunk as an independent peer, which fails when an answer is scattered across many chunks or requires abstraction across the corpus. RAPTOR (Sarthi et al., 2024) addresses this by recursively clustering and summarizing chunks into a tree, so higher-level nodes unify scattered information and a single retrieval step can surface both fine-grained leaves and abstract summaries.

## Architecture

**Traditional RAG** (`ingest.py`, `query.py`): load document, chunk with `RecursiveCharacterTextSplitter`, embed with `sentence-transformers/all-MiniLM-L6-v2`, store in a FAISS index. At query time, retrieve top-k chunks and send them to Gemini for the final answer.

**RAPTOR** (`raptor.py`): level 0 is the leaf chunks. Each subsequent level is built by clustering the current level's embeddings (UMAP for dimensionality reduction, GMM for soft clustering), summarizing each cluster with a local LLM, embedding those summaries, and feeding them as input to the next level. Tree-building stops when BIC selects one cluster, when clustering is degenerate (K equals N), or when the max-level cap is hit. At query time, all nodes across all levels are flattened into a single FAISS index and searched together (collapsed-tree retrieval).

## Setup

1. Clone the repo and `cd retrieva`
2. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Install Ollama from ollama.com, then `ollama pull llama3.2:3b`
5. Create a `.env` file with `GOOGLE_API_KEY=your_key_here` (from aistudio.google.com/apikey)

## Usage

Traditional RAG:

```
python ingest.py
python query.py
```

RAPTOR (start `ollama serve` in a separate terminal first):

```
python raptor.py
```

## Design decisions

- **Soft GMM cluster assignment (threshold 0.1)**: a chunk can belong to multiple clusters. Semantically overlapping chunks inform multiple parent summaries, matching the RAPTOR paper.
- **UMAP with cosine metric before clustering**: GMM degrades in 384-dim space (distance concentration). Reducing to ~10 dims first with a cosine metric preserves the angular similarity used at retrieval time.
- **BIC for cluster count selection**: the number of clusters is data-driven per level rather than hard-coded. BIC's complexity penalty discourages over-splitting, which matters at higher levels where node counts shrink.
- **BIC-based and degenerate stopping**: tree-building stops when BIC picks K equals 1 (level is one coherent topic) or when K equals N (no actual grouping happening). Prevents wasteful one-to-one "summary" levels.
- **Collapsed-tree retrieval**: all nodes across all levels compete in one FAISS index, so a single query can pull both a leaf chunk and a high-level summary if both are relevant.
- **Ollama for summarization, Gemini for final answer**: tree-building makes many summarization calls per document, which would exhaust Gemini's free-tier rate limit. A local model (`llama3.2:3b` via Ollama) removes that constraint. Gemini is reserved for the single answer-generation call per user query, where its stronger reasoning matters more.
