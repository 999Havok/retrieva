"""Offline unit tests for raptor.py: no Ollama, Gemini, GPU, or network."""

import hashlib
import json

import numpy as np
import pytest
from langchain_core.documents import Document

import raptor


# ---------------------------------------------------------------- helpers
class FakeEmbeddings:
    """Deterministic stand-in for HuggingFaceEmbeddings.embed_documents."""

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).normal(size=self.dim))
        return np.asarray(out)


class FakeGMM:
    """GaussianMixture stand-in with scripted BIC values and posteriors."""

    bics = None
    proba = None

    def __init__(self, n_components, random_state=None):
        self.k = n_components

    def fit(self, X):
        return self

    def bic(self, X):
        return self.bics[self.k]

    def predict_proba(self, X):
        return np.asarray(self.proba)


class IdentityUMAP:
    def __init__(self, **kwargs):
        pass

    def fit_transform(self, X):
        return np.asarray(X)


def _no_llm(*args, **kwargs):
    raise AssertionError("summarizer must not be called in this test")


def _chunks(n):
    return [
        Document(page_content=f"chunk number {i} " * 5, metadata={"title": f"T{i}"})
        for i in range(n)
    ]


@pytest.fixture
def offline(monkeypatch):
    """Token counting without downloading the Qwen tokenizer."""
    monkeypatch.setattr(raptor, "count_tokens", lambda text: len(text.split()))


# ------------------------------------------------------ is_bad_summary
@pytest.mark.parametrize(
    "text",
    [
        "",
        "Too short.",
        "x" * 39,
    ],
)
def test_is_bad_summary_rejects_short_text(text):
    assert raptor.is_bad_summary(text)


@pytest.mark.parametrize(
    "text",
    [
        "I don't see any text to summarize here, could you share it again?",
        "PLEASE PROVIDE the document you would like me to summarize for you.",
        "There is no text in your message, so there is nothing to summarize.",
        "I'm not sure what you would like summarized; the input appears empty.",
    ],
)
def test_is_bad_summary_rejects_refusals_case_insensitively(text):
    assert raptor.is_bad_summary(text)


def test_is_bad_summary_keeps_real_summary():
    text = "Honeybees use the waggle dance to tell nestmates the direction and distance of food."
    assert not raptor.is_bad_summary(text)


def test_strip_think_removes_reasoning_block():
    assert (
        raptor.strip_think("<think>\nplan\n</think>\n Final summary.")
        == "Final summary."
    )


# ------------------------------------------------- cluster_embeddings
def _patch_gmm(monkeypatch, bics, proba):
    monkeypatch.setattr(raptor.umap, "UMAP", IdentityUMAP)
    monkeypatch.setattr(raptor, "GaussianMixture", FakeGMM)
    monkeypatch.setattr(FakeGMM, "bics", bics)
    monkeypatch.setattr(FakeGMM, "proba", proba)


def test_soft_assignment_uses_probability_threshold(monkeypatch):
    proba = [
        [0.95, 0.05],  # clearly cluster 0
        [0.55, 0.45],  # overlaps both clusters
        [0.10, 0.90],  # exactly at threshold for cluster 0: not included (strict >)
        [0.02, 0.98],  # clearly cluster 1
    ]
    _patch_gmm(monkeypatch, {1: 10.0, 2: 1.0, 3: 5.0, 4: 7.0}, proba)

    assignments, k = raptor.cluster_embeddings(np.zeros((4, 3)), prob_threshold=0.1)

    assert k == 2
    assert assignments == [[0], [0, 1], [1], [1]]


def test_soft_assignment_falls_back_to_argmax(monkeypatch):
    # With a high threshold nothing qualifies; the point must not be orphaned.
    proba = [[0.40, 0.35, 0.25], [0.20, 0.30, 0.50]]
    _patch_gmm(monkeypatch, {1: 9.0, 2: 9.0, 3: 0.0}, proba)

    assignments, k = raptor.cluster_embeddings(
        np.zeros((6, 3)), max_clusters=3, prob_threshold=0.6
    )

    assert k == 3
    assert assignments == [[0], [2]]


def test_bic_picks_lowest_and_caps_k_at_sample_count(monkeypatch):
    # K=1 has the lowest BIC; K is also capped at the 3 samples.
    _patch_gmm(monkeypatch, {1: -5.0, 2: 0.0, 3: 1.0}, [[1.0]] * 3)

    assignments, k = raptor.cluster_embeddings(np.zeros((3, 3)), max_clusters=10)

    assert k == 1
    assert assignments == [[0], [0], [0]]


def test_cluster_embeddings_never_mixes_separated_blobs():
    # End-to-end through real UMAP + GMM on three well-separated blobs. BIC may
    # pick more than 3 clusters here (on this data it picks 9: UMAP pulls each
    # blob into tight sub-clumps), so only check that no cluster spans blobs
    # and that every point is assigned.
    rng = np.random.default_rng(0)
    centers = np.eye(3, 16) * 50
    points = np.vstack([c + rng.normal(scale=0.5, size=(12, 16)) for c in centers])
    blob_of = [i // 12 for i in range(len(points))]

    assignments, k = raptor.cluster_embeddings(points)

    assert 3 <= k <= 10
    assert all(assignments) and all(0 <= c < k for a in assignments for c in a)
    for c in range(k):
        blobs = {blob_of[i] for i, a in enumerate(assignments) if c in a}
        assert len(blobs) <= 1, f"cluster {c} mixes blobs {blobs}"


# ------------------------------------------------- packing / splitting
def test_fits_counts_separator_tokens():
    counts = [4, 4]
    assert raptor._fits([0, 1], counts, 12)  # (4+2) + (4+2) = 12
    assert not raptor._fits([0, 1], counts, 11)


def test_pack_to_fit_covers_every_index_once_within_budget():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(10, 4))
    counts = [30, 10, 25, 5, 40, 15, 20, 35, 10, 5]
    idx = list(range(10))

    chunks = raptor.pack_to_fit(idx, emb, counts, max_tokens=60)

    assert sorted(i for c in chunks for i in c) == idx
    assert all(raptor._fits(c, counts, 60) for c in chunks)
    assert len(chunks) > 1


def test_pack_to_fit_orders_along_principal_component():
    # Points on a line: packing must keep neighbors together, not input order.
    emb = np.array([[0.0, 0], [3, 0], [1, 0], [2, 0]])
    counts = [10, 10, 10, 10]

    chunks = raptor.pack_to_fit([0, 1, 2, 3], emb, counts, max_tokens=24)

    assert [sorted(c) for c in chunks] in ([[0, 2], [1, 3]], [[1, 3], [0, 2]])


def test_pack_to_fit_is_deterministic():
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(8, 5))
    counts = [12] * 8
    first = raptor.pack_to_fit(list(range(8)), emb, counts, 40)
    second = raptor.pack_to_fit(list(range(8)), emb, counts, 40)
    assert first == second


def test_pack_to_fit_rejects_single_oversized_text():
    emb = np.eye(3)
    with pytest.raises(ValueError, match="over the 50-token"):
        raptor.pack_to_fit([0, 1, 2], emb, [10, 60, 10], max_tokens=50)


def test_split_to_fit_packs_small_groups_without_clustering(monkeypatch):
    monkeypatch.setattr(raptor, "cluster_embeddings", _no_llm)
    emb = np.random.default_rng(3).normal(size=(4, 3))

    groups = raptor.split_to_fit([0, 1, 2, 3], emb, [20] * 4, max_tokens=50)

    assert sorted(i for g in groups for i in g) == [0, 1, 2, 3]
    assert all(raptor._fits(g, [20] * 4, 50) for g in groups)


def test_split_to_fit_packs_when_bic_finds_one_cluster(monkeypatch):
    monkeypatch.setattr(raptor, "cluster_embeddings", lambda e: ([[0]] * len(e), 1))
    emb = np.random.default_rng(4).normal(size=(6, 3))
    counts = [20] * 6

    groups = raptor.split_to_fit(list(range(6)), emb, counts, max_tokens=50)

    assert sorted(i for g in groups for i in g) == list(range(6))
    assert all(raptor._fits(g, counts, 50) for g in groups)


def test_split_to_fit_keeps_fitting_subclusters_and_overlaps(monkeypatch):
    # Two sub-clusters; point 2 is soft-assigned to both.
    assignments = [[0], [0], [0, 1], [1], [1]]
    monkeypatch.setattr(raptor, "cluster_embeddings", lambda e: (assignments, 2))
    emb = np.random.default_rng(5).normal(size=(5, 3))
    idx = [10, 11, 12, 13, 14]
    counts = {i: 10 for i in idx}

    groups = raptor.split_to_fit(
        idx, np.vstack([np.zeros((10, 3)), emb]), counts, max_tokens=40
    )

    assert groups == [[10, 11, 12], [12, 13, 14]]


# ------------------------------------------------------- tree building
def test_build_tree_stops_when_bic_chooses_one_cluster(monkeypatch, offline, tmp_path):
    monkeypatch.setattr(raptor, "cluster_embeddings", lambda e: ([[0]] * len(e), 1))
    monkeypatch.setattr(raptor, "summarize_with_ollama", _no_llm)
    tree_path = str(tmp_path / "tree.json")

    nodes = raptor.build_raptor_tree(_chunks(6), FakeEmbeddings(), tree_path=tree_path)

    assert len(nodes) == 6
    assert all(n.metadata["level"] == 0 for n in nodes)
    assert nodes[0].metadata["leaf_titles"] == ["T0"]
    _, stats = raptor.load_tree(tree_path)
    assert stats["complete"] and stats["levels_done"] == 0
    assert stats["summary_calls"] == 0


def test_build_tree_drops_bad_summaries(monkeypatch, offline):
    def fake_cluster(e):
        return ([[0], [0], [1], [1]], 2) if len(e) == 4 else ([[0]] * len(e), 1)

    replies = iter(
        [
            "Please provide the text.",
            "A real summary of chunks two and three, long enough.",
        ]
    )
    monkeypatch.setattr(raptor, "cluster_embeddings", fake_cluster)
    monkeypatch.setattr(
        raptor, "summarize_with_ollama", lambda text, model: next(replies)
    )

    nodes = raptor.build_raptor_tree(_chunks(4), FakeEmbeddings())

    summaries = [n for n in nodes if n.metadata["level"] == 1]
    assert len(summaries) == 1
    assert summaries[0].metadata["id"] == "1-0"
    assert summaries[0].metadata["children"] == ["0-2", "0-3"]
    assert summaries[0].metadata["leaf_titles"] == ["T2", "T3"]


def test_build_tree_resumes_from_checkpoint_and_summary_cache(
    monkeypatch, offline, tmp_path
):
    chunks = _chunks(4)
    tree_path = str(tmp_path / "tree.json")
    cache_path = str(tmp_path / "tree_summaries.json")
    model = "fake-model"

    # A checkpoint written after level 0, as if the previous run crashed in level 1.
    leaves = [
        Document(
            page_content=c.page_content,
            metadata={
                "id": f"0-{i}",
                "level": 0,
                "cluster": None,
                "title": c.metadata["title"],
                "children": [],
                "leaf_titles": [c.metadata["title"]],
            },
        )
        for i, c in enumerate(chunks)
    ]
    raptor.save_tree(
        leaves,
        {
            "levels_done": 0,
            "complete": False,
            "summary_calls": 0,
            "summary_seconds": [],
            "build_seconds": 12.5,
        },
        tree_path,
    )

    # Both level-1 summaries already finished before the crash.
    def key(idx):
        combined = "\n\n".join(chunks[i].page_content for i in idx)
        return hashlib.sha1(
            f"{model}\n{raptor.SUMMARY_SYSTEM}\n{combined}".encode()
        ).hexdigest()

    cached = {
        key([0, 1]): {
            "summary": "Cached summary of the first two chunks of text.",
            "seconds": 3.0,
        },
        key([2, 3]): {
            "summary": "Cached summary of the last two chunks of text here.",
            "seconds": 4.0,
        },
    }
    with open(cache_path, "w") as f:
        json.dump(cached, f)

    def fake_cluster(e):
        return ([[0], [0], [1], [1]], 2) if len(e) == 4 else ([[0]] * len(e), 1)

    monkeypatch.setattr(raptor, "cluster_embeddings", fake_cluster)
    monkeypatch.setattr(raptor, "summarize_with_ollama", _no_llm)

    nodes = raptor.build_raptor_tree(
        chunks, FakeEmbeddings(), model=model, tree_path=tree_path
    )

    level1 = [n for n in nodes if n.metadata["level"] == 1]
    assert [n.page_content for n in level1] == [
        cached[key([0, 1])]["summary"],
        cached[key([2, 3])]["summary"],
    ]
    _, stats = raptor.load_tree(tree_path)
    assert stats["complete"] and stats["levels_done"] == 1
    assert stats["summary_calls"] == 2
    assert stats["summary_seconds"] == [3.0, 4.0]
    assert stats["build_seconds"] >= 12.5

    # A finished tree is returned as-is on the next run, without re-embedding.
    emb = FakeEmbeddings()
    again = raptor.build_raptor_tree(chunks, emb, model=model, tree_path=tree_path)
    assert emb.calls == 0
    assert [n.metadata["id"] for n in again] == [n.metadata["id"] for n in nodes]
