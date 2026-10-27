"""Step 3: find the chunks that answer a question.

We combine two kinds of search, because each one misses things the other finds:

- BM25 (keyword search) is good at exact words and numbers: "Item 1A", "130,497".
- Vector search (embeddings) is good at meaning: "money spent on research"
  matches "research and development expenses".

The two ranked lists are merged with Reciprocal Rank Fusion (RRF), then an optional
cross-encoder re-ranker reads each (question, chunk) pair and re-orders the top results.
"""

import json
import math
import re

import numpy as np
from rank_bm25 import BM25Okapi

from config import CANDIDATES, EMBEDDING_MODEL, INDEX_DIR, RERANK_MODEL, TOP_K
from rag.chunker import searchable_text

MODES = ["bm25", "vector", "hybrid", "hybrid+rerank"]

COMPANY_NAMES = {
    "NVIDIA": r"\bnvidia\b",
    "Qualcomm": r"\b(qualcomm|qcom)\b",
    "Arm": r"\barm\b",
}

# bge models expect this prefix on questions (not on the chunks).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    """Lowercase words, keeping numbers like '130,497' or '75.0' as one token."""
    return re.findall(r"\d[\d,.]*\d|\w+", text.lower())


def companies_in(question: str, available: list[str]) -> list[str]:
    """Which companies does the question mention? e.g. 'Compare Arm and NVIDIA' -> ['NVIDIA', 'Arm']."""
    return [name for name, pattern in COMPANY_NAMES.items()
            if name in available and re.search(pattern, question, re.IGNORECASE)]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge several ranked lists of ids. An id ranked high in any list ends up high.

    Each list gives an id the score 1 / (k + rank). Using ranks instead of raw scores
    means we never have to compare BM25 scores with cosine similarities.
    """
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridSearcher:
    def __init__(self, chunks, embeddings, embed_query, rerank=None):
        """
        chunks:      list of chunk dicts (see chunker.py)
        embeddings:  array with one normalised vector per chunk
        embed_query: function str -> normalised vector
        rerank:      optional function (question, [texts]) -> [scores]
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.embed_query = embed_query
        self.rerank = rerank
        self.bm25 = BM25Okapi([tokenize(searchable_text(c)) for c in chunks])
        self.companies = sorted({c["company"] for c in chunks})

    def _rank(self, question: str, allowed: np.ndarray, mode: str) -> list[int]:
        """Return chunk indices, best first, searching only where `allowed` is True."""
        def best(scores):
            return [int(i) for i in np.argsort(-scores) if allowed[i]][:CANDIDATES]

        bm25_ranking = best(self.bm25.get_scores(tokenize(question)))
        if mode == "bm25":
            return bm25_ranking

        vector_ranking = best(self.embeddings @ self.embed_query(question))
        if mode == "vector":
            return vector_ranking

        fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking])[:CANDIDATES]
        if mode == "hybrid" or self.rerank is None:
            return fused

        # hybrid+rerank: the cross-encoder scores each candidate against the question
        scores = self.rerank(question, [searchable_text(self.chunks[i]) for i in fused])
        return [i for _, i in sorted(zip(scores, fused), reverse=True)]

    def search(self, question: str, k: int = TOP_K, mode: str = "hybrid+rerank") -> list[dict]:
        """Return the k best chunks. If the question names several companies,
        each company is searched separately so none of them gets crowded out."""
        names = companies_in(question, self.companies) or [None]
        per_company = max(3, math.ceil(k / len(names))) if len(names) > 1 else k

        results = []
        for name in names:
            allowed = np.array([name is None or c["company"] == name for c in self.chunks])
            results += [self.chunks[i] for i in self._rank(question, allowed, mode)[:per_company]]
        return results


# ---- Loading the real models and the saved index -------------------------------------

def load_embedder():
    """Load the local embedding model. Returns (embed_chunks, embed_query) functions."""
    from sentence_transformers import SentenceTransformer  # slow import, so only when needed
    model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_chunks(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    def embed_query(text):
        return model.encode(QUERY_PREFIX + text, normalize_embeddings=True)

    return embed_chunks, embed_query


def load_reranker():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(RERANK_MODEL)
    return lambda question, texts: model.predict([(question, t) for t in texts])


def load_searcher(index_dir=INDEX_DIR, use_reranker: bool = True) -> HybridSearcher:
    """Load the index written by build_index.py."""
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    embeddings = np.load(index_dir / "embeddings.npy")
    _, embed_query = load_embedder()
    rerank = load_reranker() if use_reranker else None
    return HybridSearcher(chunks, embeddings, embed_query, rerank)
