"""Hybrid dense-vector + BM25 retrieval with reciprocal rank fusion (RRF).

Pairs dense embeddings (semantic) with lexical BM25 so exact tokens (names,
quarters, amounts) are less likely to be missed. RRF merges ranked lists without
normalizing incompatible score scales — common production pattern for document RAG.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from rag_api.config import get_settings
from rag_api.store import query_similar

_TOKEN_RE = re.compile(
    r"\$[\d,.]+[kmbKMB]?|[\d,.]+%|\b\d{4}\b|[a-z0-9]{2,}",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


_bm25_key: tuple[str, int] | None = None
_bm25: BM25Okapi | None = None
_chunk_ids: list[str] = []
_chunk_docs: list[str] = []
_chunk_metas: list[dict[str, Any]] = []


def clear_retrieval_cache() -> None:
    """Invalidate after ingest so BM25 matches Chroma."""
    global _bm25_key, _bm25, _chunk_ids, _chunk_docs, _chunk_metas
    _bm25_key = None
    _bm25 = None
    _chunk_ids = []
    _chunk_docs = []
    _chunk_metas = []


def _ensure_lexical_index(collection: Collection) -> None:
    global _bm25_key, _bm25, _chunk_ids, _chunk_docs, _chunk_metas
    s = get_settings()
    n = int(collection.count())
    key = (s.rag_collection_name, n)
    if _bm25_key == key and _bm25 is not None and len(_chunk_ids) == n:
        return

    data = collection.get(include=["documents", "metadatas"])
    ids = list(data.get("ids") or [])
    docs = list(data.get("documents") or [])
    metas = list(data.get("metadatas") or [])
    paired = sorted(zip(ids, docs, metas), key=lambda x: x[0])
    _chunk_ids = [p[0] for p in paired]
    _chunk_docs = [str(p[1] or "") for p in paired]
    _chunk_metas = [dict(p[2] or {}) for p in paired]
    if not _chunk_ids:
        _bm25 = None
        _bm25_key = key
        return
    tokenized = [tokenize(d) for d in _chunk_docs]
    _bm25 = BM25Okapi(tokenized)
    _bm25_key = key


def _rrf_scores(ranked_ids: list[str], *, rrf_k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(ranked_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def hybrid_retrieve(
    collection: Collection,
    *,
    query_text: str,
    query_embedding: list[float],
) -> dict[str, Any]:
    """
    Chroma-compatible shape: each value is a list containing one inner list,
    matching `collection.query` for `_format_context_block`.
    """
    s = get_settings()
    _ensure_lexical_index(collection)

    vec_k = max(s.rag_top_k, s.rag_vector_candidates)
    bm25_k = max(s.rag_top_k, s.rag_bm25_candidates)

    vec_raw = query_similar(collection, query_embedding, vec_k)
    vec_ids: list[str] = list((vec_raw.get("ids") or [[]])[0] or [])

    id_to_doc = dict(zip(_chunk_ids, _chunk_docs))
    id_to_meta = dict(zip(_chunk_ids, _chunk_metas))

    merged: dict[str, float] = defaultdict(float)

    if vec_ids:
        for cid, sc in _rrf_scores(vec_ids, rrf_k=s.rag_rrf_k).items():
            merged[cid] += sc

    if _bm25 is not None and query_text.strip():
        q_tokens = tokenize(query_text)
        if q_tokens:
            scores = _bm25.get_scores(q_tokens)
            ranked_idx = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:bm25_k]
            bm25_ids = [_chunk_ids[i] for i in ranked_idx if i < len(_chunk_ids)]
            for cid, sc in _rrf_scores(bm25_ids, rrf_k=s.rag_rrf_k).items():
                merged[cid] += sc

    if not merged:
        return vec_raw

    final_ids = sorted(merged.keys(), key=lambda i: merged[i], reverse=True)[
        : s.rag_top_k
    ]
    out_docs = [id_to_doc.get(i, "") for i in final_ids]
    out_metas = [id_to_meta.get(i, {}) for i in final_ids]
    return {
        "ids": [final_ids],
        "documents": [out_docs],
        "metadatas": [out_metas],
        "distances": [[]],
    }
