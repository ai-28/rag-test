from __future__ import annotations

import json
from typing import Any, AsyncIterator, cast

from rag_api.chunks import TextChunk, chunk_markdown_blob, chunk_pages
from rag_api.config import get_settings
from rag_api.openrouter import embed_texts_batched, stream_chat_completion
from rag_api.page_vision import describe_all_pages
from rag_api.pdf_extract import extract_pdf_pages
from rag_api.retrieval import clear_retrieval_cache, hybrid_retrieve
from rag_api.store import (
    collection_count,
    get_collection,
    query_similar,
    reset_collection,
    upsert_chunks,
)


async def ingest_pdf() -> dict[str, Any]:
    s = get_settings()
    if not s.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    pages = extract_pdf_pages(s.pdf_path)
    text_chunks: list[TextChunk] = chunk_pages(
        pages, s.rag_chunk_chars, s.rag_chunk_overlap
    )

    visual_chunks: list[TextChunk] = []
    vision_pages = 0
    if s.rag_enable_page_vision:
        vision_pages = len(pages)
        page_descriptions = await describe_all_pages(s.pdf_path, len(pages))
        for page_no, md in page_descriptions:
            if not (md or "").strip():
                continue
            visual_chunks.extend(
                chunk_markdown_blob(
                    md,
                    page_no,
                    f"p{page_no:04d}_viz",
                    s.rag_chunk_chars,
                    s.rag_chunk_overlap,
                    "page_visual",
                )
            )

    all_chunks: list[TextChunk] = text_chunks + visual_chunks

    def _sort_key(c: TextChunk) -> tuple[int, int, int]:
        kind_pri = 0 if c.kind == "page_text" else 1
        return (c.page_number, kind_pri, c.chunk_index)

    all_chunks = sorted(all_chunks, key=_sort_key)

    if not all_chunks:
        raise RuntimeError(
            "No content to index: no extractable text and page vision produced nothing."
        )

    texts = [c.text for c in all_chunks]
    embeddings = await embed_texts_batched(texts, s.embedding_batch_size)

    collection = reset_collection()
    ids = [c.chunk_id for c in all_chunks]
    metadatas: list[dict[str, Any]] = [
        {
            "chunk_id": c.chunk_id,
            "page": c.page_number,
            "chunk_index": c.chunk_index,
            "kind": c.kind,
            "source_pdf": s.pdf_path.name,
        }
        for c in all_chunks
    ]
    upsert_chunks(collection, ids, texts, embeddings, metadatas)
    clear_retrieval_cache()

    return {
        "ok": True,
        "pdf": str(s.pdf_path),
        "pages": len(pages),
        "text_chunks": len(text_chunks),
        "visual_chunks": len(visual_chunks),
        "vision_pages_processed": vision_pages if s.rag_enable_page_vision else 0,
        "total_chunks": len(all_chunks),
        "collection": s.rag_collection_name,
        "page_vision_enabled": s.rag_enable_page_vision,
        "hybrid_retrieval": s.rag_hybrid_enabled,
    }


def rag_status() -> dict[str, Any]:
    s = get_settings()
    collection = get_collection()
    return {
        "collection": s.rag_collection_name,
        "chroma_dir": str(s.chroma_persist_dir),
        "chunk_count": collection_count(collection),
        "pdf_path": str(s.pdf_path),
        "page_vision_enabled": s.rag_enable_page_vision,
        "hybrid_retrieval": s.rag_hybrid_enabled,
    }


def _format_context_block(results: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    docs = (results.get("documents") or [[]])[0] or []
    metas = (results.get("metadatas") or [[]])[0] or []
    citations: list[dict[str, Any]] = []
    parts: list[str] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        page = meta.get("page") if isinstance(meta, dict) else None
        cid = meta.get("chunk_id") if isinstance(meta, dict) else None
        kind = meta.get("kind") if isinstance(meta, dict) else None
        label = f"[{i + 1}] page={page} kind={kind or '?'}"
        parts.append(f"{label}\n{doc}")
        citations.append(
            {"index": i + 1, "page": page, "chunk_id": cid, "kind": kind}
        )
    return "\n\n".join(parts), citations


def _normalize_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Keep only user/assistant turns with non-empty content (max 40 turns)."""
    if not history:
        return []
    out: list[dict[str, str]] = []
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": cast(str, role), "content": content})
    if len(out) > 40:
        out = out[-40:]
    return out


def _retrieval_query(user_message: str, history: list[dict[str, str]]) -> str:
    """Use recent dialogue + latest question so follow-ups retrieve relevant chunks."""
    if not history:
        return user_message
    parts: list[str] = []
    for turn in history[-6:]:
        c = turn["content"][:900]
        parts.append(f"{turn['role']}: {c}")
    parts.append(f"user: {user_message}")
    return "\n".join(parts)[:4000]


async def stream_rag_answer(
    user_message: str,
    *,
    history: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Yield NDJSON lines: {"type":"context"|"token"|"done"|"error", ...}."""
    s = get_settings()
    if not s.openrouter_api_key:
        yield json.dumps({"type": "error", "message": "OPENROUTER_API_KEY is not set"})
        return

    collection = get_collection()
    if collection_count(collection) == 0:
        yield json.dumps(
            {
                "type": "error",
                "message": "Index is empty. Run ingest first (CLI or POST /api/rag/ingest).",
            }
        )
        return

    hist = _normalize_history(history)
    retrieval_q = _retrieval_query(user_message, hist)
    q_emb = (await embed_texts_batched([retrieval_q], 1))[0]
    if s.rag_hybrid_enabled:
        raw = hybrid_retrieve(
            collection,
            query_text=retrieval_q,
            query_embedding=q_emb,
        )
    else:
        raw = query_similar(collection, q_emb, s.rag_top_k)
    context_block, citations = _format_context_block(raw)

    yield json.dumps({"type": "context", "citations": citations})

    system = (
        "You are a careful assistant in a multi-turn chat about a PDF. "
        "Use earlier user and assistant turns only to interpret follow-ups, pronouns, and "
        "what the user means. Every factual claim about the document must be supported by "
        "the CONTEXT in the **latest** user message below (not from memory of prior CONTEXT "
        "blocks, which are not repeated here). "
        "The context may mix page_text (extracted text) and page_visual "
        "(Markdown from rendered pages, including charts, maps, and tables). "
        "Treat page_visual as primary when it contains numbers for chart questions. "
        "page_visual may include chart-read estimates for line charts (cells prefixed with ~ "
        "or small ranges); treat those as the best available reading from the figure. "
        "If the answer is not in that CONTEXT, say you do not have enough information. "
        "When you use facts, cite the bracketed reference numbers (e.g. [1], [2]). "
        "Do not invent numbers; if figures conflict, prefer explicit tables in page_visual "
        "and note uncertainty briefly. "
        "Chart transcriptions may contain OCR-like errors; if a value is ambiguous, say so."
    )
    latest_user = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{user_message}"

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(hist)
    messages.append({"role": "user", "content": latest_user})

    async for token in stream_chat_completion(messages):
        yield json.dumps({"type": "token", "content": token})

    yield json.dumps({"type": "done"})
