from __future__ import annotations

from dataclasses import dataclass

from rag_api.pdf_extract import PageText


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    page_number: int  # 1-based for humans
    chunk_index: int
    text: str
    kind: str = "page_text"


def chunk_pages(
    pages: list[PageText],
    max_chars: int,
    overlap: int,
) -> list[TextChunk]:
    """Split long page text into overlapping windows."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = max(0, min(overlap, max_chars - 1))

    chunks: list[TextChunk] = []
    for page in pages:
        raw = (page.text or "").strip()
        if not raw:
            continue
        page_no = page.page_index + 1
        start = 0
        part = 0
        while start < len(raw):
            end = min(len(raw), start + max_chars)
            piece = raw[start:end].strip()
            if piece:
                cid = f"p{page_no:04d}_c{part:04d}"
                chunks.append(
                    TextChunk(
                        chunk_id=cid,
                        page_number=page_no,
                        chunk_index=part,
                        text=piece,
                    )
                )
                part += 1
            if end >= len(raw):
                break
            start = max(0, end - overlap)
    return chunks


def chunk_markdown_blob(
    raw: str,
    page_number: int,
    id_prefix: str,
    max_chars: int,
    overlap: int,
    kind: str,
) -> list[TextChunk]:
    """Split a long vision/Markdown summary into overlapping retrieval chunks."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = max(0, min(overlap, max_chars - 1))

    chunks: list[TextChunk] = []
    start = 0
    part = 0
    while start < len(raw):
        end = min(len(raw), start + max_chars)
        piece = raw[start:end].strip()
        if piece:
            cid = f"{id_prefix}_c{part:04d}"
            chunks.append(
                TextChunk(
                    chunk_id=cid,
                    page_number=page_number,
                    chunk_index=part,
                    text=piece,
                    kind=kind,
                )
            )
            part += 1
        if end >= len(raw):
            break
        start = max(0, end - overlap)
    return chunks
