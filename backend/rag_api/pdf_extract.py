from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageText:
    page_index: int  # 0-based
    text: str


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        pages: list[PageText] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append(PageText(page_index=i, text=text))
        return pages
    finally:
        doc.close()


def render_page_png_bytes(
    pdf_path: Path,
    page_index: int,
    *,
    dpi: float,
    max_edge_px: int,
) -> bytes:
    """Rasterize one PDF page to PNG bytes (RGB, no alpha). Scales down if too large."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if max_edge_px < 320:
        raise ValueError("max_edge_px too small for reliable vision OCR")
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        rect = page.rect
        # Nominal pixel size at dpi
        scale0 = dpi / 72.0
        pw = rect.width * scale0
        ph = rect.height * scale0
        m = max(pw, ph)
        shrink = 1.0 if m <= max_edge_px else max_edge_px / m
        z = scale0 * shrink
        mat = fitz.Matrix(z, z)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()
