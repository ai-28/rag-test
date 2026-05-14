"""Per-page raster vision via OpenRouter (ingest-time only).

Best practice for graphics-heavy PDFs: render each page at bounded DPI/resolution,
send image + structured extraction prompt, embed returned Markdown as `page_visual`
chunks alongside PyMuPDF text chunks. See OpenRouter multimodal docs (image_url
with data URI).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

from rag_api.config import get_settings
from rag_api.openrouter import chat_completion
from rag_api.pdf_extract import render_page_png_bytes

log = logging.getLogger(__name__)

VISION_SYSTEM = """You are a document analyst extracting visible content from ONE page image of a data report (e.g. Carta, venture funding). Your Markdown will be chunked and embedded for semantic search (RAG).

Output Markdown only. No preamble or closing remarks.

Include, when visible:
1. Titles and subtitles (verbatim when readable).
2. Figure type (stacked bar, line chart, map, table, etc.).
3. Axes, units, legends, and time ranges.
4. All numeric labels, percentages, and dollar amounts shown on the chart or map. Prefer GitHub-flavored Markdown tables for repeated series (e.g. by quarter).
5. Footnotes, asterisks, and source lines visible in the image.
6. For maps: state/region labels with values as printed.

**Line / curve charts (critical for RAG):** The axes and units are visible, but **most points have no text label**—only the stroke. Users will ask for **any** X tick (e.g. a single quarter), so you must **materialize** values for retrieval.
- Before the table, state **X-axis** (what each tick is, e.g. quarter), **Y-axis** (metric + **units**, e.g. percent, $M), and **value range** shown on Y (min/max tick labels if visible).
- Build a **Markdown table** with **one row per X-axis tick** shown on the chart (every quarter, month, etc.—do not skip ticks because the curve has no callout there).
- **One column per series** from the legend (color + line style if that distinguishes series, e.g. solid vs dotted).
- **Printed** numbers on the chart: copy **verbatim**.
- **Unprinted** points: treat the curve as the data. At each X tick, find where the series intersects the vertical grid (or interpolate between adjacent plotted segments), project horizontally to the **Y axis**, and record that reading with prefix `~` (e.g. `~73%`) or a **tight range** if the stroke falls between gridlines. **Do not leave cells blank** for a visible series; use `~` chart-read or `[unreadable]` only when geometry is truly unclear.
- **Multi-panel** line charts: repeat (axes summary + full table) **per panel** with the panel’s own X ticks.

Rules:
- Do not invent **printed** labels that are not on the page; axis-read `~` values must follow visible geometry.
- Do not add narrative interpretation, causal claims, or editorial trends beyond what the figure title/subtitle states verbatim.
- If the page is nearly blank, say: _[blank or minimal content]_"""


def _user_prompt(page_number: int, total_pages: int, pdf_name: str) -> str:
    return (
        f"Source file: {pdf_name}\n"
        f"Page {page_number} of {total_pages} (1-based).\n\n"
        "Extract all content from this page image following the system rules. "
        "If you see a line or curve chart, end with a Markdown table: one row per X tick, "
        "one column per legend series; use ~ where the value is read from the curve against "
        "the Y axis (not printed as text on the chart)."
    )


async def describe_page(
    pdf_path: Path,
    page_index: int,
    total_pages: int,
) -> tuple[int, str]:
    """Return (1-based page number, markdown description)."""
    s = get_settings()
    page_no = page_index + 1
    try:
        png = await asyncio.to_thread(
            render_page_png_bytes,
            pdf_path,
            page_index,
            dpi=s.rag_page_vision_dpi,
            max_edge_px=s.rag_page_vision_max_edge_px,
        )
        b64 = base64.b64encode(png).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        messages = [
            {"role": "system", "content": VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _user_prompt(page_no, total_pages, pdf_path.name),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        text = await chat_completion(
            messages,
            model=s.openrouter_vision_model,
            max_tokens=s.rag_vision_max_tokens,
            temperature=s.rag_vision_temperature,
        )
        return (page_no, (text or "").strip())
    except Exception as exc:  # noqa: BLE001 — log and continue per page
        log.exception("Vision ingest failed for page %s", page_no)
        return (
            page_no,
            f"_[vision extraction failed for page {page_no}: {exc}]_",
        )


async def describe_all_pages(pdf_path: Path, page_count: int) -> list[tuple[int, str]]:
    """Run vision on every page with bounded concurrency."""
    if page_count <= 0:
        return []

    s = get_settings()
    sem = asyncio.Semaphore(max(1, s.rag_vision_concurrency))

    async def worker(pi: int) -> tuple[int, str]:
        async with sem:
            return await describe_page(pdf_path, pi, page_count)

    results = await asyncio.gather(*[worker(i) for i in range(page_count)])
    return sorted(results, key=lambda t: t[0])
