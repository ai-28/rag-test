# RAG chatbot implementation guide

This repo implements **Next.js (App Router) + FastAPI + OpenRouter + local Chroma**—no hosted vector database.

## Architecture (current)

**Ingest — caption-and-index (multimodal PDF without multimodal embeddings):**

1. **`page_text`** — PyMuPDF extracts selectable text per page, chunked with overlap (`rag_api/pdf_extract.py`, `rag_api/chunks.py`).
2. **`page_visual`** — Each page is rasterized (DPI + max edge capped), sent to a **vision** model on OpenRouter, and the reply is stored as Markdown chunks (`rag_api/page_vision.py`). This is how **charts, maps, and tables drawn as graphics** become searchable text.
3. Both kinds are embedded with the configured **OpenRouter embedding model** and upserted into **Chroma** (`rag_api/store.py`, `rag_api/service.py`).

**Query — hybrid retrieval + multi-turn chat:**

1. **Retrieval** — By default, **dense vectors (Chroma)** and **BM25** over the same documents run in parallel; results are merged with **RRF** (`rag_api/retrieval.py`). The query string for hybrid search and the embedding input are a **retrieval query**: the latest user message, or (if `history` is sent) the **last few turns plus the latest message** concatenated (`_retrieval_query` in `rag_api/service.py`) so short follow-ups (“what about Q3?”) still match the right chunks.
2. **Generation** — OpenRouter **chat** model streams tokens. The prompt is **multi-turn**: `system` → prior **`history`** (`user` / `assistant` pairs, plain text only) → final **`user`** message that contains **only this turn’s** `CONTEXT` block (retrieved chunks) and `QUESTION` (`rag_api/service.py`). Earlier turns are for **interpretation** only; the system instructs the model to **ground facts in the latest CONTEXT**, not in memory of old context blocks.
3. **Transport** — `POST /api/rag/chat` returns **NDJSON** lines: `context` (citations), `token`, `done`, or `error`. The Next **route handler** proxies the body stream (`frontend/src/app/api/chat/route.ts`).

**Industry note:** stacks like this are pattern **(a)** “VLM → text → dense (+ lexical)”; alternatives are unified multimodal embeddings or late-interaction image retrievers (e.g. ColPali-style). This project stays on OpenRouter + local Chroma.

**Accuracy:** VLMs and chat models cannot honestly promise **bit-perfect** numbers on every chart. Vision Markdown may use **`~`** for **chart-read** values (line points without printed labels). Add a **small gold Q/A set** if you need measured numeric quality.

---

## Page vision (figures and line charts)

Prompts live in `rag_api/page_vision.py`. Besides titles, axes, legends, and verbatim printed numbers, the vision system prompt requires **line / curve charts** to produce a **Markdown table with one row per X-axis tick** and one column per legend series. Values not printed on the figure must be read from the plot against the Y axis and marked with **`~`** (or a tight range); multi-panel charts get a table per panel.

Re-**ingest** after changing `OPENROUTER_VISION_MODEL`, DPI, or prompt text.

---

## Chat API contract

`POST /api/rag/chat` — JSON body (`rag_api/routers/rag.py`):

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Current user question (required, 1–8000 chars). |
| `history` | array (optional) | Prior turns only: `{ "role": "user" \| "assistant", "content": "..." }`, max **48** entries from client; server keeps last **40** non-empty turns. **Must not** include the current `message`. |

Streaming response: `Content-Type: application/x-ndjson` (or charset variant); one JSON object per line.

---

## Prerequisites

- Python **3.10+** recommended (**3.9** works with current models; use `Optional[...]` / `List[...]` typing where needed).
- Node **20+** for the frontend.
- [OpenRouter](https://openrouter.ai/) API key.
- Vision: a **vision-capable** chat model on OpenRouter (default `openai/gpt-4o-mini`). See [OpenRouter image inputs](https://openrouter.ai/docs/guides/overview/multimodal/images).

---

## Environment variables

### Backend (`backend/.env`)

Create from `backend/.env.example` (never commit real keys).

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Bearer token for OpenRouter. |
| `OPENROUTER_BASE_URL` | No | Default `https://openrouter.ai/api/v1`. |
| `OPENROUTER_EMBEDDING_MODEL` | No | Default `openai/text-embedding-3-small`. |
| `OPENROUTER_CHAT_MODEL` | No | Default `openai/gpt-4o-mini` (streaming). |
| `OPENROUTER_VISION_MODEL` | No | Default `openai/gpt-4o-mini` (per-page ingest). |
| `RAG_PDF_PATH` | No | PDF path (default: repo root `State-of-Pre-Seed–2025-in-review.pdf` or first `State-of-Pre-Seed*.pdf`). |
| `CHROMA_PERSIST_DIR` | No | Default `backend/data/chroma`. |
| `RAG_CHUNK_CHARS` | No | Default `1600` (text + vision Markdown). |
| `RAG_CHUNK_OVERLAP` | No | Default `200`. |
| `RAG_TOP_K` | No | Default `16` (chunks after fusion). |
| `RAG_HYBRID` | No | Default `1` — dense + BM25 + RRF. `0` = vector only. |
| `RAG_VECTOR_CANDIDATES` | No | Default `28`. |
| `RAG_BM25_CANDIDATES` | No | Default `28`. |
| `RAG_RRF_K` | No | Default `60`. |
| `RAG_PAGE_VISION` | No | Default `1`. `0` = text-only ingest. |
| `RAG_PAGE_VISION_DPI` | No | Default `165`. |
| `RAG_PAGE_VISION_MAX_EDGE_PX` | No | Default `2048`. |
| `RAG_VISION_CONCURRENCY` | No | Default `3`. |
| `RAG_VISION_MAX_TOKENS` | No | Default `8192`. |
| `RAG_VISION_TEMPERATURE` | No | Default `0.1`. |

### Frontend

| Variable | Description |
|----------|-------------|
| `RAG_API_BASE_URL` | FastAPI origin, e.g. `http://127.0.0.1:8000`. Read in `frontend/src/app/api/chat/route.ts` (server-side proxy). |

Set in **`frontend/.env.local`** or **`frontend/.env`** (Next loads both; do not commit secrets).

---

## Install and ingest

**Backend**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Ingest** (rebuilds the collection; use CLI to avoid HTTP timeouts on large PDFs):

```bash
cd backend
source .venv/bin/activate
python -m rag_api.ingest_cli
```

Optional HTTP ingest: `curl -X POST http://127.0.0.1:8000/api/rag/ingest`

**Check index:** `GET http://127.0.0.1:8000/api/rag/status` — expect `chunk_count` combining `page_text` and `page_visual`.

---

## Run services

**API** (run from `backend` so `main:app` resolves):

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

From repo root:

```bash
backend/.venv/bin/uvicorn main:app --app-dir backend --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

**UI:** **`http://localhost:3000`** — the **RAG chat** is the home page (`frontend/src/app/page.tsx` → `RagChat`). **`/chat`** redirects to **`/`**. The UI posts to **`/api/chat`** with `message` and optional `history` for threaded conversation.

---

## Verify

- Ask factual questions; use a **follow-up** that depends on the first answer (tests `history` + retrieval query).
- Ask **chart** questions (bars, maps, **line** series by quarter).
- If retrieval misses, raise `RAG_TOP_K` or `RAG_*_CANDIDATES`, or tune chunk sizes.

---

## Tuning

- **Cost / latency:** lower `RAG_PAGE_VISION_DPI`, `RAG_VISION_CONCURRENCY`, or `RAG_PAGE_VISION=0`.
- **Vision quality:** stronger `OPENROUTER_VISION_MODEL`, higher DPI / `RAG_PAGE_VISION_MAX_EDGE_PX`, then re-ingest.
- **Embeddings:** changing `OPENROUTER_EMBEDDING_MODEL` requires a **full re-ingest**.

---

## Troubleshooting

| Symptom | Check |
|--------|--------|
| `401` from OpenRouter | API key; restart uvicorn. |
| Ingest HTTP timeout | Use `python -m rag_api.ingest_cli`. |
| Vision failures | Logs; failed pages may get a short failure placeholder in Markdown. |
| Empty index | Run ingest; `GET /api/rag/status`. |
| Next proxy errors | `RAG_API_BASE_URL`; API must be running. |
| “Could not import module main” | Start uvicorn with cwd `backend` or `--app-dir backend`. |

---

## Code map

| Area | Location |
|------|-----------|
| FastAPI entry | `backend/main.py` |
| Settings | `backend/rag_api/config.py` |
| OpenRouter (embed, stream chat, vision completion) | `backend/rag_api/openrouter.py` |
| PDF text + page PNG raster | `backend/rag_api/pdf_extract.py` |
| Chunking | `backend/rag_api/chunks.py` |
| Hybrid BM25 + RRF | `backend/rag_api/retrieval.py` |
| Page vision prompts + concurrency | `backend/rag_api/page_vision.py` |
| Chroma | `backend/rag_api/store.py` |
| Ingest, retrieval query, multi-turn chat | `backend/rag_api/service.py` |
| HTTP routes (`/rag/status`, `/rag/ingest`, `/rag/chat`) | `backend/rag_api/routers/rag.py` |
| Ingest CLI | `backend/rag_api/ingest_cli.py` |
| Chat UI component | `frontend/src/components/rag-chat.tsx` |
| Home page | `frontend/src/app/page.tsx` |
| Legacy `/chat` route | `frontend/src/app/chat/page.tsx` (redirects to `/`) |
| Next.js BFF proxy | `frontend/src/app/api/chat/route.ts` |
