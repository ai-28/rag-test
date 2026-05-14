# next-fastapi-test

PDF **RAG chat**: **Next.js** (App Router) UI, **FastAPI** API, **OpenRouter** for embeddings, vision (ingest), and chat, **Chroma** for local vector search. Ingest builds `page_text` + `page_visual` chunks from your PDF.

For architecture, env vars, and tuning, see **[docs/RAG_IMPLEMENTATION.md](docs/RAG_IMPLEMENTATION.md)**.

## Prerequisites

- **Python 3.10+** (3.9 often works)
- **Node.js 20+**
- An [OpenRouter](https://openrouter.ai/) API key
- Default PDF at repo root: `State-of-Pre-Seed–2025-in-review.pdf` (or set `RAG_PDF_PATH` in `backend/.env`)

## 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create **`backend/.env`** (copy from `backend/.env.example`) and set at least:

```env
OPENROUTER_API_KEY=sk-or-...
```

**Build the vector index** (vision + embeddings; can take a few minutes):

```bash
cd backend
source .venv/bin/activate
python -m rag_api.ingest_cli
```

**Run the API** (must run from the `backend` directory so `main:app` resolves):

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

- API: [http://127.0.0.1:8000](http://127.0.0.1:8000)  
- Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- RAG status: [http://127.0.0.1:8000/api/rag/status](http://127.0.0.1:8000/api/rag/status)

From the **repo root** instead:

```bash
backend/.venv/bin/uvicorn main:app --app-dir backend --reload --port 8000
```

## 2. Frontend

In **`frontend/.env.local`** or **`frontend/.env`**:

```env
RAG_API_BASE_URL=http://127.0.0.1:8000
```

Then:

```bash
cd frontend
npm install
npm run dev
```

Open **[http://localhost:3000](http://localhost:3000)** — the RAG chat is the home page. The app calls the Next **`/api/chat`** route, which proxies to your FastAPI `RAG_API_BASE_URL`.

## 3. Order

1. Start **backend** on port **8000** (after ingest).  
2. Start **frontend** with `RAG_API_BASE_URL` pointing at that host.

## Project layout

| Path | Role |
|------|------|
| `backend/` | FastAPI app, RAG ingest, Chroma under `backend/data/chroma` by default |
| `frontend/` | Next.js UI and `/api/chat` proxy |
| `docs/RAG_IMPLEMENTATION.md` | Full implementation and deployment notes |

## Security and GitHub

- **`backend/.env`** and **`frontend/.env`** hold secrets and are listed in **`.gitignore`** (root + `backend/.gitignore`). They should **not** be pushed. Use **`backend/.env.example`** as a template only.
- The **README** uses a fake placeholder (`sk-or-...`), not a real key.
- If a real key was ever committed or pasted into a ticket, **rotate it** in the OpenRouter dashboard.

Before your first push, run `git status` and confirm you do **not** see `.env` files staged.
