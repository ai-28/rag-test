from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _backend_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    return _backend_root().parent


def _default_pdf_path() -> Path:
    root = _repo_root()
    preferred = root / "State-of-Pre-Seed–2025-in-review.pdf"
    if preferred.exists():
        return preferred
    matches = sorted(root.glob("State-of-Pre-Seed*.pdf"))
    if matches:
        return matches[0]
    return preferred


@lru_cache
def get_settings() -> "Settings":
    return Settings()


class Settings:
    """Environment-driven configuration (no pydantic-settings dependency)."""

    def __init__(self) -> None:
        backend_root = _backend_root()
        repo_root = _repo_root()

        self.openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openrouter_base_url: str = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")

        self.openrouter_embedding_model: str = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
        self.openrouter_chat_model: str = os.getenv(
            "OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"
        )
        self.openrouter_vision_model: str = os.getenv(
            "OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini"
        )

        raw_pdf = os.getenv("RAG_PDF_PATH", "").strip()
        self.pdf_path: Path = (
            Path(raw_pdf).expanduser().resolve()
            if raw_pdf
            else _default_pdf_path()
        )
        if not self.pdf_path.is_absolute():
            self.pdf_path = (repo_root / self.pdf_path).resolve()

        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "").strip()
        self.chroma_persist_dir: Path = (
            Path(chroma_dir).expanduser().resolve()
            if chroma_dir
            else (backend_root / "data" / "chroma")
        )

        self.rag_chunk_chars: int = int(os.getenv("RAG_CHUNK_CHARS", "1600"))
        self.rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "16"))
        self.rag_collection_name: str = os.getenv("RAG_COLLECTION_NAME", "pdf_rag")

        self.embedding_batch_size: int = int(os.getenv("RAG_EMBED_BATCH_SIZE", "32"))

        self.rag_enable_page_vision: bool = _env_bool("RAG_PAGE_VISION", True)
        self.rag_page_vision_dpi: float = float(os.getenv("RAG_PAGE_VISION_DPI", "165"))
        self.rag_page_vision_max_edge_px: int = int(
            os.getenv("RAG_PAGE_VISION_MAX_EDGE_PX", "2048")
        )
        self.rag_vision_concurrency: int = int(os.getenv("RAG_VISION_CONCURRENCY", "3"))
        self.rag_vision_max_tokens: int = int(os.getenv("RAG_VISION_MAX_TOKENS", "8192"))
        self.rag_vision_temperature: float = float(
            os.getenv("RAG_VISION_TEMPERATURE", "0.1")
        )

        self.rag_hybrid_enabled: bool = _env_bool("RAG_HYBRID", True)
        self.rag_vector_candidates: int = int(os.getenv("RAG_VECTOR_CANDIDATES", "28"))
        self.rag_bm25_candidates: int = int(os.getenv("RAG_BM25_CANDIDATES", "28"))
        self.rag_rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
