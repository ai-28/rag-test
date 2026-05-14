from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from rag_api.config import get_settings


def get_collection() -> Collection:
    s = get_settings()
    s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(s.chroma_persist_dir))
    return client.get_or_create_collection(
        name=s.rag_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> Collection:
    s = get_settings()
    s.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(s.chroma_persist_dir))
    try:
        client.delete_collection(s.rag_collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=s.rag_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def collection_count(collection: Collection) -> int:
    return int(collection.count())


def upsert_chunks(
    collection: Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    if not ids:
        return
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_similar(
    collection: Collection,
    query_embedding: list[float],
    n_results: int,
) -> dict[str, Any]:
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
