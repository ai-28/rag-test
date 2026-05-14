from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from rag_api.config import get_settings


def _headers() -> dict[str, str]:
    s = get_settings()
    if not s.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return {
        "Authorization": f"Bearer {s.openrouter_api_key}",
        "Content-Type": "application/json",
        # Optional attribution for OpenRouter dashboards
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "next-fastapi-rag",
    }


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input string (same order)."""
    if not texts:
        return []
    s = get_settings()
    url = f"{s.openrouter_base_url}/embeddings"
    payload = {"model": s.openrouter_embedding_model, "input": texts}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
    items: list[dict[str, Any]] = data.get("data") or []
    # OpenAI format: data[i].embedding; ensure sorted by index
    items.sort(key=lambda x: x.get("index", 0))
    return [list(map(float, it["embedding"])) for it in items]


async def stream_chat_completion(
    messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    """Yield decoded text deltas from OpenRouter (OpenAI-compatible SSE)."""
    s = get_settings()
    url = f"{s.openrouter_base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": s.openrouter_chat_model,
        "messages": messages,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", url, headers=_headers(), json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    chunk = line.removeprefix("data: ").strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece


async def embed_texts_batched(texts: list[str], batch_size: int) -> list[list[float]]:
    out: list[list[float]] = []
    s = get_settings()
    bs = batch_size or s.embedding_batch_size
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        out.extend(await embed_texts(batch))
    return out


async def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Non-streaming chat completion (used for vision / structured extract)."""
    s = get_settings()
    url = f"{s.openrouter_base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model or s.openrouter_chat_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    # Some providers return a list of parts
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""
