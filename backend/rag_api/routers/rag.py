from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_api.service import ingest_pdf, rag_status, stream_rag_answer

router = APIRouter(prefix="/rag", tags=["rag"])


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=12000)


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: Optional[List[HistoryTurn]] = Field(
        default=None,
        max_length=48,
        description="Prior turns only (user/assistant), excluding the current `message`.",
    )


@router.get("/status")
async def status():
    return rag_status()


@router.post("/ingest")
async def ingest():
    return await ingest_pdf()


@router.post("/chat")
async def chat(body: ChatBody):
    hist = (
        [{"role": t.role, "content": t.content} for t in body.history]
        if body.history
        else None
    )

    async def ndjson_stream():
        async for line in stream_rag_answer(body.message, history=hist):
            yield line + "\n"

    return StreamingResponse(
        ndjson_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
