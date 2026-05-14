from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_api.routers.rag import router as rag_router

app = FastAPI(title="Test API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
