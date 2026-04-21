from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ._logging import get_logger
from .db import init_db
from .routers import documents

get_logger("pii")  # ensure the handler is attached at import time

app = FastAPI(title="PII Redactor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(documents.router)
