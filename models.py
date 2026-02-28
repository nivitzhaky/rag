from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class IngestRequest(BaseModel):
    context_tag: Optional[str] = None


class QueryRequest(BaseModel):
    query_text: str
    k: int = 3
    min_relevance: float = 0.7
    context_tag: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    sources: list[str]


class IndexResponse(BaseModel):
    documents: int
    chunks: int
    collection: str
