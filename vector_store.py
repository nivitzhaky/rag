from __future__ import annotations

import os
from io import BytesIO
import psycopg
from fastapi import HTTPException
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from pypdf import PdfReader

load_dotenv()


def get_pgvector_connection() -> str:
    direct = os.getenv("PGVECTOR_CONNECTION")
    if direct:
        return direct
    host = os.environ["POSTGRES_HOST"]
    port = os.environ["POSTGRES_PORT"]
    database = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def get_collection_name() -> str:
    return os.getenv("PGVECTOR_COLLECTION", "default")


def get_psycopg_connection() -> str:
    return get_pgvector_connection().replace("postgresql+psycopg://", "postgresql://", 1)


def ensure_context_tag_index():
    with psycopg.connect(get_psycopg_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DO $$
                BEGIN
                    IF to_regclass('public.langchain_pg_embedding') IS NOT NULL THEN
                        CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_context_tag
                        ON langchain_pg_embedding ((cmetadata->>'context_tag'));
                    END IF;
                END $$;
                """
            )
        conn.commit()


def create_vector_store(embeddings: OpenAIEmbeddings | None = None) -> PGVector:
    vector_embeddings = embeddings or OpenAIEmbeddings()
    store = PGVector(
        embeddings=vector_embeddings,
        collection_name=get_collection_name(),
        connection=get_pgvector_connection(),
        use_jsonb=True,
    )
    ensure_context_tag_index()
    return store


def create_vector_store_from_documents(documents, pre_delete_collection: bool = False) -> PGVector:
    store = PGVector.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        collection_name=get_collection_name(),
        connection=get_pgvector_connection(),
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )
    ensure_context_tag_index()
    return store


def extract_content_from_bytes(raw_bytes: bytes, source: str) -> str:
    if source.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n".join(pages).strip()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded PDF has no extractable text.")
        return content
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Uploaded file must be UTF-8 text or PDF.")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return content
