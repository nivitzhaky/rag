# RAG API with PGVector

Minimal retrieval-augmented generation (RAG) service built with FastAPI, LangChain, OpenAI embeddings/chat, and PostgreSQL + pgvector.

## What this project does

- Indexes text or PDF content into a PGVector collection.
- Stores chunk metadata (including optional `context_tag`) for filtered retrieval.
- Answers questions by retrieving relevant chunks and sending them to an LLM.
- Supports both API-based ingestion (`/index`) and script-based ingestion (`create_database.py`).

## Project structure

- `api.py` FastAPI app with `/index` and `/query` endpoints.
- `models.py` request and response models used by the API.
- `vector_store.py` PGVector connection helpers, collection setup, and file-content extraction.
- `create_database.py` bulk ingestion script for markdown files in `data/books`.
- `query_data.py` CLI query script using similarity search + LLM response.
- `docker-compose.yml` app + pgvector Postgres services.
- `init.sql` creates the `vector` extension in Postgres.

## Prerequisites

- Python 3.11+ (recommended)
- Docker and Docker Compose (for container workflow)
- OpenAI API key

## Environment variables

Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_openai_key
PGVECTOR_COLLECTION=default

# Optional if not using PGVECTOR_CONNECTION directly
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ragdb
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpass
```

Notes:
- In Docker Compose, `PGVECTOR_CONNECTION` is already set for the app service.
- If `PGVECTOR_CONNECTION` is present, it is used first.

## Run with Docker (recommended)

1. Start services:

```bash
docker compose up --build
```

2. API will be available at:

```text
http://localhost:8000
```

3. Open interactive docs:

```text
http://localhost:8000/docs
```

Details:
- Postgres runs from `pgvector/pgvector:pg16`.
- `init.sql` is mounted into `/docker-entrypoint-initdb.d/` so `CREATE EXTENSION vector` runs on first database initialization.
- App startup runs `create_vector_store()` before launching Uvicorn, so the collection/index path is initialized early.

## Run locally (without Docker)

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure a Postgres database with pgvector extension is available.

3. Start API:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Ingestion and query flows

### API ingestion (`/index`)

- Method: `POST`
- Content type: `multipart/form-data`
- Required file field: `file`
- Optional query params:
  - `metadata_json` JSON object as a string
  - `reset_collection` boolean (default `false`)
  - `context_tag` string

Example:

```bash
curl -X POST "http://localhost:8000/index?reset_collection=true&context_tag=book&metadata_json=%7B%22source%22%3A%22alice_in_wonderland.md%22%7D" \
  -F "file=@data/alice_in_wonderland.md" \
  -H "accept: application/json"
```

### API query (`/query`)

- Method: `POST`
- Content type: `application/json`
- Body:
  - `query_text` string
  - `k` int (default `3`)
  - `min_relevance` float (default `0.7`)
  - `context_tag` optional string for metadata filtering

Example:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Who is Alice?",
    "k": 3,
    "min_relevance": 0.7,
    "context_tag": "book"
  }'
```

## Script-based workflows

### Bulk index markdown files

Indexes `data/books/*.md` and resets collection:

```bash
python create_database.py
```

Ensure your markdown files are under `data/books` when using this script.

### Query from CLI

```bash
python query_data.py "What happens to Alice?"
```

## Testing

Run tests:

```bash
pytest -q
```

Current tests include:
- `test_models.py` for API data models
- `test_vector_store.py` for file-content extraction behavior

## Troubleshooting

- `Unable to find matching results`: lower `min_relevance` or index more data.
- `Uploaded file must be UTF-8 text or PDF`: upload UTF-8 text or a PDF with extractable text.
- Connection errors: verify Postgres is reachable and env vars match actual credentials.