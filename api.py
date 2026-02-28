import json
from fastapi import FastAPI, HTTPException
from fastapi import File, Query, UploadFile
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from create_database import (
    load_documents,
    split_text,
    save_to_pgvector,
    set_context_tag,
)
from models import QueryRequest, QueryResponse, IndexResponse
from query_data import PROMPT_TEMPLATE
from vector_store import create_vector_store, get_collection_name, extract_content_from_bytes


app = FastAPI(title="RAG API")


@app.post("/index", response_model=IndexResponse)
def index_documents(
    file: UploadFile = File(...),
    metadata_json: str | None = Query(default=None),
    reset_collection: bool = Query(default=False),
    context_tag: str | None = Query(default=None),
):
    try:
        base_metadata = {}
        if metadata_json:
            loaded_metadata = json.loads(metadata_json)
            if not isinstance(loaded_metadata, dict):
                raise HTTPException(status_code=400, detail="metadata_json must be a JSON object.")
            base_metadata = loaded_metadata
        resolved_source = file.filename or "api_document"
        if "source" not in base_metadata:
            base_metadata["source"] = resolved_source
        content = extract_content_from_bytes(file.file.read(), resolved_source)
        documents = [Document(page_content=content, metadata=base_metadata)]
        chunks = split_text(documents)
        chunks = set_context_tag(chunks, context_tag)
        save_to_pgvector(chunks, pre_delete_collection=reset_collection)
        return IndexResponse(
            documents=len(documents),
            chunks=len(chunks),
            collection=get_collection_name(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        db = create_vector_store(OpenAIEmbeddings())
        query_filter = {"context_tag": request.context_tag} if request.context_tag else None
        results = db.similarity_search_with_relevance_scores(
            request.query_text,
            k=request.k,
            filter=query_filter,
        )
        if len(results) == 0 or results[0][1] < request.min_relevance:
            raise HTTPException(status_code=404, detail="Unable to find matching results.")
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text, question=request.query_text
        )
        response_text = ChatOpenAI().invoke(prompt).content
        sources = [doc.metadata.get("source", "") for doc, _score in results]
        return QueryResponse(response=response_text, sources=sources)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
