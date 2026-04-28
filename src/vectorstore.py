"""Store and retrieve chunks with ChromaDB and OpenAI embeddings.

This module is the retrieval milestone only. It embeds chunks, saves them in a
persistent local Chroma database, and retrieves similar chunks for a question.
It does not call an LLM or generate answers yet.
"""

from typing import Any

import chromadb
from langchain_openai import OpenAIEmbeddings

from src.config import CHROMA_DIR, get_openai_api_key


COLLECTION_NAME = "agentic_rag_tutor"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_vectorstore():
    """Open the persistent Chroma collection used by the app."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _get_embeddings(openai_api_key: str | None = None) -> OpenAIEmbeddings:
    """Create the OpenAI embedding model.

    A UI-provided key is preferred, with .env as a local development fallback.
    """
    api_key = get_openai_api_key(openai_api_key)

    if not api_key:
        raise ValueError("OpenAI API key is missing. Add one in the sidebar first.")

    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)


def index_chunks(
    chunks: list[dict[str, Any]],
    openai_api_key: str | None = None,
) -> int:
    """Save embedded chunks in the persistent Chroma collection.

    Returns the number of non-empty chunks sent to Chroma.
    """
    collection = get_vectorstore()
    embeddings = _get_embeddings(openai_api_key)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        metadata = dict(chunk.get("metadata", {}))

        if not text:
            continue

        ids.append(_chunk_id(metadata))
        documents.append(text)
        metadatas.append(_to_chroma_metadata(metadata))

    if not documents:
        return 0

    document_embeddings = embeddings.embed_documents(documents)

    # upsert lets a beginner click the button more than once without duplicate
    # ID errors. Existing chunks with the same ID are replaced.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=document_embeddings,
    )

    return len(documents)


def retrieve_chunks(
    query: str,
    k: int = 8,
    openai_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top matching chunks for a question."""
    if not query.strip():
        return []

    collection = get_vectorstore()
    embeddings = _get_embeddings(openai_api_key)
    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks: list[dict[str, Any]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        retrieved_chunks.append(
            {
                "text": text,
                "metadata": _from_chroma_metadata(metadata),
                "distance": distance,
            }
        )

    return retrieved_chunks


def _chunk_id(metadata: dict[str, Any]) -> str:
    """Build a stable Chroma ID from chunk metadata."""
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id", "0")

    return f"{source}:page-{page}:chunk-{chunk_id}"


def _to_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert app metadata into values Chroma can store."""
    page = metadata.get("page")

    return {
        "source": str(metadata.get("source", "")),
        "page": "" if page is None else page,
        "chunk_id": int(metadata.get("chunk_id", 0)),
    }


def _from_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert Chroma metadata back into the app's simple metadata shape."""
    page = metadata.get("page")

    return {
        "source": metadata.get("source"),
        "page": None if page == "" else page,
        "chunk_id": metadata.get("chunk_id"),
    }
