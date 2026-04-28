"""Store and retrieve chunks with ChromaDB.

This module is the retrieval milestone only. It embeds chunks, saves them in a
persistent local Chroma database, and retrieves similar chunks for a question.
It does not call an LLM or generate answers yet.
"""

from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from src.config import CHROMA_DIR


COLLECTION_NAME = "agentic_rag_tutor"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_function() -> OpenAIEmbeddingFunction:
    """Create the OpenAI embedding function used by Chroma."""
    load_dotenv()

    # Chroma's OpenAIEmbeddingFunction reads OPENAI_API_KEY when this env var
    # name is supplied. Keeping the key in .env avoids hard-coding secrets.
    return OpenAIEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        api_key_env_var="OPENAI_API_KEY",
    )


def index_chunks(chunks: list[dict[str, Any]]) -> int:
    """Save embedded chunks in the persistent Chroma collection.

    Returns the number of non-empty chunks sent to Chroma.
    """
    collection = _get_collection()

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

    # upsert lets a beginner click the button more than once without duplicate
    # ID errors. Existing chunks with the same ID are replaced.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return len(documents)


def retrieve_chunks(query: str, k: int = 8) -> list[dict[str, Any]]:
    """Retrieve the top matching chunks for a question."""
    if not query.strip():
        return []

    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
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


def _get_collection():
    """Open the persistent Chroma collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


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
