"""Split extracted documents into overlapping text chunks.

For now, chunking is character-based because it is easy to understand and
works well enough for a first RAG prototype. Later, this can be replaced with
token-aware chunking from LangChain or another text splitter.
"""


def chunk_documents(
    documents: list[dict[str, object]],
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
) -> list[dict[str, object]]:
    """Turn extracted documents into smaller overlapping chunks.

    Args:
        documents: Dictionaries with source, text, and page keys.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters repeated between neighboring chunks.

    Returns:
        A list of chunk dictionaries with text and metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[dict[str, object]] = []

    for document in documents:
        text = str(document.get("text", "")).strip()

        if not text:
            continue

        source = document.get("source")
        page = document.get("page")
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "page": page,
                        "chunk_id": len(chunks),
                    },
                }
            )

            if end >= len(text):
                break

            start = end - chunk_overlap

    return chunks
