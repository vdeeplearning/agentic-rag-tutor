"""Answer questions using retrieved chunks.

This file intentionally stays small for the current milestone:
retrieve chunks -> format them as context -> ask the LLM for a cited answer.
LangGraph, query rewriting, and more advanced agent behavior can come later.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.prompts import ANSWER_PROMPT


CHAT_MODEL = "gpt-4o-mini"


def answer_question(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """Generate a citation-grounded answer from retrieved chunks."""
    if not retrieved_chunks:
        return "The answer is not found in the uploaded documents."

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Add it to a .env file first.")

    context = _format_chunks_for_context(retrieved_chunks)
    prompt = ANSWER_PROMPT.format(question=question, context=context)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    response = llm.invoke(prompt)

    return str(response.content)


def _format_chunks_for_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Turn retrieved chunks into plain text the model can cite."""
    formatted_chunks: list[str] = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id", "unknown")
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        formatted_chunks.append(
            (
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Text:\n{text}"
            )
        )

    return "\n\n---\n\n".join(formatted_chunks)
