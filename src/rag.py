"""Answer questions using retrieved chunks.

This file exposes both the simple answer helper and the higher-level agentic
RAG entry point used by the Streamlit app.
"""

from typing import Any

from langchain_openai import ChatOpenAI

from src.config import get_openai_api_key
from src.prompts import ANSWER_PROMPT


CHAT_MODEL = "gpt-4o-mini"


def answer_question(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    openai_api_key: str | None = None,
) -> str:
    """Generate a citation-grounded answer from retrieved chunks."""
    if not retrieved_chunks:
        return "The answer is not found in the uploaded documents."

    api_key = get_openai_api_key(openai_api_key)

    if not api_key:
        raise ValueError("OpenAI API key is missing. Add one in the sidebar first.")

    context = _format_chunks_for_context(retrieved_chunks)
    prompt = ANSWER_PROMPT.format(question=question, context=context)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=api_key)
    response = llm.invoke(prompt)

    return str(response.content)


def run_agentic_rag(
    question: str,
    openai_api_key: str | None = None,
) -> dict[str, Any]:
    """Run the LangGraph retrieve -> grade -> answer loop."""
    from src.graph import build_agentic_rag_graph

    graph = build_agentic_rag_graph()
    initial_state = {
        "user_question": question,
        "current_query": question,
        "attempts": 0,
        "retrieved_chunks": [],
        "decision": {},
        "answer": "",
        "trace": [],
        "openai_api_key": openai_api_key,
    }
    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "trace": final_state.get("trace", []),
    }


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
                f"Citation: [{source}, page {page}, chunk {chunk_id}]\n"
                f"Source filename: {source}\n"
                f"Page: {page}\n"
                f"Chunk: {chunk_id}\n"
                f"Text:\n{text}"
            )
        )

    return "\n\n---\n\n".join(formatted_chunks)
