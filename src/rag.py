"""Answer questions using retrieved chunks.

This file exposes both the simple answer helper and the higher-level agentic
RAG entry point used by the Streamlit app.
"""

import re
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import get_openai_api_key
from src.prompts import ANSWER_PROMPT, ANSWER_REPAIR_PROMPT


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
    answer = str(response.content)

    if _answer_uses_only_allowed_citations(answer, retrieved_chunks):
        return answer

    repair_prompt = ANSWER_REPAIR_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
        allowed_citations=_format_allowed_citations(retrieved_chunks),
    )
    repaired_response = llm.invoke(repair_prompt)
    repaired_answer = str(repaired_response.content)

    if _answer_uses_only_allowed_citations(repaired_answer, retrieved_chunks):
        return repaired_answer

    return "The answer is not found in the uploaded documents."


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


def _answer_uses_only_allowed_citations(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> bool:
    """Check that answer citations all point to retrieved chunks."""
    not_found_text = "The answer is not found in the uploaded documents."
    if not_found_text in answer:
        return True

    citations = _extract_citation_keys(answer)
    if not citations:
        return False

    allowed_citations = _allowed_citation_keys(retrieved_chunks)
    return citations.issubset(allowed_citations)


def _format_allowed_citations(retrieved_chunks: list[dict[str, Any]]) -> str:
    """List the exact citations the answer is allowed to use."""
    citations = []

    for source, page, chunk_id in sorted(_allowed_citation_keys(retrieved_chunks)):
        citations.append(f"[{source}, page {page}, chunk {chunk_id}]")

    return "\n".join(citations)


def _allowed_citation_keys(
    retrieved_chunks: list[dict[str, Any]],
) -> set[tuple[str, str, int]]:
    """Build comparable citation keys from retrieved chunk metadata."""
    allowed = set()

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        allowed.add(
            (
                str(metadata.get("source", "")).strip(),
                _normalize_page(metadata.get("page")),
                int(metadata.get("chunk_id", -1)),
            )
        )

    return allowed


def _extract_citation_keys(answer: str) -> set[tuple[str, str, int]]:
    """Parse citations like [file.pdf, page 1, chunk 2] from an answer."""
    citation_pattern = re.compile(
        r"\[(?P<source>.*?),\s*page\s*(?P<page>.*?),\s*chunk\s*(?P<chunk>\d+)\]",
        re.IGNORECASE,
    )
    citations = set()

    for match in citation_pattern.finditer(answer):
        citations.add(
            (
                match.group("source").strip(),
                _normalize_page(match.group("page")),
                int(match.group("chunk")),
            )
        )

    return citations


def _normalize_page(page: Any) -> str:
    """Normalize page values so generated citations can match metadata."""
    if page is None:
        return ""

    return str(page).strip()
