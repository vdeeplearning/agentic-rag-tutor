"""Shared data shapes used across the app."""

from typing import Any, TypedDict


class AgentState(TypedDict):
    """State passed between LangGraph nodes.

    TypedDict keeps the graph beginner-friendly: it is just a dictionary with
    expected keys, but editors can still help us catch spelling mistakes.
    """

    user_question: str
    current_query: str
    attempts: int
    retrieved_chunks: list[dict[str, Any]]
    decision: dict[str, Any]
    answer: str
    trace: list[dict[str, Any]]
    openai_api_key: str | None
