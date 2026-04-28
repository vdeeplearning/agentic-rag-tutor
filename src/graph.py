"""Agentic RAG workflow built with LangGraph.

The graph adds one important behavior on top of basic RAG:
it asks the LLM whether the retrieved chunks are good enough before answering.
If they are related but weak, the graph retries with a rewritten query.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.prompts import EVIDENCE_GRADER_PROMPT
from src.rag import _format_chunks_for_context, answer_question
from src.schemas import AgentState
from src.vectorstore import retrieve_chunks


MAX_ATTEMPTS = 3
CHAT_MODEL = "gpt-4o-mini"
NOT_FOUND_ANSWER = "The answer is not found in the uploaded documents."


def build_agentic_rag_graph():
    """Create and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("grade_evidence", _grade_evidence)
    workflow.add_node("answer", _answer)
    workflow.add_node("not_enough_evidence", _not_enough_evidence)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_evidence")
    workflow.add_conditional_edges(
        "grade_evidence",
        _choose_next_step,
        {
            "retrieve": "retrieve",
            "answer": "answer",
            "not_enough_evidence": "not_enough_evidence",
        },
    )
    workflow.add_edge("answer", END)
    workflow.add_edge("not_enough_evidence", END)

    return workflow.compile()


def _retrieve(state: AgentState) -> dict[str, Any]:
    """Retrieve chunks for the current query and record the attempt."""
    retrieved_chunks = retrieve_chunks(state["current_query"])
    attempts = state["attempts"] + 1
    trace = list(state["trace"])

    trace.append(
        {
            "attempt": attempts,
            "query": state["current_query"],
            "retrieved_chunks": retrieved_chunks,
            "decision": "",
            "reason": "",
            "rewritten_query": "",
        }
    )

    return {
        "attempts": attempts,
        "retrieved_chunks": retrieved_chunks,
        "trace": trace,
    }


def _grade_evidence(state: AgentState) -> dict[str, Any]:
    """Ask the LLM whether the retrieved chunks are enough to answer."""
    decision = _normalize_grader_decision(_call_evidence_grader(state), state)
    trace = list(state["trace"])

    if trace:
        trace[-1]["decision"] = decision["decision"]
        trace[-1]["reason"] = decision["reason"]
        trace[-1]["rewritten_query"] = decision["rewritten_query"]

    updates: dict[str, Any] = {
        "decision": decision,
        "trace": trace,
    }

    if decision["decision"] == "retry" and state["attempts"] < MAX_ATTEMPTS:
        updates["current_query"] = decision["rewritten_query"] or state["current_query"]

    return updates


def _answer(state: AgentState) -> dict[str, str]:
    """Generate the final citation-grounded answer."""
    return {
        "answer": answer_question(
            state["user_question"],
            state["retrieved_chunks"],
        )
    }


def _not_enough_evidence(state: AgentState) -> dict[str, str]:
    """Return a clear answer when retrieval did not find enough evidence."""
    return {"answer": NOT_FOUND_ANSWER}


def _choose_next_step(state: AgentState) -> str:
    """Route the graph based on the grader decision."""
    decision = state["decision"].get("decision", "stop")

    if decision == "answer":
        return "answer"

    if decision == "retry" and state["attempts"] < MAX_ATTEMPTS:
        return "retrieve"

    return "not_enough_evidence"


def _call_evidence_grader(state: AgentState) -> dict[str, str]:
    """Call the LLM grader and normalize its JSON decision."""
    context = _format_chunks_for_context(state["retrieved_chunks"])
    prompt = EVIDENCE_GRADER_PROMPT.format(
        question=state["user_question"],
        query=state["current_query"],
        attempts=state["attempts"],
        max_attempts=MAX_ATTEMPTS,
        context=context,
    )

    llm = _get_chat_model()
    response = llm.invoke(prompt)
    return _parse_grader_json(str(response.content))


def _normalize_grader_decision(
    decision: dict[str, str],
    state: AgentState,
) -> dict[str, str]:
    """Enforce retry-before-stop behavior for early weak evidence.

    The LLM grader can still stop early when it clearly says the chunks are
    completely unrelated and a rewrite is unlikely to help. Otherwise, attempts
    1 and 2 keep searching.
    """
    if decision["decision"] == "retry" and state["attempts"] >= MAX_ATTEMPTS:
        return {
            "decision": "stop",
            "reason": (
                f"{decision['reason']} The graph stopped because attempt "
                f"{state['attempts']} reached the maximum of {MAX_ATTEMPTS}."
            ),
            "rewritten_query": "",
        }

    if decision["decision"] != "stop" or state["attempts"] >= MAX_ATTEMPTS:
        return decision

    reason = decision["reason"].lower()
    can_stop_early = "unrelated" in reason and "unlikely" in reason

    if can_stop_early:
        return decision

    rewritten_query = decision["rewritten_query"] or _fallback_rewritten_query(state)

    return {
        "decision": "retry",
        "reason": (
            f"{decision['reason']} The graph is retrying because this is attempt "
            f"{state['attempts']} of {MAX_ATTEMPTS}, and weak or incomplete "
            "evidence should be retried before stopping."
        ),
        "rewritten_query": rewritten_query,
    }


def _fallback_rewritten_query(state: AgentState) -> str:
    """Create a simple rewrite if the grader asked to stop without one."""
    return f"{state['user_question']} key supporting facts from uploaded documents"


def _get_chat_model() -> ChatOpenAI:
    """Create the chat model used by the evidence grader."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Add it to a .env file first.")

    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def _parse_grader_json(content: str) -> dict[str, str]:
    """Parse the grader JSON and fall back safely if it is malformed."""
    cleaned_content = content.strip()

    if cleaned_content.startswith("```"):
        cleaned_content = cleaned_content.strip("`").strip()
        if cleaned_content.startswith("json"):
            cleaned_content = cleaned_content[4:].strip()

    try:
        parsed = json.loads(cleaned_content)
    except json.JSONDecodeError:
        return {
            "decision": "stop",
            "reason": "The evidence grader did not return valid JSON.",
            "rewritten_query": "",
        }

    decision = parsed.get("decision", "stop")

    if decision not in {"answer", "retry", "stop"}:
        decision = "stop"

    return {
        "decision": decision,
        "reason": str(parsed.get("reason", "")),
        "rewritten_query": str(parsed.get("rewritten_query", "")),
    }
