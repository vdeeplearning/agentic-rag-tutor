"""Quiz generation and grading helpers.

Quiz Mode uses the same retrieved chunks as the RAG answer flow. The LLM is
only allowed to create and grade questions from those chunks, so quizzes stay
grounded in the uploaded documents.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.rag import _format_chunks_for_context
from src.vectorstore import retrieve_chunks


CHAT_MODEL = "gpt-4o-mini"
GENERAL_QUIZ_QUERY = "key concepts main ideas important definitions"


QUIZ_GENERATION_PROMPT = """
You are creating one beginner-friendly quiz question from uploaded document
chunks.

Use only the chunks below. Do not use outside knowledge.

Return only valid JSON with this exact shape:
{{
  "question": "...",
  "expected_answer": "..."
}}

Rules:
- Ask one clear question.
- The expected_answer must be supported by the chunks.
- Keep the question useful for learning, not tricky.

Retrieved chunks:
{context}
""".strip()


QUIZ_GRADING_PROMPT = """
You are grading a student's answer using only the source chunks below.

Return only valid JSON with this exact shape:
{{
  "score": 0,
  "is_correct": false,
  "feedback": "...",
  "ideal_answer": "...",
  "citation": "[source, page X, chunk Y]"
}}

Rules:
- Do not grade using outside knowledge.
- If the source chunks do not support the quiz question, say so in feedback.
- Keep feedback encouraging and concise.
- Cite the source chunk used with this format:
  [<source filename>, page <page>, chunk <chunk_id>]

Quiz question:
{question}

Student answer:
{user_answer}

Source chunks:
{context}
""".strip()


def generate_quiz_question(topic: str | None = None) -> dict[str, Any]:
    """Generate one quiz question from retrieved document chunks."""
    query = topic.strip() if topic and topic.strip() else GENERAL_QUIZ_QUERY
    source_chunks = retrieve_chunks(query, k=6)

    if not source_chunks:
        return {
            "question": "No indexed document chunks were found for a quiz.",
            "expected_answer": "",
            "source_chunks": [],
        }

    context = _format_chunks_for_context(source_chunks)
    prompt = QUIZ_GENERATION_PROMPT.format(context=context)
    response = _get_chat_model().invoke(prompt)
    parsed = _parse_json_response(str(response.content))

    return {
        "question": str(parsed.get("question", "")),
        "expected_answer": str(parsed.get("expected_answer", "")),
        "source_chunks": source_chunks,
    }


def grade_quiz_answer(
    question: str,
    user_answer: str,
    source_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Grade a quiz answer using only the chunks that created the question."""
    if not source_chunks:
        return {
            "score": 0,
            "is_correct": False,
            "feedback": "I could not grade this because no source chunks were available.",
            "ideal_answer": "",
            "citation": "",
        }

    context = _format_chunks_for_context(source_chunks)
    prompt = QUIZ_GRADING_PROMPT.format(
        question=question,
        user_answer=user_answer,
        context=context,
    )
    response = _get_chat_model().invoke(prompt)
    parsed = _parse_json_response(str(response.content))

    return {
        "score": int(parsed.get("score", 0)),
        "is_correct": bool(parsed.get("is_correct", False)),
        "feedback": str(parsed.get("feedback", "")),
        "ideal_answer": str(parsed.get("ideal_answer", "")),
        "citation": str(parsed.get("citation", "")),
    }


def _get_chat_model() -> ChatOpenAI:
    """Create the chat model used for quiz generation and grading."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Add it to a .env file first.")

    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from the model and return a safe fallback on errors."""
    cleaned_content = content.strip()

    if cleaned_content.startswith("```"):
        cleaned_content = cleaned_content.strip("`").strip()
        if cleaned_content.startswith("json"):
            cleaned_content = cleaned_content[4:].strip()

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        return {}
