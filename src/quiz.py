"""Quiz generation and grading helpers.

Quiz Mode uses the same retrieved chunks as the RAG answer flow. The LLM is
only allowed to create and grade questions from those chunks, so quizzes stay
grounded in the uploaded documents.
"""

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import get_openai_api_key
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
- The score must be an integer from 0 to 100, where 100 is fully correct.
- Set is_correct to true only when score is 70 or higher.
- Use this partial-credit rubric:
  - 90-100: complete, accurate, and clearly supported by the source chunks.
  - 70-89: mostly correct, with only minor missing detail or wording issues.
  - 40-69: partially correct, but missing important details or containing some
    confusion.
  - 1-39: minimally related, with major gaps or mostly incorrect content.
  - 0: incorrect, blank, unsupported, or not answerable from the source chunks.
- Cite the source chunk used with this format:
  [<source filename>, page <page>, chunk <chunk_id>]

Quiz question:
{question}

Student answer:
{user_answer}

Source chunks:
{context}
""".strip()


def generate_quiz_question(
    topic: str | None = None,
    openai_api_key: str | None = None,
    source_filters: list[str] | None = None,
) -> dict[str, Any]:
    """Generate one quiz question from retrieved document chunks."""
    query = topic.strip() if topic and topic.strip() else GENERAL_QUIZ_QUERY
    source_chunks = retrieve_chunks(
        query,
        k=6,
        openai_api_key=openai_api_key,
        source_filters=source_filters,
    )

    if not source_chunks:
        return {
            "question": "No indexed document chunks were found for a quiz.",
            "expected_answer": "",
            "source_chunks": [],
        }

    context = _format_chunks_for_context(source_chunks)
    prompt = QUIZ_GENERATION_PROMPT.format(context=context)
    response = _get_chat_model(openai_api_key).invoke(prompt)
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
    openai_api_key: str | None = None,
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
    response = _get_chat_model(openai_api_key).invoke(prompt)
    parsed = _parse_json_response(str(response.content))

    score = _normalize_score(parsed.get("score", 0))

    return {
        "score": score,
        "is_correct": score >= 70,
        "feedback": str(parsed.get("feedback", "")),
        "ideal_answer": str(parsed.get("ideal_answer", "")),
        "citation": str(parsed.get("citation", "")),
        "cited_chunks": _find_cited_chunks(
            str(parsed.get("citation", "")),
            source_chunks,
        ),
    }


def _get_chat_model(openai_api_key: str | None = None) -> ChatOpenAI:
    """Create the chat model used for quiz generation and grading."""
    api_key = get_openai_api_key(openai_api_key)

    if not api_key:
        raise ValueError("OpenAI API key is missing. Add one in the sidebar first.")

    return ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=api_key)


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


def _normalize_score(raw_score: Any) -> int:
    """Convert common LLM score formats into a 0-100 integer.

    Sometimes models return 1 for a fully correct answer, meaning 1.0. Since the
    UI displays points out of 100, values between 0 and 1 are treated as ratios.
    """
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0

    if 0 <= score <= 1:
        score = score * 100

    score = max(0, min(100, score))
    return int(round(score))


def _find_cited_chunks(
    citation_text: str,
    source_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only chunks referenced by the grader's citation string."""
    cited_chunks: list[dict[str, Any]] = []
    cited_keys = set()

    citation_pattern = re.compile(
        r"\[(?P<source>.*?),\s*page\s*(?P<page>.*?),\s*chunk\s*(?P<chunk>\d+)\]",
        re.IGNORECASE,
    )

    for match in citation_pattern.finditer(citation_text):
        cited_keys.add(
            (
                match.group("source").strip(),
                _normalize_page(match.group("page")),
                int(match.group("chunk")),
            )
        )

    for chunk in source_chunks:
        metadata = chunk.get("metadata", {})
        chunk_key = (
            str(metadata.get("source", "")).strip(),
            _normalize_page(metadata.get("page")),
            int(metadata.get("chunk_id", -1)),
        )

        if chunk_key in cited_keys:
            cited_chunks.append(chunk)

    return cited_chunks


def _normalize_page(page: Any) -> str:
    """Normalize page values so citation text can match Chroma metadata."""
    if page is None:
        return ""

    return str(page).strip()
