import html
import re

import streamlit as st

from src.chunking import chunk_documents
from src.config import get_openai_api_key
from src.dedup import compute_file_hash, is_file_indexed, mark_file_indexed
from src.ingest import extract_text_from_file, save_uploaded_file
from src.quiz import generate_quiz_question, grade_quiz_answer
from src.rag import run_agentic_rag
from src.vectorstore import delete_chunks_for_source, index_chunks


def _show_api_settings():
    """Let users provide their own OpenAI key for this session."""
    with st.sidebar:
        st.header("API Settings")
        entered_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Your key is used only for this session and is not stored.",
        )

        if entered_key:
            st.session_state.openai_api_key = entered_key


def _get_active_openai_api_key():
    """Return the session key first, then the local .env fallback."""
    return get_openai_api_key(st.session_state.get("openai_api_key"))


def _warn_missing_api_key():
    """Show one consistent warning when an OpenAI key is needed."""
    st.warning("Please enter an OpenAI API key in the sidebar first.")


def _format_page(page):
    """Show missing page numbers in a friendly way."""
    return page or "N/A"


def _format_distance(distance):
    """Show Chroma distance values consistently."""
    if isinstance(distance, (int, float)):
        return f"{distance:.4f}"

    return "N/A"


def _preview_text(text, limit=3000):
    """Trim long chunks so the UI stays easy to scan."""
    if len(text) <= limit:
        return text

    return f"{text[:limit]}..."


def _extract_citation_keys(text):
    """Parse citations like [file.pdf, page 2, chunk 1]."""
    citation_pattern = re.compile(
        r"\[(?P<source>.*?),\s*page\s*(?P<page>.*?),\s*chunk\s*(?P<chunk>\d+)\]",
        re.IGNORECASE,
    )
    citations = set()

    for match in citation_pattern.finditer(text):
        citations.add(
            (
                match.group("source").strip(),
                str(match.group("page")).strip(),
                int(match.group("chunk")),
            )
        )

    return citations


def _chunk_key(chunk):
    """Create a comparable key from chunk metadata."""
    metadata = chunk.get("metadata", {})
    return (
        str(metadata.get("source", "")).strip(),
        str(metadata.get("page", "")).strip(),
        int(metadata.get("chunk_id", -1)),
    )


def _find_cited_chunks_in_trace(answer, trace):
    """Find only chunks cited by the final answer."""
    cited_keys = _extract_citation_keys(answer)
    cited_chunks = []
    seen_keys = set()

    for attempt in trace:
        for chunk in attempt.get("retrieved_chunks", []):
            key = _chunk_key(chunk)
            if key in cited_keys and key not in seen_keys:
                cited_chunks.append(chunk)
                seen_keys.add(key)

    return cited_chunks


def _highlight_relevant_excerpt(chunk_text, answer_text):
    """Show the full cited chunk and highlight answer terms when possible."""
    clean_answer = re.sub(r"\[[^\]]+\]", " ", answer_text)
    stop_words = {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "page",
        "chunk",
        "speaks",
        "following",
    }
    terms = []

    for term in re.findall(r"[A-Za-z][A-Za-z0-9+-]{1,}", clean_answer):
        normalized = term.lower()
        is_short_acronym = len(term) <= 3 and term.isupper()

        if (
            (len(term) >= 4 or is_short_acronym)
            and normalized not in stop_words
            and normalized not in terms
        ):
            terms.append(normalized)

    escaped_excerpt = html.escape(chunk_text)

    for term in sorted(terms, key=len, reverse=True):
        escaped_term = re.escape(html.escape(term))
        escaped_excerpt = re.sub(
            escaped_term,
            lambda match: f"<mark>{match.group(0)}</mark>",
            escaped_excerpt,
            flags=re.IGNORECASE,
        )

    return escaped_excerpt


def _show_trace_summary(trace):
    """Display a short summary of the agent loop."""
    total_attempts = len(trace)
    final_decision = trace[-1]["decision"] if trace else "none"
    answer_found = final_decision == "answer"

    st.info(
        f"Agent summary: {total_attempts} attempt(s) | "
        f"final decision: {final_decision.upper()} | "
        f"answer found: {'Yes' if answer_found else 'No'}"
    )


def _show_retrieved_chunk(index, chunk, evidence_text=None):
    """Display one retrieved chunk with readable metadata."""
    metadata = chunk["metadata"]
    source = metadata.get("source", "unknown")
    page = _format_page(metadata.get("page"))
    chunk_id = metadata.get("chunk_id", "unknown")
    distance = _format_distance(chunk.get("distance"))
    text = chunk.get("text", "")

    st.markdown(f"**Chunk {index}**")
    st.write(f"Source: {source}")
    st.write(f"Page: {page}")
    st.write(f"Chunk ID: {chunk_id}")
    st.write(f"Distance: {distance}")

    if evidence_text:
        st.markdown("**Relevant chunk**")
        st.markdown(
            _highlight_relevant_excerpt(text, evidence_text),
            unsafe_allow_html=True,
        )
    else:
        st.write(_preview_text(text))


def _show_agent_trace(trace):
    """Render the full per-attempt agent trace."""
    st.subheader("Agent Trace")

    if not trace:
        st.write("No agent trace yet.")
        return

    _show_trace_summary(trace)

    for attempt in trace:
        decision = attempt["decision"] or "no decision"
        title = f"Attempt {attempt['attempt']} - {decision.upper()}"

        with st.expander(title):
            st.write(f"Retrieval query used: {attempt['query']}")
            st.write(f"LLM decision: {decision}")
            st.write(f"LLM reason: {attempt['reason']}")

            if decision == "retry" and attempt["rewritten_query"]:
                st.write(f"Rewritten query: {attempt['rewritten_query']}")

            retrieved_count = len(attempt.get("retrieved_chunks", []))
            st.write(f"Retrieved chunks: {retrieved_count}")
            st.caption("Only cited evidence is shown outside the trace.")


def _show_cited_chunks(cited_chunks, evidence_text):
    """Show only the chunks cited by an answer or quiz evaluation."""
    st.subheader("Cited Evidence")

    if not cited_chunks:
        st.write("No cited chunk could be matched to the retrieved source chunks.")
        return

    for index, chunk in enumerate(cited_chunks, start=1):
        _show_retrieved_chunk(index, chunk, evidence_text=evidence_text)

        if index < len(cited_chunks):
            st.divider()


def _show_indexing_controls():
    """Shared upload and indexing controls used before both app modes."""
    if "active_sources" not in st.session_state:
        st.session_state.active_sources = []

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    if st.button("Index documents"):
        openai_api_key = _get_active_openai_api_key()

        if not openai_api_key:
            _warn_missing_api_key()
            return

        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Saving, chunking, and indexing documents..."):
                documents = []
                chunks = []
                indexed_count = 0
                files_indexed = 0
                skipped_files = []
                active_sources = []

                for uploaded_file in uploaded_files:
                    saved_path = save_uploaded_file(uploaded_file)
                    active_sources.append(saved_path.name)
                    file_hash = compute_file_hash(saved_path)

                    if is_file_indexed(file_hash):
                        skipped_files.append(saved_path.name)
                        continue

                    file_documents = extract_text_from_file(saved_path)
                    file_chunks = chunk_documents(file_documents)
                    delete_chunks_for_source(saved_path.name)
                    file_indexed_count = index_chunks(
                        file_chunks,
                        openai_api_key=openai_api_key,
                    )

                    # Only mark the file after Chroma accepts its chunk embeddings.
                    if file_indexed_count > 0:
                        mark_file_indexed(file_hash, saved_path.name)
                        files_indexed += 1

                    documents.extend(file_documents)
                    chunks.extend(file_chunks)
                    indexed_count += file_indexed_count

            st.session_state.active_sources = active_sources
            st.session_state.quiz_item = None
            st.session_state.quiz_grade = None

            if files_indexed > 0:
                st.success("Documents saved, chunked, and indexed.")
            else:
                st.info("No new files were indexed in this run.")

            st.write(f"Files uploaded: {len(uploaded_files)}")
            st.write(f"Files indexed: {files_indexed}")
            st.write(f"Files skipped (already indexed): {len(skipped_files)}")
            st.write(f"Extracted text sections/pages: {len(documents)}")
            st.write(f"Chunks created: {len(chunks)}")
            st.write(f"Chunks indexed: {indexed_count}")

            for filename in skipped_files:
                st.info(f"Skipping already indexed file: {filename}")

            with st.expander("Preview extracted text"):
                if not documents:
                    if skipped_files:
                        st.write(
                            "All uploaded files were skipped because they were "
                            "already indexed. You can ask questions now."
                        )
                    else:
                        st.write("No text could be extracted from these files.")
                else:
                    for index, document in enumerate(documents, start=1):
                        page = _format_page(document["page"])
                        preview = document["text"][:1000]

                        st.markdown(
                            f"**Section {index}: {document['source']} "
                            f"(page: {page})**"
                        )
                        st.write(preview)

            with st.expander("Preview chunks"):
                if not chunks:
                    if skipped_files:
                        st.write(
                            "No chunks were created in this run because the "
                            "uploaded files were already indexed."
                        )
                    else:
                        st.write("No chunks were created.")
                else:
                    for index, chunk in enumerate(chunks, start=1):
                        metadata = chunk["metadata"]
                        page = _format_page(metadata["page"])
                        preview = chunk["text"][:1000]

                        st.markdown(
                            f"**Chunk {index}: {metadata['source']} "
                            f"(page: {page}, id: {metadata['chunk_id']})**"
                        )
                        st.write(preview)


def _show_ask_questions_tab():
    """Show the existing agentic RAG question-answering UI."""
    question = st.text_input("Ask a question about your documents")

    if st.button("Ask"):
        openai_api_key = _get_active_openai_api_key()

        if not openai_api_key:
            _warn_missing_api_key()
            return

        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Running agentic retrieval and generating an answer..."):
                result = run_agentic_rag(
                    question,
                    openai_api_key=openai_api_key,
                )

            st.subheader("Answer")
            st.write(result["answer"])
            _show_cited_chunks(
                _find_cited_chunks_in_trace(result["answer"], result["trace"]),
                result["answer"],
            )

            _show_agent_trace(result["trace"])


def _show_quiz_mode_tab():
    """Show the beginner-friendly quiz workflow."""
    st.subheader("Quiz Mode")

    if "quiz_item" not in st.session_state:
        st.session_state.quiz_item = None

    if "quiz_grade" not in st.session_state:
        st.session_state.quiz_grade = None

    topic = st.text_input(
        "Topic",
        placeholder="Optional: enter a topic from your documents",
    )

    active_sources = st.session_state.get("active_sources", [])
    if active_sources:
        st.info(f"Quiz Mode is using: {', '.join(active_sources)}")
    else:
        st.info(
            "Quiz Mode will search all indexed documents. Index or re-index "
            "uploaded files in this session to focus the quiz on those files."
        )

    if st.button("Generate Quiz Question"):
        openai_api_key = _get_active_openai_api_key()

        if not openai_api_key:
            _warn_missing_api_key()
            return

        with st.spinner("Generating a quiz question from your documents..."):
            st.session_state.quiz_item = generate_quiz_question(
                topic,
                openai_api_key=openai_api_key,
                source_filters=active_sources,
            )
            st.session_state.quiz_grade = None

    if not st.session_state.quiz_item:
        st.write("Generate a question when you are ready to practice.")
        return

    quiz_item = st.session_state.quiz_item

    st.markdown("**Question**")
    st.write(quiz_item["question"])

    user_answer = st.text_area(
        "Your answer",
        key="quiz_user_answer",
        height=140,
    )

    if st.button("Submit Answer"):
        openai_api_key = _get_active_openai_api_key()

        if not openai_api_key:
            _warn_missing_api_key()
            return

        if not user_answer.strip():
            st.warning("Please enter an answer before submitting.")
        else:
            with st.spinner("Grading your answer..."):
                st.session_state.quiz_grade = grade_quiz_answer(
                    quiz_item["question"],
                    user_answer,
                    quiz_item["source_chunks"],
                    openai_api_key=openai_api_key,
                )

    if st.session_state.quiz_grade:
        grade = st.session_state.quiz_grade

        st.subheader("Quiz Result")
        st.write(f"Score: {grade['score']}/100")
        st.write(f"Result: {'Correct' if grade['is_correct'] else 'Incorrect'}")
        st.write(f"Feedback: {grade['feedback']}")
        st.write(f"Ideal answer: {grade['ideal_answer']}")
        st.write(f"Citation: {grade['citation']}")
        _show_cited_chunks(
            grade.get("cited_chunks", []),
            f"{grade['ideal_answer']} {grade['feedback']}",
        )


st.set_page_config(page_title="Agentic RAG Tutor", layout="wide")

st.title("Agentic RAG Tutor")

_show_api_settings()
_show_indexing_controls()

mode = st.radio(
    "Mode",
    ["Ask Questions", "Quiz Mode"],
    horizontal=True,
    key="active_mode",
)

if mode == "Ask Questions":
    _show_ask_questions_tab()
else:
    _show_quiz_mode_tab()
