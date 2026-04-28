import streamlit as st

from src.chunking import chunk_documents
from src.config import get_openai_api_key
from src.dedup import compute_file_hash, is_file_indexed, mark_file_indexed
from src.ingest import extract_text_from_file, save_uploaded_file
from src.quiz import generate_quiz_question, grade_quiz_answer
from src.rag import run_agentic_rag
from src.vectorstore import index_chunks


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


def _preview_text(text, limit=800):
    """Trim long chunks so the UI stays easy to scan."""
    if len(text) <= limit:
        return text

    return f"{text[:limit]}..."


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


def _show_retrieved_chunk(index, chunk):
    """Display one retrieved chunk with readable metadata."""
    metadata = chunk["metadata"]
    source = metadata.get("source", "unknown")
    page = _format_page(metadata.get("page"))
    chunk_id = metadata.get("chunk_id", "unknown")
    distance = _format_distance(chunk.get("distance"))
    preview = _preview_text(chunk.get("text", ""))

    st.markdown(f"**Chunk {index}**")
    st.write(f"Source: {source}")
    st.write(f"Page: {page}")
    st.write(f"Chunk ID: {chunk_id}")
    st.write(f"Distance: {distance}")
    st.write(preview)


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

            st.markdown("**Retrieved chunks**")
            if not attempt["retrieved_chunks"]:
                st.write("No chunks were retrieved.")
            else:
                for index, chunk in enumerate(attempt["retrieved_chunks"], start=1):
                    _show_retrieved_chunk(index, chunk)

                    if index < len(attempt["retrieved_chunks"]):
                        st.divider()


def _show_source_chunks(source_chunks):
    """Show quiz source chunks for transparency."""
    with st.expander("Source chunks"):
        if not source_chunks:
            st.write("No source chunks are available.")
            return

        for index, chunk in enumerate(source_chunks, start=1):
            _show_retrieved_chunk(index, chunk)

            if index < len(source_chunks):
                st.divider()


def _show_indexing_controls():
    """Shared upload and indexing controls used before both app modes."""
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

                for uploaded_file in uploaded_files:
                    saved_path = save_uploaded_file(uploaded_file)
                    file_hash = compute_file_hash(saved_path)

                    if is_file_indexed(file_hash):
                        skipped_files.append(saved_path.name)
                        continue

                    file_documents = extract_text_from_file(saved_path)
                    file_chunks = chunk_documents(file_documents)
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

            st.success("Documents saved, chunked, and indexed.")
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

    if st.button("Generate Quiz Question"):
        openai_api_key = _get_active_openai_api_key()

        if not openai_api_key:
            _warn_missing_api_key()
            return

        with st.spinner("Generating a quiz question from your documents..."):
            st.session_state.quiz_item = generate_quiz_question(
                topic,
                openai_api_key=openai_api_key,
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

    _show_source_chunks(quiz_item["source_chunks"])


st.set_page_config(page_title="Agentic RAG Tutor", layout="wide")

st.title("Agentic RAG Tutor")

_show_api_settings()
_show_indexing_controls()

ask_tab, quiz_tab = st.tabs(["Ask Questions", "Quiz Mode"])

with ask_tab:
    _show_ask_questions_tab()

with quiz_tab:
    _show_quiz_mode_tab()
