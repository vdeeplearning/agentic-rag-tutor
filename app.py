import streamlit as st

from src.chunking import chunk_documents
from src.dedup import compute_file_hash, is_file_indexed, mark_file_indexed
from src.ingest import extract_text_from_file, save_uploaded_file
from src.rag import run_agentic_rag
from src.vectorstore import index_chunks


def _format_page(page):
    """Show missing page numbers in a friendly way."""
    return page or "N/A"


def _format_distance(distance):
    """Show Chroma distance values consistently."""
    if isinstance(distance, (int, float)):
        return f"{distance:.4f}"

    return "N/A"


def _preview_text(text, limit=800):
    """Trim long chunks so the trace stays easy to scan."""
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
    """Display one retrieved chunk inside the trace."""
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
        title = f"Attempt {attempt['attempt']} — {decision.upper()}"

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


st.set_page_config(page_title="Agentic RAG Tutor", layout="wide")

st.title("Agentic RAG Tutor")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True
)

if st.button("Index documents"):
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
                file_indexed_count = index_chunks(file_chunks)

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
                    page = document["page"] or "N/A"
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
                    page = metadata["page"] or "N/A"
                    preview = chunk["text"][:1000]

                    st.markdown(
                        f"**Chunk {index}: {metadata['source']} "
                        f"(page: {page}, id: {metadata['chunk_id']})**"
                    )
                    st.write(preview)

question = st.text_input("Ask a question about your documents")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Running agentic retrieval and generating an answer..."):
            result = run_agentic_rag(question)

        st.subheader("Answer")
        st.write(result["answer"])

        _show_agent_trace(result["trace"])
