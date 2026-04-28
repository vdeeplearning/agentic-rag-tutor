import streamlit as st

from src.chunking import chunk_documents
from src.ingest import ingest_uploaded_files
from src.vectorstore import index_chunks, retrieve_chunks


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
            documents = ingest_uploaded_files(uploaded_files)
            chunks = chunk_documents(documents)
            indexed_count = index_chunks(chunks)

        st.success("Documents saved, chunked, and indexed.")
        st.write(f"Files processed: {len(uploaded_files)}")
        st.write(f"Extracted text sections/pages: {len(documents)}")
        st.write(f"Chunks created: {len(chunks)}")
        st.write(f"Chunks indexed: {indexed_count}")

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
        with st.spinner("Retrieving relevant chunks..."):
            retrieved_chunks = retrieve_chunks(question)

        st.success(f"Retrieved {len(retrieved_chunks)} chunks.")

        st.subheader("Retrieved Chunks")
        if not retrieved_chunks:
            st.write("No chunks were retrieved.")
        else:
            for index, chunk in enumerate(retrieved_chunks, start=1):
                metadata = chunk["metadata"]
                page = metadata["page"] or "N/A"
                preview = chunk["text"][:1000]

                title = (
                    f"Result {index}: {metadata['source']} "
                    f"(page: {page}, id: {metadata['chunk_id']}, "
                    f"distance: {chunk['distance']:.4f})"
                )

                with st.expander(title):
                    st.write(preview)

st.subheader("Answer")
st.write("LLM answering is not implemented yet. This milestone only retrieves chunks.")

st.subheader("Agent Trace")
st.write("No agent trace yet.")
