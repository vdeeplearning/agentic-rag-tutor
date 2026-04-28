import streamlit as st

from src.chunking import chunk_documents
from src.ingest import ingest_uploaded_files


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
        documents = ingest_uploaded_files(uploaded_files)
        chunks = chunk_documents(documents)

        st.success("Documents saved and text extracted.")
        st.write(f"Files processed: {len(uploaded_files)}")
        st.write(f"Extracted text sections/pages: {len(documents)}")
        st.write(f"Chunks created: {len(chunks)}")

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
    st.success("Answer placeholder - not implemented yet.")

st.subheader("Answer")
st.write("No answer yet.")

st.subheader("Agent Trace")
st.write("No agent trace yet.")
