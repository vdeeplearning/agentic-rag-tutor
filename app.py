import streamlit as st

st.set_page_config(page_title="Agentic RAG Tutor", layout="wide")

st.title("Agentic RAG Tutor")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True
)

if st.button("Index documents"):
    st.info("Indexing placeholder - not implemented yet.")

question = st.text_input("Ask a question about your documents")

if st.button("Ask"):
    st.success("Answer placeholder - not implemented yet.")

st.subheader("Answer")
st.write("No answer yet.")

st.subheader("Agent Trace")
st.write("No agent trace yet.")
