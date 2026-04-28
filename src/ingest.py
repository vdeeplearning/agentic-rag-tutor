"""Save uploaded files and extract text for indexing.

This module intentionally stops before embeddings or Chroma. It only handles
the first step of a RAG app: turning uploaded documents into plain text plus
simple metadata.
"""

from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader

from src.config import UPLOAD_DIR


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".markdown"}


def save_uploaded_file(uploaded_file: Any, upload_dir: Path = UPLOAD_DIR) -> Path:
    """Save one Streamlit uploaded file and return its saved path."""
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Path(...).name strips any folder information and keeps just the filename.
    filename = Path(uploaded_file.name).name
    saved_path = upload_dir / filename

    with saved_path.open("wb") as output_file:
        output_file.write(uploaded_file.getbuffer())

    return saved_path


def extract_text_from_file(file_path: Path) -> list[dict[str, object]]:
    """Extract text from a saved document.

    Each returned dictionary has the same shape so later indexing code can
    handle PDFs, Word documents, and plain text in one simple loop.
    """
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        return _extract_pdf_text(file_path)

    if extension == ".docx":
        return _extract_docx_text(file_path)

    return _extract_plain_text(file_path)


def ingest_uploaded_files(uploaded_files: list[Any]) -> list[dict[str, object]]:
    """Save and extract text from all uploaded Streamlit files."""
    documents: list[dict[str, object]] = []

    for uploaded_file in uploaded_files:
        saved_path = save_uploaded_file(uploaded_file)
        documents.extend(extract_text_from_file(saved_path))

    return documents


def _extract_pdf_text(file_path: Path) -> list[dict[str, object]]:
    """Extract one document dictionary per PDF page."""
    reader = PdfReader(str(file_path))
    documents: list[dict[str, object]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {
                    "source": file_path.name,
                    "text": text,
                    "page": page_number,
                }
            )

    return documents


def _extract_docx_text(file_path: Path) -> list[dict[str, object]]:
    """Extract text from a Word document as one section."""
    document = Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    text = "\n".join(paragraphs).strip()

    if not text:
        return []

    return [
        {
            "source": file_path.name,
            "text": text,
            "page": None,
        }
    ]


def _extract_plain_text(file_path: Path) -> list[dict[str, object]]:
    """Extract text from TXT or Markdown files as one section."""
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return []

    return [
        {
            "source": file_path.name,
            "text": text,
            "page": None,
        }
    ]
