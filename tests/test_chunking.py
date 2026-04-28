"""Tests for character-based document chunking."""

import pytest

from src.chunking import chunk_documents


def test_chunk_documents_preserves_metadata():
    documents = [
        {
            "source": "lesson.pdf",
            "text": "abcdef",
            "page": 2,
        }
    ]

    chunks = chunk_documents(documents, chunk_size=3, chunk_overlap=1)

    assert chunks == [
        {
            "text": "abc",
            "metadata": {
                "source": "lesson.pdf",
                "page": 2,
                "chunk_id": 0,
            },
        },
        {
            "text": "cde",
            "metadata": {
                "source": "lesson.pdf",
                "page": 2,
                "chunk_id": 1,
            },
        },
        {
            "text": "ef",
            "metadata": {
                "source": "lesson.pdf",
                "page": 2,
                "chunk_id": 2,
            },
        },
    ]


def test_chunk_documents_skips_empty_text():
    documents = [
        {"source": "empty.txt", "text": "   ", "page": None},
        {"source": "notes.md", "text": "Useful notes", "page": None},
    ]

    chunks = chunk_documents(documents, chunk_size=20, chunk_overlap=5)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Useful notes"
    assert chunks[0]["metadata"]["source"] == "notes.md"


def test_chunk_documents_rejects_overlap_larger_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=100, chunk_overlap=100)
