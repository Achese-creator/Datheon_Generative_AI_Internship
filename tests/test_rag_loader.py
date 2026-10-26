from pathlib import Path

import pytest

from task2_rag_assistant.chunker import chunk_pages
from task2_rag_assistant.loader import DocumentPage, load_document


def test_load_text_file_preserves_source(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("First sentence.\nSecond sentence.", encoding="utf-8")

    pages = load_document(file_path)

    assert len(pages) == 1
    assert pages[0].source == "notes.txt"
    assert pages[0].page_number is None
    assert "First sentence." in pages[0].text


def test_missing_document_raises_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.txt")


def test_unsupported_file_type_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.docx"
    file_path.write_text("Not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)


def test_empty_text_file_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   ", encoding="utf-8")

    with pytest.raises(ValueError, match="No extractable text"):
        load_document(file_path)


def test_chunking_preserves_source_and_page_number() -> None:
    text = " ".join(
        f"Sentence {number} contains useful information."
        for number in range(30)
    )
    pages = [
        DocumentPage(
            text=text,
            source="report.pdf",
            page_number=3,
        )
    ]

    chunks = chunk_pages(pages, chunk_size=150, chunk_overlap=25)

    assert len(chunks) > 1
    assert all(chunk["metadata"]["source"] == "report.pdf" for chunk in chunks)
    assert all(chunk["metadata"]["page_number"] == 3 for chunk in chunks)
    assert all(chunk["text"].strip() for chunk in chunks)


def test_invalid_chunk_settings_are_rejected() -> None:
    pages = [DocumentPage(text="Some text", source="notes.txt")]

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=100, chunk_overlap=100)
