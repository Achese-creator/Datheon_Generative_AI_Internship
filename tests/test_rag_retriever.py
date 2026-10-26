from pathlib import Path

import pytest

from task2_rag_assistant.retriever import LocalRetriever


class FakeEmbeddingModel:
    """Deterministic test double; not intended for real semantic search."""

    keywords = ("retrieval", "pdf", "medicine")

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count(keyword)) + 0.01
            for keyword in self.keywords
        ]

    def embed(self, documents):
        return (self._vector(text) for text in documents)

    def query_embed(self, query):
        return iter([self._vector(query)])


@pytest.fixture
def retriever(tmp_path: Path) -> LocalRetriever:
    return LocalRetriever(
        persist_directory=tmp_path / "chroma",
        collection_name="test_chunks",
        embedding_model=FakeEmbeddingModel(),
    )


def test_add_chunks_and_retrieve_relevant_source(
    retriever: LocalRetriever,
) -> None:
    chunks = [
        {
            "text": "Retrieval finds relevant document passages.",
            "metadata": {"source": "guide.txt", "page_number": None},
        },
        {
            "text": "PDF files can preserve their original page numbers.",
            "metadata": {"source": "manual.pdf", "page_number": 4},
        },
    ]

    assert retriever.add_chunks(chunks) == 2

    results = retriever.search("retrieval", top_k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "guide.txt"
    assert "retrieval" in results[0]["text"].lower()
    assert results[0]["distance"] >= 0


def test_pdf_page_metadata_is_preserved(
    retriever: LocalRetriever,
) -> None:
    retriever.add_chunks(
        [
            {
                "text": "Medicine storage instructions.",
                "metadata": {"source": "medicine.pdf", "page_number": 7},
            }
        ]
    )

    results = retriever.search("medicine")

    assert results[0]["metadata"]["source"] == "medicine.pdf"
    assert results[0]["metadata"]["page_number"] == 7


def test_none_metadata_is_not_stored(
    retriever: LocalRetriever,
) -> None:
    retriever.add_chunks(
        [
            {
                "text": "Text file content about retrieval.",
                "metadata": {"source": "notes.txt", "page_number": None},
            }
        ]
    )

    results = retriever.search("retrieval")

    assert results[0]["metadata"]["source"] == "notes.txt"
    assert "page_number" not in results[0]["metadata"]


def test_empty_chunks_do_not_add_records(
    retriever: LocalRetriever,
) -> None:
    assert retriever.add_chunks([]) == 0
    assert retriever.search("retrieval") == []


def test_rejects_empty_question(retriever: LocalRetriever) -> None:
    with pytest.raises(ValueError, match="Question cannot be empty"):
        retriever.search("   ")


def test_rejects_invalid_top_k(retriever: LocalRetriever) -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search("retrieval", top_k=0)


def test_rejects_chunk_without_source(
    retriever: LocalRetriever,
) -> None:
    with pytest.raises(ValueError, match="source filename"):
        retriever.add_chunks(
            [{"text": "Some content", "metadata": {}}]
        )


def test_readding_same_chunk_does_not_duplicate(
    retriever: LocalRetriever,
) -> None:
    chunk = {
        "text": "Retrieval works with embeddings.",
        "metadata": {"source": "guide.txt", "page_number": None},
    }

    retriever.add_chunks([chunk])
    retriever.add_chunks([chunk])

    assert retriever.collection.count() == 1
