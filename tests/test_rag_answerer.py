from pathlib import Path

import pytest

from task2_rag_assistant.answerer import (
    answer_question,
    build_answer_prompt,
)
from task2_rag_assistant.retriever import LocalRetriever


class FakeEmbeddingModel:
    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("overlap")) + 0.01,
            float(lowered.count("retrieval")) + 0.01,
            float(lowered.count("evidence")) + 0.01,
        ]

    def embed(self, documents):
        return (self._vector(text) for text in documents)

    def query_embed(self, query):
        return iter([self._vector(query)])


@pytest.fixture
def retriever(tmp_path: Path) -> LocalRetriever:
    instance = LocalRetriever(
        persist_directory=tmp_path / "chroma",
        collection_name="answerer_tests",
        embedding_model=FakeEmbeddingModel(),
    )
    instance.add_chunks(
        [
            {
                "text": (
                    "Chunk overlap preserves context between adjacent "
                    "text chunks."
                ),
                "metadata": {"source": "guide.pdf", "page_number": 2},
            },
            {
                "text": "RAG answers should be grounded in retrieved evidence.",
                "metadata": {"source": "guide.pdf", "page_number": 3},
            },
        ]
    )
    return instance


def test_prompt_includes_question_evidence_and_citation_labels() -> None:
    prompt = build_answer_prompt(
        "What does overlap do?",
        [
            {
                "text": "Overlap preserves context.",
                "metadata": {"source": "guide.pdf", "page_number": 2},
            }
        ],
    )

    assert "What does overlap do?" in prompt
    assert "Overlap preserves context." in prompt
    assert "[Source 1: guide.pdf, page 2]" in prompt
    assert "Do not invent facts" in prompt


def test_answer_question_returns_answer_and_source(
    retriever: LocalRetriever,
) -> None:
    def fake_llm(prompt: str) -> str:
        assert "Chunk overlap preserves context" in prompt
        return "Overlap preserves context between chunks [Source 1]."

    result = answer_question(
        "What does overlap do?",
        retriever,
        fake_llm,
        top_k=1,
    )

    assert "preserves context" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].source == "guide.pdf"
    assert result.sources[0].page_number == 2


def test_empty_question_is_rejected(retriever: LocalRetriever) -> None:
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(" ", retriever, lambda prompt: "Answer")


def test_empty_model_response_is_rejected(
    retriever: LocalRetriever,
) -> None:
    with pytest.raises(RuntimeError, match="empty answer"):
        answer_question("What is overlap?", retriever, lambda prompt: "  ")
