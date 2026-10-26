from collections.abc import Callable
from dataclasses import dataclass

from task2_rag_assistant.retriever import LocalRetriever


@dataclass(frozen=True)
class SourceReference:
    source: str
    page_number: int | None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[SourceReference]


def build_answer_prompt(question: str, passages: list[dict]) -> str:
    """Build a grounded prompt from retrieved evidence."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not passages:
        raise ValueError("At least one evidence passage is required.")

    evidence_sections: list[str] = []

    for index, passage in enumerate(passages, start=1):
        metadata = passage.get("metadata", {})
        source = metadata.get("source", "Unknown source")
        page_number = metadata.get("page_number")
        location = f"{source}, page {page_number}" if page_number else source

        evidence_sections.append(
            f"[Source {index}: {location}]\n{passage['text']}"
        )

    evidence = "\n\n".join(evidence_sections)

    return f"""
You are a document question-answering assistant.

Answer the user's question using only the evidence passages below.
Do not add facts from your general knowledge.
If the evidence does not answer the question, clearly say that the
provided documents do not contain enough information to answer it.
Do not invent facts, quotations, filenames, page numbers, or citations.
Cite factual claims using the exact source labels, such as [Source 1].
Use concise, clear language.

EVIDENCE:
{evidence}

QUESTION:
{question.strip()}

ANSWER:
""".strip()


def answer_question(
    question: str,
    retriever: LocalRetriever,
    llm_call: Callable[[str], str],
    top_k: int = 3,
) -> AnswerResult:
    """Retrieve evidence and generate an answer grounded in that evidence."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    passages = retriever.search(question, top_k=top_k)

    if not passages:
        return AnswerResult(
            answer="I could not find relevant evidence in the indexed documents.",
            sources=[],
        )

    prompt = build_answer_prompt(question, passages)
    answer = llm_call(prompt)

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The language model returned an empty answer.")

    unique_sources: list[SourceReference] = []
    seen: set[tuple[str, int | None]] = set()

    for passage in passages:
        metadata = passage.get("metadata", {})
        source = metadata.get("source")

        if not isinstance(source, str) or not source.strip():
            continue

        page_number = metadata.get("page_number")
        key = (source, page_number)

        if key not in seen:
            seen.add(key)
            unique_sources.append(
                SourceReference(
                    source=source,
                    page_number=page_number,
                )
            )

    return AnswerResult(answer=answer.strip(), sources=unique_sources)
