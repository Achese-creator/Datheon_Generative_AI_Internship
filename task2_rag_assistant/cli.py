import argparse
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from task2_rag_assistant.answerer import answer_question
from task2_rag_assistant.chunker import chunk_pages
from task2_rag_assistant.gemini_provider import generate_answer
from task2_rag_assistant.loader import load_document
from task2_rag_assistant.retriever import LocalRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENT = (
    PROJECT_ROOT
    / "task2_rag_assistant"
    / "documents"
    / "sample_document.txt"
)
STORAGE_PATH = PROJECT_ROOT / "task2_rag_assistant" / "storage" / "cli"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF or text document."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_DOCUMENT),
        help="Path to a PDF or TXT document.",
    )
    parser.add_argument(
        "--question",
        default="Why is chunk overlap useful in a RAG pipeline?",
        help="Question to answer from the document.",
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="Demo uses a fixed answer; live calls Gemini.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of passages to retrieve.",
    )
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("Question cannot be empty.")

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    load_dotenv(PROJECT_ROOT / ".env")

    document_path = Path(args.input).expanduser().resolve()
    if not document_path.is_file():
        parser.error(f"Document not found: {document_path}")

    try:
        pages = load_document(document_path)
        chunks = chunk_pages(pages)

        # Rebuild this CLI's collection on each run to prevent stale
        # chunks from a previously indexed document affecting answers.
        client = chromadb.PersistentClient(path=str(STORAGE_PATH))
        try:
            client.delete_collection("cli_document")
        except Exception as exc:
            if "does not exist" not in str(exc).lower():
                raise

        retriever = LocalRetriever(
            persist_directory=STORAGE_PATH,
            collection_name="cli_document",
        )
        retriever.add_chunks(chunks)

        if args.mode == "demo":
            def llm_call(prompt: str) -> str:
                return (
                    "Chunk overlap preserves some context between adjacent "
                    "chunks. This helps keep related information connected "
                    "when a document is split into smaller sections. "
                    "[Source 1]"
                )
        else:
            llm_call = generate_answer

        result = answer_question(
            question=args.question,
            retriever=retriever,
            llm_call=llm_call,
            top_k=args.top_k,
        )

        print(f"\nMode: {args.mode}")
        print(f"Document: {document_path.name}")
        print(f"Chunks indexed: {len(chunks)}")
        print(f"\nQuestion: {args.question}")
        print(f"\nAnswer:\n{result.answer}")
        print("\nRetrieved sources:")

        for source in result.sources:
            if source.page_number is not None:
                print(f"- {source.source}, page {source.page_number}")
            else:
                print(f"- {source.source}")

    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
