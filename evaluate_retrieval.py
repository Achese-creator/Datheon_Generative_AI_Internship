from pathlib import Path
from tempfile import TemporaryDirectory
import gc

from task2_rag_assistant.loader import load_document
from task2_rag_assistant.chunker import chunk_pages
from task2_rag_assistant.retriever import LocalRetriever

document_path = Path("task2_rag_assistant/documents/sample_document.txt")
pages = load_document(document_path)
chunks = chunk_pages(pages)

print(f"Document: {document_path.name}")
print(f"Chunks created: {len(chunks)}")

with TemporaryDirectory(prefix="datheon_rag_eval_") as temp_dir:
    retriever = LocalRetriever(
        persist_directory=Path(temp_dir) / "chroma",
        collection_name="evaluation",
    )
    retriever.add_chunks(chunks)

    questions = [
        "Why is chunk overlap useful?",
        "How can a PDF answer cite its original page?",
        "What should the system do when the document lacks the answer?",
    ]

    try:
        for question in questions:
            print("\n" + "=" * 70)
            print(f"QUESTION: {question}")

            for rank, result in enumerate(
                retriever.search(question, top_k=2), start=1
            ):
                print(
                    f"\nResult {rank} | "
                    f"distance={result['distance']:.4f}"
                )
                print(result["text"][:600])
    finally:
        client = retriever.client
        retriever = None
        client.clear_system_cache()
        del client
        gc.collect()

print("\nEvaluation completed.")
