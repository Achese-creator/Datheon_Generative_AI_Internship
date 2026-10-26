from pathlib import Path

from task2_rag_assistant.chunker import chunk_pages
from task2_rag_assistant.loader import load_document
from task2_rag_assistant.retriever import LocalRetriever

document_path = Path("task2_rag_assistant/documents/sample_document.txt")
pages = load_document(document_path)
chunks = chunk_pages(pages)

retriever = LocalRetriever(
    persist_directory="task2_rag_assistant/storage/smoke_test",
    collection_name="sample_document_test",
)

added = retriever.add_chunks(chunks)
print(f"Chunks added: {added}")

question = "Why is chunk overlap useful in a RAG pipeline?"
results = retriever.search(question, top_k=2)

print(f"\nQuestion: {question}")
for index, result in enumerate(results, start=1):
    print(f"\nResult {index}")
    print(f"Source: {result['metadata']['source']}")
    print(f"Distance: {result['distance']:.4f}")
    print(result["text"][:500])
