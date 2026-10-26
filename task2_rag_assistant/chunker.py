from langchain_text_splitters import RecursiveCharacterTextSplitter

from task2_rag_assistant.loader import DocumentPage


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    """Split document pages into chunks while preserving citations."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []

    for page in pages:
        if not page.text.strip():
            continue

        documents = splitter.create_documents(
            texts=[page.text],
            metadatas=[
                {
                    "source": page.source,
                    "page_number": page.page_number,
                }
            ],
        )

        for document in documents:
            chunks.append(
                {
                    "text": document.page_content,
                    "metadata": document.metadata,
                }
            )

    return chunks
