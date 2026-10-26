import hashlib
from pathlib import Path
from typing import Any

import chromadb
from fastembed import TextEmbedding


class LocalRetriever:
    """Store and search document chunks using local embeddings and ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = "task2_rag_assistant/storage/chroma",
        collection_name: str = "document_chunks",
        embedding_model: Any | None = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = embedding_model or TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    @staticmethod
    def _chunk_id(chunk: dict) -> str:
        """Generate a stable ID so adding the same chunk is idempotent."""
        identity = (
            f"{chunk['metadata'].get('source', '')}|"
            f"{chunk['metadata'].get('page_number', '')}|"
            f"{chunk['text']}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def add_chunks(self, chunks: list[dict]) -> int:
        """Embed and store chunks. Return the number of chunks processed."""
        if not chunks:
            return 0

        texts: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []
        ids: list[str] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every chunk must contain non-empty text.")

            source = metadata.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Every chunk must have a source filename.")

            # ChromaDB does not accept None as a metadata value.
            clean_metadata = {
                key: value
                for key, value in metadata.items()
                if value is not None
                and isinstance(value, (str, int, float, bool))
            }

            texts.append(text)
            metadatas.append(clean_metadata)
            ids.append(self._chunk_id(chunk))

        embeddings = [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in self.embedding_model.embed(texts)
        ]

        if len(embeddings) != len(texts):
            raise RuntimeError("The embedding model returned an unexpected number of vectors.")

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(texts)

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        """Return the most relevant stored chunks for a question."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        stored_count = self.collection.count()
        if stored_count == 0:
            return []

        query_vectors = list(self.embedding_model.query_embed(question.strip()))
        if not query_vectors:
            raise RuntimeError("The embedding model returned no query vector.")

        query_vector = query_vectors[0]
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        else:
            query_vector = list(query_vector)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, stored_count),
            include=["documents", "metadatas", "distances"],
        )

        matches: list[dict] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )

        return matches
