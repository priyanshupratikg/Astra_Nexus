"""Persistent ChromaDB storage and semantic retrieval."""

import hashlib
from pathlib import Path

import chromadb

from .embeddings import EmbeddingService


class VectorStore:
    def __init__(
        self,
        persist_directory="model/data/chroma_db",
        collection_name="astra_documents",
        embedding_service=None,
    ):
        self.client = chromadb.PersistentClient(
            path=str(Path(persist_directory))
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_service = embedding_service or EmbeddingService()

    def add_chunks(self, chunks, document_id=None):
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embedding_service.embed_documents(texts)

        ids, metadatas = [], []

        for index, chunk in enumerate(chunks):
            metadata = dict(chunk["metadata"])
            metadata["document_id"] = str(
                document_id or metadata.get("source", "unknown")
            )

            identity = (
                f"{metadata['document_id']}|"
                f"{metadata.get('page_number', 0)}|"
                f"{metadata.get('chunk_index', index)}|"
                f"{chunk['text']}"
            )
            ids.append(hashlib.sha256(identity.encode()).hexdigest())
            metadatas.append(metadata)

        # Upsert makes repeated ingestion of the same chunks idempotent.
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
        return len(texts)

    def search(self, query, top_k=5, document_id=None):
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        count = self.collection.count()
        if count == 0:
            return []

        where = {"document_id": str(document_id)} if document_id else None

        result = self.collection.query(
            query_embeddings=[self.embedding_service.embed_query(query)],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            matches.append({
                "text": text,
                "metadata": metadata,
                "distance": distance,
            })

        return matches
