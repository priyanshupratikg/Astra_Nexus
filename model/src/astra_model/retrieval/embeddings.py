"""Text embeddings for semantic document retrieval."""

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]
