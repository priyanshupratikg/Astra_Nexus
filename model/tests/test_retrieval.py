from astra_model.retrieval import chunk_records, VectorStore


class FakeEmbeddingService:
    """Deterministic test embeddings; not a production semantic model."""

    def embed_documents(self, texts):
        return [
            [
                float("astronomy" in text.lower()),
                float("cooking" in text.lower()),
                1.0,
            ]
            for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_chunk_records_preserves_source_metadata():
    records = [{
        "text": "Astronomy studies stars and galaxies. " * 60,
        "source": "science.pdf",
        "page_number": 2,
        "content_type": "pdf_page",
        "extraction_method": "native_pdf_text",
    }]

    chunks = chunk_records(records, chunk_size=200, chunk_overlap=30)

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 200 for chunk in chunks)
    assert all(chunk["metadata"]["source"] == "science.pdf" for chunk in chunks)
    assert all(chunk["metadata"]["page_number"] == 2 for chunk in chunks)


def test_skips_empty_records():
    assert chunk_records([{"text": "  "}]) == []


def test_rejects_invalid_chunk_settings():
    import pytest

    with pytest.raises(ValueError):
        chunk_records([], chunk_size=100, chunk_overlap=100)


def test_vector_store_adds_and_searches_chunks(tmp_path):
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_documents",
        embedding_service=FakeEmbeddingService(),
    )

    chunks = [
        {
            "text": "Astronomy explores stars and galaxies.",
            "metadata": {
                "source": "science.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        },
        {
            "text": "Cooking pasta requires boiling water.",
            "metadata": {
                "source": "cooking.pdf",
                "page_number": 3,
                "chunk_index": 0,
            },
        },
    ]

    assert store.add_chunks(chunks) == 2

    results = store.search("astronomy stars", top_k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "science.pdf"
    assert results[0]["metadata"]["page_number"] == 1


def test_vector_store_empty_collection(tmp_path):
    store = VectorStore(
        persist_directory=str(tmp_path / "empty"),
        collection_name="empty_documents",
        embedding_service=FakeEmbeddingService(),
    )

    assert store.search("anything") == []
