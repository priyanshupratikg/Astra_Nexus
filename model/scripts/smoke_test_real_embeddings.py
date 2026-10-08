"""End-to-end smoke test using real sentence-transformer embeddings."""

from pathlib import Path
from tempfile import TemporaryDirectory

from astra_model.retrieval import EmbeddingService, VectorStore, chunk_records


def main():
    print("Loading real embedding model...")
    embeddings = EmbeddingService()

    records = [
        {
            "text": (
                "Astronomy is the scientific study of stars, galaxies, "
                "planets, and the wider universe."
            ),
            "source": "astronomy.pdf",
            "page_number": 1,
            "content_type": "pdf_page",
            "extraction_method": "native_pdf_text",
        },
        {
            "text": (
                "Cooking pasta involves boiling water, adding pasta, "
                "and cooking until it reaches the desired texture."
            ),
            "source": "cooking.pdf",
            "page_number": 2,
            "content_type": "pdf_page",
            "extraction_method": "native_pdf_text",
        },
    ]

    chunks = chunk_records(records)
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"

    with TemporaryDirectory() as temp_dir:
        store = VectorStore(
            persist_directory=str(Path(temp_dir) / "chroma"),
            collection_name="embedding_smoke_test",
            embedding_service=embeddings,
        )

        added = store.add_chunks(chunks)
        print(f"Indexed {added} chunks.")

        question = "Which science studies stars and galaxies?"
        results = store.search(question, top_k=2)

        if not results:
            raise RuntimeError("Retrieval returned no results.")

        print(f"\nQuestion: {question}")
        print("\nRetrieved passages:")

        for rank, result in enumerate(results, start=1):
            print(f"\n{rank}. Source: {result['metadata']['source']}")
            print(f"   Page: {result['metadata']['page_number']}")
            print(f"   Distance: {result['distance']:.4f}")
            print(f"   Text: {result['text']}")

        if results[0]["metadata"]["source"] != "astronomy.pdf":
            raise AssertionError(
                "Semantic retrieval did not rank the astronomy passage first."
            )

    print("\nSUCCESS: Real embeddings and semantic retrieval are working.")


if __name__ == "__main__":
    main()
