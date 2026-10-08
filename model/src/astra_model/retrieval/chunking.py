"""Split extracted document records into searchable text chunks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_records(records, chunk_size=800, chunk_overlap=120):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for record in records:
        text = record.get("text", "").strip()
        if not text:
            continue

        metadata = {
            "source": str(record.get("source", "unknown")),
            "page_number": record.get("page_number") or 0,
            "content_type": str(record.get("content_type", "text")),
            "extraction_method": str(
                record.get("extraction_method", "unknown")
            ),
        }

        for index, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": index,
                },
            })

    return chunks
