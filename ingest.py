"""
Ingest official Passport Seva PDFs into ChromaDB with sentence-transformers embeddings.

SMAI Assignment 3 - T10.1 Passport Assistant (Tier 1). No training; indexing only.
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PDF_DIR,
)


def find_pdf_files(pdf_dir: Path) -> list[Path]:
    """Return sorted list of PDF paths in the given directory."""
    if not pdf_dir.is_dir():
        return []
    return sorted(pdf_dir.glob("*.pdf"))


def clean_pdf_text(raw: str) -> str:
    """
    Normalize extracted PDF text:
    - collapse repeated spaces/tabs within lines
    - collapse repeated blank lines
    - strip leading/trailing whitespace
    """
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping character chunks.
    step = max(chunk_size - overlap, 1) to avoid infinite loops.
    """
    text = text.strip()
    if not text:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step

    return chunks


def reset_chroma_collection(
    chroma_client: chromadb.PersistentClient,
) -> chromadb.Collection:
    """Delete the collection if it exists, then create a fresh one (cosine similarity)."""
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Removed existing collection: {COLLECTION_NAME}")
    except Exception:
        print(f"No existing collection to remove (or delete failed cleanly): {COLLECTION_NAME}")

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def main() -> None:
    pdf_dir = Path(PDF_DIR)
    pdfs = find_pdf_files(pdf_dir)

    if not pdfs:
        print("\nERROR: No PDF files found.")
        print(f"Place official Passport Seva PDFs inside: {pdf_dir.resolve()}\n")
        return

    print(f"\nFound {len(pdfs)} PDF file(s) in {pdf_dir.resolve()}")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = reset_chroma_collection(chroma_client)

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    embeddings_list: list[list[float]] = []

    total_chunks = 0

    for pdf_path in pdfs:
        print(f"\nProcessing: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)

        chunks_for_this_pdf = 0

        for page_index in range(page_count):
            page = reader.pages[page_index]
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                print(f"  Warning: Could not extract text from page {page_index + 1}: {exc}")
                continue

            cleaned = clean_pdf_text(raw_text)
            if not cleaned:
                continue

            page_number = page_index + 1
            chunks = split_into_chunks(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk_idx, chunk_text in enumerate(chunks):
                doc_id = f"{pdf_path.stem}_p{page_number}_c{chunk_idx}"

                embedding = embedder.encode(chunk_text, convert_to_numpy=True).tolist()

                ids.append(doc_id)
                documents.append(chunk_text)
                embeddings_list.append(embedding)
                metadatas.append(
                    {
                        "source": pdf_path.name,
                        "page": int(page_number),
                        "chunk_index": int(chunk_idx),
                    }
                )
                chunks_for_this_pdf += 1

        total_chunks += chunks_for_this_pdf
        print(f"  Chunks created from this PDF: {chunks_for_this_pdf}")

    if not ids:
        print("\nERROR: No text chunks could be created from your PDFs.")
        print("The PDFs may be empty, image-only, or extraction failed.\n")
        return

    print(f"\nTotal chunks prepared: {len(ids)}")
    print("Encoding and uploading to ChromaDB (CPU may take a while)...")

    batch_size = 64
    for i in range(0, len(ids), batch_size):
        batch_slice = slice(i, i + batch_size)
        collection.add(
            ids=ids[batch_slice],
            documents=documents[batch_slice],
            embeddings=embeddings_list[batch_slice],
            metadatas=metadatas[batch_slice],
        )

    print(f"\nStored {len(ids)} chunk(s) in collection '{COLLECTION_NAME}'")
    print(f"Total chunks stored: {len(ids)}")
    print(f"ChromaDB path: {Path(CHROMA_DIR).resolve()}\n")


if __name__ == "__main__":
    main()
