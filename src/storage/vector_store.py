import time
from pathlib import Path

from langchain_chroma import Chroma
from src.models.chunking import chunk_resume
from src.models.embedding_service import load_embedding


def create_vector_store(*, allow_demo=False):
    project_root = Path(__file__).resolve().parents[2]
    index_dir = project_root / "chromadb_full"
    demo_dir = project_root / "demo" / "chromadb"
    if allow_demo and not (index_dir / "chroma.sqlite3").is_file():
        if not (demo_dir / "chroma.sqlite3").is_file():
            raise FileNotFoundError("No local or synthetic demo ChromaDB index found.")
        index_dir = demo_dir

    embedding = load_embedding()

    vector_store = Chroma(
        collection_name="Resume_collection",
        embedding_function=embedding,
        persist_directory=str(index_dir)
    )

    return vector_store


def insert_chunk():
    start_time = time.perf_counter()

    vector_store = create_vector_store()

    documents = chunk_resume()
    batch_size = 100
    total_chunks = len(documents)

    for i in range(0, total_chunks, batch_size):
        batch = documents[i:i + batch_size]

        ids = [
            f"chunk_{j}"
            for j in range(i, i + len(batch))
        ]

        # Check existing IDs before generating embeddings
        existing = vector_store.get(ids=ids, include=[])
        existing_ids = set(existing["ids"])

        new_documents = []
        new_ids = []

        for doc, chunk_id in zip(batch, ids):
            if chunk_id not in existing_ids:
                new_documents.append(doc)
                new_ids.append(chunk_id)

        # Generate embeddings ONLY for missing chunks
        if new_documents:
            embeddings = load_embedding().embed_documents(
                [doc.page_content for doc in new_documents]
            )

            vector_store._collection.upsert(
                ids=new_ids,
                embeddings=embeddings,
                documents=[
                    doc.page_content for doc in new_documents
                ],
                metadatas=[
                    doc.metadata for doc in new_documents
                ]
            )

        print(
            f"Processed: {min(i + batch_size, total_chunks)}/{total_chunks}"
            f" | Stored: {vector_store._collection.count()}"
        )

    total_minutes = (time.perf_counter() - start_time) / 60

    print("\n========== FINAL REPORT ==========")
    print("Total Chunks:", total_chunks)
    print("Total Stored:", vector_store._collection.count())
    print(f"Processing Time: {total_minutes:.2f} minutes")
    print("Vector Dimension: 1024")
    print("==================================")


if __name__ == "__main__":
    store = create_vector_store()
    print("Stored Chunks:", store._collection.count())
