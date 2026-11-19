import time

from src.models.chunking import chunk_resume
from src.models.embedding_service import load_embedding
from src.storage.vector_store import create_vector_store


def setup_vector_store(batch_size=100, build=False):

    start_time = time.perf_counter()

    vector_store = create_vector_store()
    collection = vector_store._collection

    documents = chunk_resume()

    total_chunks = len(documents)
    skipped = 0
    inserted = 0

    for i in range(0, total_chunks, batch_size):

        batch = documents[i:i + batch_size]

        ids = [
            f"chunk_{j}"
            for j in range(i, i + len(batch))
        ]

        # Check existing records
        existing = collection.get(
            ids=ids,
            include=["documents"]
        )

        stored = dict(zip(
            existing["ids"],
            existing["documents"]
        ))

        new_documents = []
        new_ids = []

        for doc, chunk_id in zip(batch, ids):

            if chunk_id in stored:

                # Prevent skipping a different chunk
                if stored[chunk_id] != doc.page_content:
                    raise ValueError(
                        f"Chunk ID mismatch: {chunk_id}"
                    )

                skipped += 1

            else:
                new_documents.append(doc)
                new_ids.append(chunk_id)

        # Generate and store only missing embeddings
        if new_documents and build:

            texts = [
                doc.page_content
                for doc in new_documents
            ]

            embeddings = load_embedding().embed_documents(texts)

            collection.upsert(
                ids=new_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    doc.metadata
                    for doc in new_documents
                ]
            )

            inserted += len(new_ids)

    print("\n========== FINAL REPORT ==========")
    print("Total Chunks:", total_chunks)
    print("Existing Chunks:", skipped)
    print("Newly Inserted:", inserted)
    print("Stored in ChromaDB:", collection.count())
    print(
        "Processing Time:",
        round((time.perf_counter() - start_time) / 60, 2),
        "minutes"
    )
    print("==================================")


if __name__ == "__main__":
    import sys

    setup_vector_store(
        build="--build" in sys.argv
    )