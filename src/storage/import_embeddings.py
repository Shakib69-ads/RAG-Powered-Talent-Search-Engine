
import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import chromadb


# Project file paths
BASE_DIR = Path(__file__).resolve().parents[2]

ZIP_PATH = BASE_DIR / "resume_embeddings.zip"
JSON_PATH = BASE_DIR / "resume_chunks.json"
DB_PATH = BASE_DIR / "chromadb_full"


def import_embeddings():

    # Step 1: Load resume chunks
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    chunk_lookup = {chunk["id"]: chunk for chunk in chunks}

    print("Total JSON Chunks:", len(chunks))


    # Step 2: Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name="Resume_collection",
        embedding_function=None
    )

    print("Previously Stored:", collection.count())


    # Step 3: Read all NPZ files from ZIP
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        npz_files = sorted(
            name for name in archive.namelist()
            if name.endswith(".npz")
        )

        print("Total NPZ Files:", len(npz_files))

        if not npz_files:
            raise ValueError("No NPZ files found in ZIP")


        # Step 4: Process each NPZ file
        for name in npz_files:

            with np.load(BytesIO(archive.read(name)), allow_pickle=False) as data:
                ids = data["ids"].tolist()
                vectors = data["vectors"]

                if vectors.shape != (len(ids), 1024):
                    raise ValueError(f"Invalid vector shape in {name}")


                # Step 5: Insert in batches
                batch_size = 500

                for start in range(0, len(ids), batch_size):

                    end = start + batch_size

                    batch_ids = ids[start:end]
                    batch_vectors = vectors[start:end].tolist()

                    batch_texts = [
                        chunk_lookup[chunk_id]["text"]
                        for chunk_id in batch_ids
                    ]

                    batch_metadata = [
                        chunk_lookup[chunk_id]["metadata"]
                        for chunk_id in batch_ids
                    ]

                    collection.upsert(
                        ids=batch_ids,
                        embeddings=batch_vectors,
                        documents=batch_texts,
                        metadatas=batch_metadata
                    )

            print(f"Imported {name} | Stored: {collection.count()}")


    # Step 6: Final verification
    total_stored = collection.count()

    print("\nImport Completed!")
    print("Total JSON Chunks:", len(chunks))
    print("Total Stored Chunks:", total_stored)

    if total_stored != len(chunks):
        raise ValueError("Import incomplete! Chunk counts do not match.")

    print("All Embeddings Imported Successfully!")


if __name__ == "__main__":
    import_embeddings()
