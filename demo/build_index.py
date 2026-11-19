"""Build the public demo index using synthetic resumes only."""

import csv
from pathlib import Path

from langchain_chroma import Chroma

from src.models.embedding_service import load_embedding


def build_demo_index():
    demo_dir = Path(__file__).resolve().parent
    with (demo_dir / "synthetic_resumes.csv").open(encoding="utf-8", newline="") as file:
        resumes = list(csv.DictReader(file))

    index_dir = demo_dir / "chromadb"
    index_dir.mkdir(exist_ok=False)
    store = Chroma(
        collection_name="Resume_collection",
        embedding_function=load_embedding(),
        persist_directory=str(index_dir),
    )
    store.add_texts(
        texts=[resume["Text"] for resume in resumes],
        metadatas=[
            {
                "resume_id": index,
                "category": resume["category"],
                "job_title": resume["job_title"],
            }
            for index, resume in enumerate(resumes)
        ],
        ids=[f"chunk_{index}" for index in range(len(resumes))],
    )
    print(f"Indexed {store._collection.count()} synthetic resumes.")


if __name__ == "__main__":
    build_demo_index()
