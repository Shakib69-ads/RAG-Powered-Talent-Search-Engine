from src.models.preprocessing import preprocess_data
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_resume():
    df = preprocess_data()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    documents = []

    for index, row in df.iterrows():
        chunks = splitter.create_documents(
            [row["Text"]],
            metadatas=[{
                "resume_id": int(index),
                "category": str(row["category"]),
                "job_title": str(row["job_title"])
            }]
        )

        documents.extend(chunks)

    return documents


if __name__ == "__main__":
    documents = chunk_resume()

    print("Total Chunks:", len(documents))
    print("First Chunk:", documents[0].page_content[:300])
    print("Metadata:", documents[0].metadata)