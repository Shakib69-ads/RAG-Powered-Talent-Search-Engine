from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )


if __name__ == "__main__":

    model = load_embedding()

    texts = [
        "Python developer with NLP experience",
        "Data Scientist skilled in Machine Learning"
    ]

    embeddings = model.embed_documents(texts)

    print("Total Embeddings:", len(embeddings))
    print("Embedding Dimension:", len(embeddings[0]))