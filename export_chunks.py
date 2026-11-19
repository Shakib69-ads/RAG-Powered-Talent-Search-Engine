import json
from src.models.chunking import chunk_resume

documents = chunk_resume()

data = []

for i, doc in enumerate(documents):
    data.append({
        "id": f"chunk_{i}",
        "text": doc.page_content,
        "metadata": doc.metadata
    })

with open("resume_chunks.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print("Exported Chunks:", len(data))