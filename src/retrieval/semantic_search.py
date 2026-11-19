
from src.storage.vector_store import create_vector_store
from src.models.preprocessing import preprocess_data


# 1. SEMANTIC SEARCH & UNIQUE CANDIDATE RANKING

def search_candidates(query, store, k=20, top_n=3):
    """
    Search CV chunks using semantic similarity.
    Return top unique candidates with distance scores.
    """

    if not isinstance(query, str) or not query.strip():
        return []

    if k <= 0 or top_n <= 0:
        return []

    # Search relevant chunks from ChromaDB
    scored_results = store.similarity_search_with_score(
        query=query,
        k=k
    )

    # Remove duplicate resume IDs
    seen_ids = set()
    unique_results = []

    for document, score in scored_results:

        resume_id = document.metadata.get("resume_id")

        if resume_id is None:
            continue

        if resume_id in seen_ids:
            continue

        seen_ids.add(resume_id)

        unique_results.append((document, score))

        # Stop when enough unique candidates are found
        if len(unique_results) >= top_n:
            break

    return unique_results


# 2. FULL CV RETRIEVAL & RECONSTRUCTION

def get_full_cv(store, resume_id):
    """
    Retrieve all chunks for a resume and reconstruct
    its full text by removing exact overlaps.
    """

    cv_data = store.get(
        where={"resume_id": int(resume_id)},
        include=["documents"]
    )

    if not cv_data["ids"]:
        return ""

    # Sort chunks by numeric chunk ID
    ordered_chunks = sorted(
        zip(cv_data["ids"], cv_data["documents"]),
        key=lambda item: int(item[0].split("_")[-1])
    )

    # Start with first chunk
    full_cv = ordered_chunks[0][1]

    # Merge remaining chunks
    for chunk_id, chunk_text in ordered_chunks[1:]:

        overlap = 0

        max_overlap = min(
            150,
            len(full_cv),
            len(chunk_text)
        )

        # Find exact overlapping text
        for size in range(max_overlap, 9, -1):

            if full_cv.endswith(chunk_text[:size]):
                overlap = size
                break

        if overlap > 0:
            full_cv += chunk_text[overlap:]
        else:
            full_cv += "\n" + chunk_text

    return full_cv


# 3. FULL CV ACCURACY VERIFICATION

def verify_full_cv(df, resume_id, reconstructed_cv):
    """
    Compare reconstructed CV with original
    preprocessed CV after whitespace normalization.
    """

    if resume_id not in df.index:
        return {
            "exact_match": False,
            "error": "Resume ID not found in original dataset"
        }

    original_cv = str(df.loc[resume_id, "Text"])

    # Normalize whitespace for fair comparison
    original = " ".join(original_cv.split())
    reconstructed = " ".join(reconstructed_cv.split())

    return {
        "exact_match": original == reconstructed,
        "original_length": len(original),
        "reconstructed_length": len(reconstructed)
    }


# 4. MAIN TEST

if __name__ == "__main__":

    # Connect to existing ChromaDB
    store = create_vector_store()

    # Recruiter's job requirement
    query = (
        "Python developer with Machine Learning "
        "and NLP experience"
    )

    # Search top 3 unique candidates
    candidates = search_candidates(
        query=query,
        store=store,
        k=20,
        top_n=3
    )

    print("\n========== SEMANTIC SEARCH ==========")
    print("Search Query:", query)
    print("Total Unique Candidates:", len(candidates))

    # Load original preprocessed data for verification only
    df = preprocess_data()

    # Display results
    for rank, (document, score) in enumerate(
        candidates,
        start=1
    ):

        resume_id = document.metadata["resume_id"]

        # Retrieve complete CV
        full_cv = get_full_cv(
            store=store,
            resume_id=resume_id
        )

        print(f"\n========== CANDIDATE {rank} ==========")

        print("Resume ID:", resume_id)

        print(
            "Job Title:",
            document.metadata.get("job_title")
        )

        print(
            "Category:",
            document.metadata.get("category")
        )

        print("Distance Score:", round(score, 4))

        print("Full CV Length:", len(full_cv))

        print("CV Preview:", full_cv[:300])

        # Verify reconstructed CV accuracy
        verification = verify_full_cv(
            df=df,
            resume_id=resume_id,
            reconstructed_cv=full_cv
        )

        print("\n--- ACCURACY VERIFICATION ---")

        print(
            "Exact Match:",
            verification["exact_match"]
        )

        if "error" in verification:
            print("Error:", verification["error"])
        else:
            print(
                "Original Length:",
                verification["original_length"]
            )

            print(
                "Reconstructed Length:",
                verification["reconstructed_length"]
            )

        print("====================================")

    print("\nSemantic Search Test Completed!")
