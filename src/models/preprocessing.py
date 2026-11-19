from src.models.data_loader import load_data


def preprocess_data():
    df = load_data()

    # 1. Remove missing resume text
    df = df.dropna(subset=["Text"])

    # 2. Remove HTML tags
    df["Text"] = df["Text"].str.replace(r"<[^>]+>", " ", regex=True)

    # 3. Clean extra whitespace
    df["Text"] = df["Text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # 4. Remove empty resumes
    df = df[df["Text"] != ""]

    # 5. Remove duplicate resumes
    df = df.drop_duplicates(subset=["Text"])

    # 5. Reset index
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = preprocess_data()

    print("Dataset Shape:", df.shape)
    print("Missing Values:\n", df.isna().sum())
    print("Duplicate Resumes:", df.duplicated(subset=["Text"]).sum())
    print("Empty Resumes:", df["Text"].eq("").sum())