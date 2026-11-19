from pathlib import Path
import pandas as pd

def load_data():
    csv_path = Path(__file__).resolve().parents[2]/"data"/"Resume dataset.csv"

    if not csv_path.exists():
       raise FileNotFoundError("Dataset file not found")

    df = pd.read_csv(csv_path)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.shape)