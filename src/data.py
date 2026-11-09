"""Loading the SMS Spam Collection dataset."""
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/raw/spam.csv")
LABEL_MAP = {"ham": 0, "spam": 1}


def load_dataset(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Return a DataFrame with columns: text, label (0 = Not Spam, 1 = Spam).

    Expects the Kaggle 'spam.csv' layout (columns v1 = label, v2 = message).
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Download the SMS Spam Collection from Kaggle "
            "(uciml/sms-spam-collection-dataset) and place spam.csv in data/raw/."
        )
    df = pd.read_csv(path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    df["label"] = df["label"].str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label", "text"]).drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)
