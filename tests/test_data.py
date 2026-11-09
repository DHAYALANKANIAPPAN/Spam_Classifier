import pandas as pd
import pytest

from src.data import load_dataset


def test_load_dataset_maps_labels_and_drops_duplicates(tmp_path):
    csv = tmp_path / "spam.csv"
    pd.DataFrame({
        "v1": ["ham", "spam", "spam", "ham"],
        "v2": ["see you later", "free prize", "free prize", "ok"],
        "Unnamed: 2": [None, None, None, None],
    }).to_csv(csv, index=False, encoding="latin-1")
    df = load_dataset(csv)
    assert list(df.columns) == ["label", "text"]
    assert len(df) == 3  # duplicate 'free prize' removed
    assert set(df["label"]) == {0, 1}


def test_load_dataset_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_dataset(tmp_path / "nope.csv")
