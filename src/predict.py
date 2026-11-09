"""Load the trained model and classify new messages."""
from functools import lru_cache
from pathlib import Path

import joblib

from src.preprocess import preprocess

MODEL_PATH = Path("models/spam_model.joblib")


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"No model at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_message(text: str) -> dict:
    """Return {'label': 'Spam' | 'Not Spam', 'spam_probability': float}."""
    model = load_model()
    prob = float(model.predict_proba([preprocess(text)])[0][1])
    return {"label": "Spam" if prob >= 0.5 else "Not Spam", "spam_probability": prob}
