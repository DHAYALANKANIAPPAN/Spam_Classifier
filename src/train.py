"""Train, evaluate and save the spam classifier (TF-IDF + Logistic Regression)."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.data import DEFAULT_PATH, load_dataset
from src.preprocess import preprocess

MODEL_PATH = Path("models/spam_model.joblib")
METRICS_PATH = Path("reports/metrics.json")
CM_PATH = Path("reports/confusion_matrix.png")


def build_model():
    # class_weight='balanced' helps because "not spam" dominates the data.
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.data)
    print(f"Loaded {len(df)} unique messages "
          f"({(df.label == 1).sum()} spam / {(df.label == 0).sum()} ham)")

    df["clean"] = df["text"].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=args.test_size,
        stratify=df["label"], random_state=args.seed)

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "test_size": int(len(y_test)),
    }
    print(json.dumps(metrics, indent=2))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=["Not Spam", "Spam"])
    disp.plot(cmap="Blues")
    plt.title("Spam classifier - confusion matrix")
    plt.savefig(CM_PATH, dpi=120, bbox_inches="tight")
    print(f"Saved model -> {MODEL_PATH}\nSaved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
