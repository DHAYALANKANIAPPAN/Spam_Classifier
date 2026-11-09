# Spam Message Classifier

Classifies SMS / email / support-ticket text as **Spam** or **Not Spam (Ham)**.

## Dataset
[SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
(~5,500 labelled messages, Kaggle/UCI). Download `spam.csv` and place it in `data/raw/`.

## Pipeline
1. **Cleaning** - lowercase; remove URLs, HTML tags, digits and special characters (`src/preprocess.py`).
2. **NLP** - NLTK tokenisation, stop-word removal, verb lemmatisation.
3. **Features** - TF-IDF with unigrams and bigrams.
4. **Model** - Logistic Regression with balanced class weights (`src/train.py`).
5. **Evaluation** - accuracy, precision, recall, F1 and a confusion matrix, saved to `reports/`.
6. **UI** - Streamlit app with colour-coded output, confidence bar and optional CSV batch check (`app.py`).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python -m src.train          # trains, evaluates and saves models/spam_model.joblib
pytest -q                    # run tests
streamlit run app.py         # launch the demo at http://localhost:8501
```

## Results
Run `python -m src.train` to generate `reports/metrics.json` and `reports/confusion_matrix.png`.

| Metric    | Score |
|-----------|-------|
| Accuracy  | _fill in after training_ |
| Precision | _fill in_ |
| Recall    | _fill in_ |
| F1        | _fill in_ |

## Branch workflow
`main` (stable, released) <- `develop` (integration) <- `feature/*` (one per task).
