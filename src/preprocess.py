"""Text cleaning and NLTK-based preprocessing for spam classification."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_resources() -> None:
    """Download the NLTK resources we need, only if they are missing."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_resources()

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
HTML_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")  # drops digits, punctuation, symbols
STOP_WORDS = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, HTML tags, numbers and special characters."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenise -> drop stop words -> lemmatise."""
    tokens = word_tokenize(clean_text(text))
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    # Verb lemmatisation so 'winning' -> 'win', as in the project spec.
    return " ".join(_lemmatizer.lemmatize(t, pos="v") for t in tokens)
