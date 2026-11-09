from src.preprocess import clean_text, preprocess


def test_clean_text_removes_urls_html_digits_and_symbols():
    raw = "WIN <b>£1000</b> now!!! Visit https://spam.example/x or www.bad.com"
    assert clean_text(raw) == "win now visit or"


def test_preprocess_removes_stopwords_and_lemmatises_verbs():
    tokens = preprocess("The team is winning the match").split()
    assert "the" not in tokens and "is" not in tokens
    assert "win" in tokens


def test_preprocess_handles_empty_input():
    assert preprocess("") == ""
