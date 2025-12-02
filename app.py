"""Reusable NLP preprocessing helpers for the educational visualizer."""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# 🚚 Make sure all NLTK downloads exist before running the pipeline
@lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    resource_paths = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }

    for resource, path in resource_paths.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_data()

# 🧰 Create helper objects once so the app stays fast
def _stop_words() -> set[str]:
    ensure_nltk_data()
    return set(stopwords.words("english"))


def _stemmer() -> PorterStemmer:
    ensure_nltk_data()
    return PorterStemmer()


def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data()
    return WordNetLemmatizer()


stop_words = _stop_words()
stemmer = _stemmer()
lemmatizer = _lemmatizer()


# 🔧 Text processing helpers --------------------------------------------------
def basic_cleaning(text: str) -> str:
    """Make text lowercase, remove URLs/numbers/punctuation, and trim spaces."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    """Split text into smaller word pieces (tokens)."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop very common words such as 'the' or 'is'."""
    return [token for token in tokens if token not in stop_words]


def apply_stemming(tokens: list[str]) -> list[str]:
    """Cut tokens to their rough root form."""
    return [stemmer.stem(token) for token in tokens]


def apply_lemmatization(tokens: list[str]) -> list[str]:
    """Find the proper dictionary form of each token."""
    return [lemmatizer.lemmatize(token) for token in tokens]


def run_pipeline(raw_text: str) -> dict:
    """Run all preprocessing steps and keep every result."""
    cleaned = basic_cleaning(raw_text)
    tokens = tokenize_text(cleaned)
    tokens_no_sw = remove_stopwords(tokens)
    stemmed = apply_stemming(tokens_no_sw)
    lemmas = apply_lemmatization(tokens_no_sw)

    return {
        "cleaned": cleaned,
        "tokens": tokens,
        "tokens_no_sw": tokens_no_sw,
        "stemmed": stemmed,
        "lemmas": lemmas,
    }


if __name__ == "__main__":
    sample_text = "Natural Language Processing helps computers understand people."
    report = run_pipeline(sample_text)
    for key, value in report.items():
        print(f"--- {key.upper()} ---")
        print(value)
        print()
