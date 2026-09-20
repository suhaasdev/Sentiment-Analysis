"""Text preprocessing with NLTK: cleaning, tokenizing, stopwords, lemmatizing."""
from __future__ import annotations

import argparse
import re
import string

import pandas as pd

# --- NLTK resources (downloaded on demand) -------------------------------
# Only what we actually use: stopwords + wordnet lemmatizer (regex tokenization).
_NLTK_RESOURCES = ["stopwords", "wordnet", "omw-1.4"]
_DOWNLOADED = False


def download_nltk(quiet: bool = False) -> None:
    """Ensure NLTK data exists. Runs at most once per process."""
    global _DOWNLOADED
    if _DOWNLOADED:
        return
    import nltk

    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(_resource_path(resource))
        except LookupError:
            try:
                nltk.download(resource, quiet=quiet)
            except Exception:  # offline / SSL issues — proceed with what we have
                pass
    _DOWNLOADED = True


def _resource_path(resource: str) -> str:
    return {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }[resource]


# --- Cleaning regexes ----------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HANDLE_RE = re.compile(r"@\w+")
_HTML_RE = re.compile(r"&\w+;")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am",
}
# Keep negation words — they flip sentiment.
_KEEP = {"not", "no", "nor", "never", "hardly", "barely", "very", "really"}

_STOP: set | None = None
_LEMMA = None


def _init_nltk_objects() -> None:
    """Lazily build the stopword set and lemmatizer (NLTK data must exist)."""
    global _STOP, _LEMMA
    if _STOP is None:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        _STOP = set(stopwords.words("english")) - _KEEP
        _LEMMA = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip noise, expand contractions; keep words only."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    for bad, good in _CONTRACTIONS.items():
        text = text.replace(bad, good)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    download_nltk(quiet=True)
    _init_nltk_objects()

    tokens = re.findall(r"[a-z]+", clean_text(text))
    return [_LEMMA.lemmatize(t) for t in tokens if t not in _STOP and len(t) > 1]


def preprocess_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda t: " ".join(tokenize(t)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NLTK resources")
    parser.add_argument("--download-nltk", action="store_true")
    args = parser.parse_args()
    if args.download_nltk:
        download_nltk(quiet=False)
        print("NLTK resources ready.")


if __name__ == "__main__":
    main()
