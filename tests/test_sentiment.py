"""Tests for the sentiment pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_corpus import generate  # noqa: E402
from src.preprocess import clean_text, tokenize  # noqa: E402


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return generate(n=1200)


def test_generation(corpus):
    assert len(corpus) == 1200
    assert set(corpus["sentiment"]) == {"positive", "negative", "neutral"}
    assert corpus["text"].str.len().min() > 5


def test_clean_text_strips_noise():
    dirty = "Check <b>THIS</b> out http://x.co/1 @user &amp; it's WONDERFUL!!!"
    cleaned = clean_text(dirty)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "&amp;" not in cleaned
    assert "wonderful" in cleaned


def test_tokenize_keeps_negations():
    tokens = tokenize("This is not good, it never works")
    assert "not" in tokens and "never" in tokens
    assert "is" not in tokens and "this" not in tokens  # stopwords removed


def test_lemmatization():
    tokens = tokenize("running dogs and better batteries")
    assert "dog" in tokens or "battery" in tokens


def test_model_pipeline_end_to_end(corpus):
    from src.preprocess import download_nltk, preprocess_series

    download_nltk(quiet=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    X = preprocess_series(corpus["text"])
    y = corpus["sentiment"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    pipe = Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("model", LogisticRegression(max_iter=500)),
    ]).fit(X_tr, y_tr)
    assert pipe.score(X_te, y_te) > 0.6
