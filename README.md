# 💬 Sentiment Analysis

NLP pipeline that classifies reviews and social posts as **positive, negative, or neutral** — from text preprocessing (NLTK) through TF-IDF vectorization, model comparison, error analysis, and an interactive demo CLI.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NLTK](https://img.shields.io/badge/NLTK-Preprocessing-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LinearSVC-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## ✨ Features

- **12,000 synthetic but realistic texts** across 3 classes — templated reviews and posts with sentiment-bearing lexicon, negations, intensifiers, and noise (URLs, handles, HTML entities, repeated characters)
- **Full text preprocessing** — lowercasing, URL/handle stripping, HTML entity repair, contraction expansion, tokenization, stopword removal (negation-preserving), lemmatization
- **Two vectorizers compared** — TF-IDF (word 1–2 grams) vs TF-IDF + char n-grams
- **3-model benchmark** — Multinomial Naive Bayes, Logistic Regression, Linear SVM (stratified 5-fold macro-F1 CV)
- **Error analysis** — most-confusing class pairs + top discriminating n-grams per class exported to `reports/`
- **Interactive demo** — `python -m src.predict "text here"` prints sentiment + per-class probabilities

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python -m src.preprocess --download-nltk   # one-time NLTK data download

python data/generate_corpus.py     # → data/reviews.csv
python -m src.train                # 6-combo benchmark + reports
python -m src.predict "The battery life is amazing but the screen scratches way too easily"
```

```
text : "Absolutely love this thing, works flawlessly"
→ POSITIVE  (p=0.86)
   positive 0.86 · neutral 0.14 · negative 0.00
```

Short neutral-heavy inputs can come back NEUTRAL with mixed probabilities —
the REPL view of per-class probabilities makes these cases transparent.

## 📊 Results (default seed)

6-combination benchmark (2 vectorizers × 3 models):

| Vectorizer + Model | Macro F1 (5-fold CV) |
|---|---|
| TF-IDF char+word + Linear SVM | ~0.96 |
| TF-IDF char+word + Logistic Regression | ~0.96 |
| TF-IDF word + Linear SVM | ~0.96 |
| TF-IDF word + Logistic Regression | ~0.95 |
| TF-IDF word + Naive Bayes | ~0.91 |
| TF-IDF char+word + Naive Bayes | ~0.89 |

Held-out test macro-F1 ≈ **0.97**. The corpus is deliberately challenging:
neutral texts mix positive and negative clauses, ~30% of sentiment words are
negated, and 5% of labels are annotation noise — so the NB-vs-SVM gap (~0.91
vs ~0.96) shows how model choice matters even on simple bag-of-words features.

Exact figures: `reports/metrics.json`. Confusion heatmap + error analysis:
`reports/confusion_matrix.png`, `reports/error_analysis.md`.

## 📁 Structure

```
├── data/generate_corpus.py
├── src/{preprocess,train,predict}.py
├── tests/test_sentiment.py
├── artifacts/ · reports/       # generated
└── requirements.txt
```

## 📄 License

MIT — see [LICENSE](LICENSE).
