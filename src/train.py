"""Benchmark sentiment models; train the best; export evaluation artifacts.

Usage:
    python -m src.train [--csv data/reviews.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from . import TARGET
from .preprocess import preprocess_series

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

VECTORIZERS = {
    "tfidf_word": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                          sublinear_tf=True),
    "tfidf_word_char": lambda: FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=5,
                                 max_features=200000, sublinear_tf=True)),
    ]),
}
MODELS = {
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, C=4.0),
    "linear_svm": lambda: LinearSVC(C=1.0),
}


def build_pipeline(vec_name: str, model_name: str) -> Pipeline:
    return Pipeline([
        ("vec", VECTORIZERS[vec_name]()),
        ("model", MODELS[model_name]()),
    ])


def train(csv_path: Path = ROOT / "data" / "reviews.csv") -> dict:
    sns.set_theme(style="whitegrid")
    from .preprocess import download_nltk

    download_nltk(quiet=True)

    df = pd.read_csv(csv_path)
    X = preprocess_series(df["text"])
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    cv_results = {}
    for vec_name in VECTORIZERS:
        for model_name in MODELS:
            combo = f"{vec_name}+{model_name}"
            pipe = build_pipeline(vec_name, model_name)
            scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1_macro")
            cv_results[combo] = scores
            print(f"{combo:<32} macro-F1: {scores.mean():.4f} ± {scores.std():.4f}")

    best_combo = max(cv_results, key=lambda k: cv_results[k].mean())
    vec_name, model_name = best_combo.split("+")
    print(f"\nSelected: {best_combo}")

    model = build_pipeline(vec_name, model_name)
    if model_name == "linear_svm":
        # Calibrate only the final model so predict_proba works for the demo CLI.
        model = Pipeline([("vec", VECTORIZERS[vec_name]()),
                          ("model", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "sentiment_model.joblib")
    joblib.dump(best_combo, ARTIFACTS / "model_combo.joblib")

    metrics = {
        "best_combo": best_combo,
        "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
        "cv_macro_f1": {k: {"mean": round(v.mean(), 4), "std": round(v.std(), 4)}
                        for k, v in cv_results.items()},
        "classification_report": classification_report(y_te, pred, output_dict=True),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_confusion(y_te, pred)
    _error_analysis(model, X_te, y_te)
    print(f"Test macro-F1 {metrics['test_macro_f1']} -> {REPORTS/'metrics.json'}")
    return metrics


def _plot_confusion(y_true, pred) -> None:
    labels = sorted(set(y_true) | set(pred))
    cm = confusion_matrix(y_true, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Sentiment — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _error_analysis(model, X_te: pd.Series, y_te: pd.Series, top_n: int = 10) -> None:
    """Log most-confused pairs and top discriminating features per class."""
    pred = model.predict(X_te)
    confused = (
        pd.DataFrame({"true": y_te, "pred": pred})
        .query("true != pred")
        .groupby(["true", "pred"]).size()
        .sort_values(ascending=False).head(top_n)
    )
    lines = ["# Sentiment Error Analysis\n", "## Most confused class pairs\n"]
    lines += [f"- {t} → {p}: {n} times" for (t, p), n in confused.items()]

    vec = model.named_steps["vec"]
    clf = model.named_steps["model"]
    try:
        names = vec.get_feature_names_out()
        if hasattr(clf, "coef_"):  # logistic regression
            classes = clf.classes_
            lines.append("\n## Top discriminating n-grams per class\n")
            for i, cls in enumerate(classes):
                top_idx = clf.coef_[i].argsort()[::-1][:8]
                grams = ", ".join(names[j] for j in top_idx)
                lines.append(f"- **{cls}**: {grams}")
        elif hasattr(clf, "feature_log_prob_"):  # naive bayes
            lines.append("\n## Most indicative n-grams per class (log-prob gap)\n")
            classes = clf.classes_
            gap = clf.feature_log_prob_
            for i, cls in enumerate(classes):
                top_idx = gap[i].argsort()[::-1][:8]
                grams = ", ".join(names[j] for j in top_idx)
                lines.append(f"- **{cls}**: {grams}")
        else:
            lines.append("\n(model has no linear feature attributions — skipped)")
    except Exception as exc:  # FeatureUnion etc.
        lines.append(f"\n(feature attribution unavailable: {exc})")

    out = REPORTS / "error_analysis.md"
    out.write_text("\n".join(lines))
    print(f"Error analysis -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "reviews.csv")
    args = parser.parse_args()
    train(args.csv)
