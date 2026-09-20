"""Interactive sentiment prediction.

Usage:
    python -m src.predict "Absolutely love this thing, works flawlessly"
    python -m src.predict            # REPL mode
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def load_model():
    return joblib.load(ARTIFACTS / "sentiment_model.joblib")


def predict_one(model, text: str) -> dict:
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    order = proba.argsort()[::-1]
    return {
        "sentiment": classes[order[0]],
        "confidence": float(proba[order[0]]),
        "probabilities": {classes[i]: round(float(proba[i]), 3) for i in order},
    }


def main() -> None:
    model = load_model()
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = predict_one(model, text)
        print(f'text : "{text}"')
        print(f"→ {result['sentiment'].upper()}  (p={result['confidence']:.2f})")
        print("   " + " · ".join(f"{k} {v:.2f}" for k, v in result["probabilities"].items()))
        return

    print("Sentiment REPL — type a sentence, Ctrl-D to exit.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            continue
        result = predict_one(model, text)
        print(f"→ {result['sentiment'].upper()}  " + " · ".join(f"{k} {v:.2f}" for k, v in result["probabilities"].items()))


if __name__ == "__main__":
    main()
