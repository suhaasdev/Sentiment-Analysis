"""Generate a 3-class sentiment corpus with realistic noise and structure.

Each text is assembled from sentiment templates + lexicon words, optionally
negated, intensified, and wrapped in social-media noise (URLs, @handles,
HTML entities, elongated chars). Neutral texts mix positive and negative
clauses so they are learnable but genuinely harder.

Usage:
    python data/generate_corpus.py   # → data/reviews.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
HERE = Path(__file__).parent

POSITIVE = [
    "amazing", "excellent", "fantastic", "love", "perfect", "brilliant", "superb",
    "outstanding", "delighted", "wonderful", "smooth", "reliable", "impressive",
    "great value", "works flawlessly", "highly recommend", "best purchase",
    "exceeded expectations", "top quality", "five stars",
]
NEGATIVE = [
    "terrible", "awful", "disappointing", "hate", "broken", "useless", "waste of money",
    "poor quality", "stopped working", "slow", "crashed", "refund", "defective",
    "never again", "worst experience", "cheap plastic", "does not work", "annoying",
    "regret buying", "one star",
]
NEUTRAL_FILLER = [
    "the package arrived on tuesday", "it comes in two colors", "the manual has 12 pages",
    "i bought this in march", "the box includes a cable", "it weighs about 300 grams",
    "the company is based in ohio", "the model number is xj-220", "it launched last year",
    "the store had it in stock", "the app connects over bluetooth", "it uses usb-c",
]
INTENSIFIERS = ["really", "very", "absolutely", "incredibly", "totally"]
NEGATIONS = ["not", "never", "hardly", "barely"]
NOISE_WRAPPERS = [
    "check {t} http://spam.link/{n}",
    "<b>{t}</b> &amp; more",
    "@user123 said {t}",
    "{t}!!! sooo {e}",
]
PRODUCTS = ["phone", "laptop", "headphones", "charger", "keyboard", "backpack", "blender", "watch"]


def _elongate(word: str) -> str:
    return word[:-1] + word[-1] * RNG.integers(2, 5)


def _make_text(sentiment: str) -> str:
    parts = []
    n_lex = int(RNG.integers(1, 3))

    if sentiment == "neutral":
        # Neutral texts mix both polarities + filler: genuinely hard to separate.
        parts.append(str(RNG.choice(POSITIVE)))
        parts.append(str(RNG.choice(NEGATIVE)))
        parts.append(str(RNG.choice(NEUTRAL_FILLER)))
    else:
        lex = POSITIVE if sentiment == "positive" else NEGATIVE
        for _ in range(n_lex):
            phrase = str(RNG.choice(lex))
            if RNG.random() < 0.35:
                phrase = f"{RNG.choice(INTENSIFIERS)} {phrase}"
            if RNG.random() < 0.30:
                # Negated sentiment word — surface form contradicts the label.
                phrase = f"{RNG.choice(NEGATIONS)} {phrase}"
            parts.append(phrase)
        if RNG.random() < 0.25:
            parts.append(str(RNG.choice(NEUTRAL_FILLER)))
    parts.append(f"for a {RNG.choice(PRODUCTS)}")
    text = ", ".join(parts)

    if RNG.random() < 0.30:
        wrapper = str(RNG.choice(NOISE_WRAPPERS))
        text = wrapper.format(t=text, n=RNG.integers(100, 999), e="goooood")
    return text


def generate(n: int = 12_000) -> pd.DataFrame:
    labels = RNG.choice(["positive", "negative", "neutral"], n, p=[0.42, 0.36, 0.22])
    rows = [{"text": _make_text(s), "sentiment": s} for s in labels]
    df = pd.DataFrame(rows)
    # 5% annotation noise — real-world corpora are never perfectly labeled.
    flip = RNG.random(n) < 0.05
    noise_labels = RNG.choice(["positive", "negative", "neutral"], n)
    df.loc[flip, "sentiment"] = noise_labels[flip]
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def main() -> None:
    out = HERE / "reviews.csv"
    df = generate()
    out.write_text(df.to_csv(index=False))
    print(f"Wrote {len(df)} texts -> {out}")
    print(df["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
