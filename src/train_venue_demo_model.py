"""Train and serialise the venue-aware model the demo's second panel loads.

The title-only model (train_demo_model.py) can only see a typed title. This
one also sees the venue, chosen from a dropdown of the ten most frequent
training venues plus an "unknown venue" option. Feature set B4_title_venue
(title TF-IDF + venue one-hot), logistic regression, same fixed bands and
same frozen split as everything else.

Trained on train+val and scored once on the held-out test split, like the
title-only model. Its scores go to results/results_title_venue.csv, NOT the
shared results/results.csv.

Usage
-----
    python src/train_venue_demo_model.py

Writes models/demo_model_venue.joblib (gitignored). Needs
data/citations_openalex.parquet.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

import config as C
from data import load_dataset
from pipeline import build_pipeline
from binning import BandSpec
from metrics import evaluate, log_result

FEATURE_SET = "B4_title_venue"
MODEL_NAME = "logreg"
N_VENUES = 10
RESULTS_PATH = str(C.RESULTS_DIR / "results_title_venue.csv")
SPEC = BandSpec("fixed", [0.0, 2.0, 7.0, float("inf")],
                ["Uncited", "Low", "Medium", "High"])


def top_venues(train: pd.DataFrame, n: int = N_VENUES) -> list[str]:
    """The n most frequent non-null venues in the TRAINING split only."""
    counts = train["venue"].dropna().value_counts()
    return list(counts.head(n).index)


def main() -> int:
    print(f"training the venue-aware demo model: {MODEL_NAME} / {FEATURE_SET}\n")

    splits = load_dataset()
    label_of = dict(enumerate(SPEC.labels))
    dev = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    y_dev = dev["band"].map(label_of)
    y_test = splits["test"]["band"].map(label_of)

    venues = top_venues(splits["train"])
    print("venues in the dropdown:", venues, "\n")

    pipe = build_pipeline(feature_set=FEATURE_SET, model_name=MODEL_NAME,
                          model_kwargs={"class_weight": None})
    pipe.fit(dev, y_dev)

    pred = pipe.predict(splits["test"])
    result = evaluate(y_test, pred, labels=SPEC.labels, y_train=y_dev,
                      model_name=f"{MODEL_NAME} / {FEATURE_SET} (venue demo model)",
                      feature_set=FEATURE_SET, eval_split="test")
    log_result(result, notes="venue-aware demo model, trained on train+val, scored on test",
               path=RESULTS_PATH)

    C.ensure_dirs()
    out_path = C.MODELS_DIR / "demo_model_venue.joblib"
    joblib.dump({"pipeline": pipe, "spec": SPEC, "feature_set": FEATURE_SET,
                 "venues": venues}, out_path)
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
