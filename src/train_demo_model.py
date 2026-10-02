"""Train and serialise the one model the demo app loads.

The demo only ever gets a typed title as input -- no venue, no subfield, no
reference count -- so it has to use the title-only feature set (B1), even
though the ablations show venue/metadata carry more signal. That's a
constraint of the input, not an oversight; see the B1 numbers in
results/results.csv for the honest accuracy this implies.

Trained on train+val (the "dev" set), matching the collaborator's own
final-test methodology, so the held-out test split stays untouched and the
reported test score means what it says.

Usage
-----
    python src/train_demo_model.py

Writes models/demo_model.joblib (gitignored -- regenerate locally, same as
the real parquet). Needs data/citations_openalex.parquet downloaded first
(see README.md for the Kaggle dataset name).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

import config as C
from data import load_dataset
from pipeline import build_pipeline
from binning import BandSpec
from metrics import evaluate, log_result

FEATURE_SET = "B1_title_only"
MODEL_NAME = "logreg"
SPEC = BandSpec("fixed", [0.0, 2.0, 7.0, float("inf")],
                ["Uncited", "Low", "Medium", "High"])


def main() -> int:
    print(f"training the demo model: {MODEL_NAME} / {FEATURE_SET}")
    print("(title-only -- the demo can't see venue/metadata, only a typed title)\n")

    splits = load_dataset()
    label_of = dict(enumerate(SPEC.labels))

    dev = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    y_dev = dev["band"].map(label_of)
    y_test = splits["test"]["band"].map(label_of)

    pipe = build_pipeline(feature_set=FEATURE_SET, model_name=MODEL_NAME,
                          model_kwargs={"class_weight": None})
    pipe.fit(dev, y_dev)

    pred = pipe.predict(splits["test"])
    result = evaluate(y_test, pred, labels=SPEC.labels, y_train=y_dev,
                      model_name=f"{MODEL_NAME} / {FEATURE_SET} (demo model)",
                      feature_set=FEATURE_SET, eval_split="test")
    log_result(result, notes="final demo model, trained on train+val, scored on test")

    C.ensure_dirs()
    out_path = C.MODELS_DIR / "demo_model.joblib"
    joblib.dump({"pipeline": pipe, "spec": SPEC, "feature_set": FEATURE_SET}, out_path)
    print(f"\nsaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
