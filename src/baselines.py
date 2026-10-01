"""The three baselines every model in this project is measured against.

A classifier is only interesting to the extent it beats these. With a target
this imbalanced, a model can hit high accuracy by always predicting the largest
band, so a bare accuracy number means nothing on its own.

Usage
-----
    from baselines import fit_baselines, baseline_scores
    base = fit_baselines(X_train, y_train)
    baseline_scores(base, X_test, y_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

import config as C

__all__ = ["fit_baselines", "baseline_scores", "majority_baseline_accuracy"]

STRATEGIES = {
    "majority": "most_frequent",
    "stratified": "stratified",
    "uniform": "uniform",
}


def fit_baselines(X, y, seed: int = C.SEED) -> dict[str, DummyClassifier]:
    """Fit all three dummy classifiers. X is ignored by DummyClassifier but is
    required by the sklearn API, so pass whatever you pass the real models."""
    out = {}
    for name, strategy in STRATEGIES.items():
        clf = DummyClassifier(strategy=strategy, random_state=seed)
        clf.fit(X, y)
        out[name] = clf
    return out


def baseline_scores(baselines: dict, X, y, verbose: bool = True) -> pd.DataFrame:
    """Score the baselines. Returns a frame with accuracy and macro-F1."""
    rows = []
    for name, clf in baselines.items():
        pred = clf.predict(X)
        rows.append({
            "model": f"baseline: {name}",
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        })
    table = pd.DataFrame(rows).set_index("model").round(4)
    if verbose:
        print(table.to_string())
    return table


def majority_baseline_accuracy(y) -> float:
    """The number to beat. If a model's accuracy is not meaningfully above
    this, it has learned nothing."""
    s = pd.Series(np.asarray(y))
    return float(s.value_counts(normalize=True).max())


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from make_synthetic import make_synthetic
    from binning import make_bands

    df = make_synthetic(2000)
    y, _ = make_bands(df["cites_2yr"], n_bands=4, verbose=False)
    X = df[["n_references", "year"]]

    base = fit_baselines(X, y)
    print("\nbaseline scores on the training data:")
    baseline_scores(base, X, y)
    print(f"\nmajority baseline accuracy: {majority_baseline_accuracy(y):.4f}")