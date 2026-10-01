"""Evaluation. Every score comes with its baseline attached.

The rule this module exists to enforce: a number like "0.94 accuracy" is
meaningless on this problem until you know the majority-class baseline sitting
next to it. `evaluate()` will not report one without the other.

Both team members use this module, the same split, and the same CV harness.
Otherwise the numbers in the report are not comparable.

Usage
-----
    from metrics import evaluate, log_result, results_table
    res = evaluate(y_test, y_pred, y_train=y_train, model_name="LogReg + TFIDF")
    log_result(res, notes="title only, no venue")
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)

import config as C

__all__ = ["evaluate", "confusion_frame", "log_result", "results_table",
           "compare"]

RESULTS_PATH = str(C.RESULTS_CSV)


def evaluate(
    y_true,
    y_pred,
    y_train=None,
    model_name: str = "unnamed",
    labels: list | None = None,
    verbose: bool = True,
) -> dict:
    """Score a model and print it against its baselines.

    Parameters
    ----------
    y_true, y_pred : the test labels and the model's predictions
    y_train : training labels. Used to compute the majority baseline the way it
        would actually be applied — predicting the *training* majority class on
        the test set. Falls back to y_true if not given, which slightly
        flatters the baseline; pass y_train.
    model_name : goes into the results log
    labels : class order for the confusion matrix

    Returns
    -------
    dict of metrics, ready for `log_result`
    """
    y_true = pd.Series(np.asarray(y_true))
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = list(pd.unique(pd.concat(
            [y_true, pd.Series(y_pred)]).dropna()))
        try:
            labels = sorted(labels)
        except TypeError:
            pass

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    balanced = balanced_accuracy_score(y_true, y_pred)

    # Majority baseline, applied honestly: the class most common in TRAINING,
    # scored on the test set.
    ref = pd.Series(np.asarray(y_train)) if y_train is not None else y_true
    majority_class = ref.value_counts().idxmax()
    base_pred = np.full(len(y_true), majority_class, dtype=object)
    base_acc = accuracy_score(y_true, base_pred)
    base_macro = f1_score(y_true, base_pred, average="macro", zero_division=0)

    n_classes = len(labels)
    random_acc = 1.0 / n_classes if n_classes else float("nan")

    result = {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro), 4),
        "weighted_f1": round(float(weighted), 4),
        "balanced_accuracy": round(float(balanced), 4),
        "baseline_accuracy": round(float(base_acc), 4),
        "baseline_macro_f1": round(float(base_macro), 4),
        "accuracy_lift": round(float(acc - base_acc), 4),
        "macro_f1_lift": round(float(macro - base_macro), 4),
        "random_accuracy": round(float(random_acc), 4),
        "n_test": int(len(y_true)),
        "n_classes": int(n_classes),
    }

    if verbose:
        print(f"\n{'=' * 62}")
        print(f"  {model_name}")
        print(f"{'=' * 62}")
        print(f"  accuracy           {acc:.4f}    "
              f"(majority baseline {base_acc:.4f}, "
              f"lift {acc - base_acc:+.4f})")
        print(f"  macro F1           {macro:.4f}    "
              f"(majority baseline {base_macro:.4f}, "
              f"lift {macro - base_macro:+.4f})")
        print(f"  balanced accuracy  {balanced:.4f}")
        print(f"  weighted F1        {weighted:.4f}")
        print(f"  random guess       {random_acc:.4f}  "
              f"({n_classes} classes, n={len(y_true):,})")

        verdict = _verdict(acc, base_acc, macro, base_macro)
        print(f"\n  >> {verdict}")

        print("\n  per class:")
        rep = classification_report(y_true, y_pred, labels=labels,
                                    zero_division=0, output_dict=True)
        for lab in labels:
            r = rep.get(str(lab), rep.get(lab))
            if r:
                print(f"    {str(lab):<14} precision {r['precision']:.3f}  "
                      f"recall {r['recall']:.3f}  f1 {r['f1-score']:.3f}  "
                      f"n={int(r['support'])}")

        print("\n  confusion matrix (rows = true, cols = predicted):")
        cm = confusion_frame(y_true, y_pred, labels)
        print("    " + cm.to_string().replace("\n", "\n    "))

    return result


def _verdict(acc, base_acc, macro, base_macro) -> str:
    """A blunt read on whether the model did anything.

    Macro F1 is the headline metric on this problem, not accuracy. With an
    imbalanced target, `class_weight="balanced"` deliberately sacrifices
    accuracy to pick up the minority bands, so accuracy below the majority
    baseline is expected there and is not a failure by itself.
    """
    acc_up = acc > base_acc + 1e-9
    macro_up = macro > base_macro + 0.02

    if not acc_up and not macro_up:
        return ("NO BETTER THAN THE MAJORITY BASELINE on either metric. "
                "Do not report this as a working model.")

    if not acc_up and macro_up:
        return ("Accuracy is below the majority baseline but macro F1 is "
                "above it — the model is spreading predictions across bands "
                "instead of collapsing to the largest one. Expected with "
                "class_weight='balanced'. Report macro F1 as the headline "
                "and say why.")

    if acc_up and not macro_up:
        return ("Accuracy is up but macro F1 is flat — the model is mostly "
                "predicting the majority class and the accuracy gain is "
                "hollow. Check the confusion matrix before claiming anything.")

    if acc - base_acc < 0.02 and macro - base_macro < 0.05:
        return ("Marginal gain on both metrics. Honest framing: weak signal "
                "in these features. That is a legitimate finding, not a "
                "failure.")

    return "Beats the majority baseline on both accuracy and macro F1."


def confusion_frame(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame, ready to print or plot."""
    if labels is None:
        labels = sorted(pd.unique(pd.Series(np.asarray(y_true)).dropna()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm,
                        index=[f"true {l}" for l in labels],
                        columns=[f"pred {l}" for l in labels])


def log_result(result: dict, notes: str = "", path: str | None = None) -> str:
    """Append one result row to the shared results.csv. Both team members write
    to the same file so the comparison table is assembled automatically."""
    path = path or RESULTS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    row = dict(result)
    row["notes"] = notes
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    frame = pd.DataFrame([row])
    header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=header, index=False)
    print(f"  logged to {path}")
    return path


def results_table(path: str | None = None,
                  sort_by: str = "macro_f1") -> pd.DataFrame:
    """Read the shared log back as the comparison table for the report."""
    path = path or RESULTS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"no results at {path} — nothing logged yet")
    df = pd.read_csv(path)
    cols = ["model", "accuracy", "baseline_accuracy", "accuracy_lift",
            "macro_f1", "baseline_macro_f1", "macro_f1_lift", "notes"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(sort_by, ascending=False).reset_index(drop=True)


def compare(*results: dict) -> pd.DataFrame:
    """Put several `evaluate` outputs side by side without touching the log."""
    return (pd.DataFrame(list(results))
            .set_index("model")[["accuracy", "baseline_accuracy",
                                 "accuracy_lift", "macro_f1",
                                 "macro_f1_lift"]])


if __name__ == "__main__":
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(__file__))
    from make_synthetic import make_synthetic
    from binning import make_bands
    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyClassifier

    df = make_synthetic(3000)
    y, spec = make_bands(df["cites_2yr"], n_bands=4, verbose=False)
    X = df[["n_references", "year"]]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    r = evaluate(yte, clf.predict(Xte), y_train=ytr,
                 model_name="DummyClassifier (sanity check)",
                 labels=spec.labels)
    print("\nexpected: the verdict above says it is no better than guessing.")