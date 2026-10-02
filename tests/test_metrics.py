"""Tests for metrics.py.

The one property this module exists to guarantee is that the majority
baseline is computed from y_train, not y_test -- evaluate() would quietly
flatter the baseline otherwise. That gets a dedicated test with a train/test
split that disagrees about which class is largest.

`labels` is a required argument (not optional) -- without an explicit order,
per-class columns and the confusion matrix silently fall back to alphabetical
and scramble between the two team members' runs. Every evaluate() call here
passes it explicitly, same as production code must.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from metrics import evaluate, confusion_frame, log_result, results_table, compare


def test_labels_is_required():
    with pytest.raises(ValueError, match="labels is required"):
        evaluate(["a", "b"], ["a", "a"], labels=None, verbose=False)


def test_evaluate_returns_expected_keys():
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "a", "a", "b"]
    result = evaluate(y_true, y_pred, labels=["a", "b"], model_name="toy",
                      verbose=False)
    expected_keys = {
        "model", "feature_set", "eval_split", "accuracy", "macro_f1",
        "weighted_f1", "balanced_accuracy", "baseline_accuracy",
        "baseline_macro_f1", "accuracy_lift", "macro_f1_lift",
        "random_accuracy", "n_test", "n_classes", "f1_a", "f1_b",
    }
    assert expected_keys <= set(result)
    assert result["n_test"] == 4
    assert result["n_classes"] == 2
    assert result["feature_set"] is None  # not passed -> stays None
    assert result["eval_split"] is None


def test_evaluate_logs_feature_set_and_eval_split():
    result = evaluate(["a", "b"], ["a", "b"], labels=["a", "b"],
                      model_name="toy", feature_set="A2_no_venue",
                      eval_split="test", verbose=False)
    assert result["feature_set"] == "A2_no_venue"
    assert result["eval_split"] == "test"


def test_evaluate_uses_training_majority_not_test_majority():
    # Training data is almost all "a". Test data is almost all "b". A
    # baseline computed from y_test would report near-perfect "majority"
    # accuracy; the honest baseline (from y_train) should score badly on
    # this test set.
    y_train = ["a"] * 90 + ["b"] * 10
    y_test = ["b"] * 9 + ["a"] * 1
    y_pred = y_test  # irrelevant to this check
    result = evaluate(y_test, y_pred, labels=["a", "b"], y_train=y_train,
                      model_name="toy", verbose=False)
    assert result["baseline_accuracy"] == pytest.approx(0.1)  # "a" on mostly-"b" test


def test_evaluate_falls_back_to_y_true_without_y_train():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "a", "a"]
    result = evaluate(y_true, y_pred, labels=["a", "b"], model_name="toy",
                      verbose=False)
    # majority class of y_true is "a", occurring 3/4 of the time.
    assert result["baseline_accuracy"] == pytest.approx(0.75)


def test_per_class_f1_columns_match_classification_report():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    result = evaluate(y_true, y_pred, labels=["a", "b"], model_name="toy",
                      verbose=False)
    # a: 1 true, predicted once correctly (precision 1, recall .5, f1 2/3)
    # b: 2 true + 1 false positive, both found (precision 2/3, recall 1, f1 .8)
    assert result["f1_a"] == pytest.approx(2 / 3, abs=1e-3)
    assert result["f1_b"] == pytest.approx(0.8, abs=1e-3)


def test_confusion_frame_shape_matches_labels():
    y_true = ["a", "b", "c", "a"]
    y_pred = ["a", "b", "b", "a"]
    cm = confusion_frame(y_true, y_pred, labels=["a", "b", "c"])
    assert cm.shape == (3, 3)
    assert list(cm.index) == ["true a", "true b", "true c"]
    assert list(cm.columns) == ["pred a", "pred b", "pred c"]


def test_log_result_appends_without_duplicating_header(tmp_path):
    path = tmp_path / "results.csv"
    r1 = evaluate(["a", "b"], ["a", "a"], labels=["a", "b"], model_name="m1",
                 verbose=False)
    r2 = evaluate(["a", "b"], ["b", "b"], labels=["a", "b"], model_name="m2",
                 verbose=False)
    log_result(r1, notes="first", path=str(path))
    log_result(r2, notes="second", path=str(path))

    lines = path.read_text().splitlines()
    assert lines[0].startswith("model,")  # single header
    assert sum(1 for l in lines if l.startswith("model,")) == 1
    df = pd.read_csv(path)
    assert len(df) == 2
    assert list(df["model"]) == ["m1", "m2"]


def test_results_table_sorts_by_macro_f1_descending(tmp_path):
    path = tmp_path / "results.csv"
    low = evaluate(["a", "a", "b"], ["a", "a", "a"], labels=["a", "b"],
                   model_name="low", verbose=False)
    high = evaluate(["a", "b"], ["a", "b"], labels=["a", "b"],
                    model_name="high", verbose=False)
    log_result(low, path=str(path))
    log_result(high, path=str(path))

    table = results_table(path=str(path))
    assert list(table["model"]) == ["high", "low"]


def test_results_table_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        results_table(path=str(tmp_path / "nope.csv"))


def test_compare_lines_up_results_by_model():
    r1 = evaluate(["a", "b"], ["a", "b"], labels=["a", "b"],
                 model_name="perfect", verbose=False)
    r2 = evaluate(["a", "b"], ["b", "a"], labels=["a", "b"],
                 model_name="reversed", verbose=False)
    table = compare(r1, r2)
    assert list(table.index) == ["perfect", "reversed"]
    assert table.loc["perfect", "accuracy"] == 1.0
    assert table.loc["reversed", "accuracy"] == 0.0
