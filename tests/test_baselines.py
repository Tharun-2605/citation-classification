"""Tests for baselines.py."""
from __future__ import annotations

import pandas as pd
import pytest

from baselines import fit_baselines, baseline_scores, majority_baseline_accuracy
from binning import make_bands


@pytest.fixture
def banded(synthetic_df):
    y, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4, verbose=False)
    X = synthetic_df[["n_references", "year"]]
    return X, y, spec


def test_fit_baselines_returns_all_three_strategies(banded):
    X, y, _ = banded
    base = fit_baselines(X, y)
    assert set(base) == {"majority", "stratified", "uniform"}


def test_majority_baseline_predicts_the_training_majority_class(banded):
    X, y, _ = banded
    base = fit_baselines(X, y)
    expected = pd.Series(y).value_counts().idxmax()
    preds = base["majority"].predict(X)
    assert set(preds) == {expected}


def test_majority_baseline_accuracy_matches_manual_calc(banded):
    _, y, _ = banded
    acc = majority_baseline_accuracy(y)
    expected = pd.Series(y).value_counts(normalize=True).max()
    assert acc == pytest.approx(expected)


def test_baseline_scores_has_accuracy_and_macro_f1_columns(banded):
    X, y, _ = banded
    base = fit_baselines(X, y)
    table = baseline_scores(base, X, y, verbose=False)
    assert set(table.columns) == {"accuracy", "macro_f1"}
    assert len(table) == 3
    assert (table["accuracy"] >= 0).all() and (table["accuracy"] <= 1).all()


def test_uniform_baseline_is_worse_than_majority_on_accuracy(banded):
    X, y, _ = banded
    base = fit_baselines(X, y)
    table = baseline_scores(base, X, y, verbose=False)
    assert table.loc["baseline: majority", "accuracy"] >= table.loc["baseline: uniform", "accuracy"]
