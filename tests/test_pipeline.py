"""Tests for pipeline.py -- the single assembled Pipeline that the ablations
and train.py actually call.

The two things most likely to break silently here: a dense-only model fed
sparse TF-IDF/one-hot output (crashes inside .fit with a confusing scipy
error if DenseTransformer isn't wired in), and a non-negative-only model
(ComplementNB) fed StandardScaler'd numeric features that can go negative
(crashes inside .fit with a confusing "Negative values in data" error).
Both are caught up front here with a clear message instead.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from binning import make_bands
from pipeline import build_pipeline, DenseTransformer


@pytest.fixture
def banded_split(synthetic_df):
    y, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4, verbose=False)
    train, test, ytr, yte = train_test_split(
        synthetic_df, y, test_size=0.2, random_state=42, stratify=y)
    return train, test, ytr, yte, spec


@pytest.mark.parametrize("feature_set,model_name", [
    ("A1_all", "logreg"),
    ("A2_no_venue", "linear_svc"),
    ("A3_venue_only", "random_forest"),
    ("A3_venue_only", "hist_gbm"),       # dense-only model on sparse features
    ("B1_title_only", "complement_nb"),  # non-negative-only model
    ("B3_meta_only", "mlp"),
])
def test_pipeline_fits_and_predicts(feature_set, model_name, banded_split):
    train, test, ytr, yte, spec = banded_split
    pipe = build_pipeline(feature_set=feature_set, model_name=model_name)
    pipe.fit(train, ytr)
    pred = pipe.predict(test)
    assert len(pred) == len(test)
    assert set(pred) <= set(spec.labels)


def test_dense_only_model_gets_a_to_dense_step(banded_split):
    train, _, ytr, _, _ = banded_split
    pipe = build_pipeline(feature_set="A1_all", model_name="hist_gbm")
    assert "to_dense" in pipe.named_steps
    pipe.fit(train, ytr)  # would raise if features stayed sparse


def test_sparse_safe_model_has_no_to_dense_step(banded_split):
    pipe = build_pipeline(feature_set="A1_all", model_name="logreg")
    assert "to_dense" not in pipe.named_steps


def test_complement_nb_with_use_meta_raises_clearly():
    # A1_all has use_meta=True -> StandardScaler'd numeric columns -> can go
    # negative -> ComplementNB would blow up deep inside .fit without this
    # upfront check.
    with pytest.raises(ValueError, match="non-negative"):
        build_pipeline(feature_set="A1_all", model_name="complement_nb")


def test_complement_nb_without_use_meta_is_fine(banded_split):
    train, test, ytr, _, _ = banded_split
    pipe = build_pipeline(feature_set="B1_title_only", model_name="complement_nb")
    pipe.fit(train, ytr)
    pred = pipe.predict(test)
    assert len(pred) == len(test)


def test_unknown_feature_set_raises():
    with pytest.raises(KeyError, match="unknown feature set"):
        build_pipeline(feature_set="not_a_real_set", model_name="logreg")


def test_feature_overrides_change_column_count(synthetic_df):
    with_char = build_pipeline(feature_set="B1_title_only", model_name="logreg")
    without_char = build_pipeline(feature_set="B1_title_only", model_name="logreg",
                                  feature_overrides={"char_ngrams": False})
    with_char.named_steps["features"].fit(synthetic_df)
    without_char.named_steps["features"].fit(synthetic_df)
    n_with = len(with_char.named_steps["features"].get_feature_names_out())
    n_without = len(without_char.named_steps["features"].get_feature_names_out())
    assert n_without < n_with


def test_fit_on_train_predict_on_test_with_unseen_category(synthetic_df):
    # Same leakage-style check as test_features.py, but now through the full
    # assembled pipeline including the classifier.
    y, _ = make_bands(synthetic_df["cites_2yr"], n_bands=4, verbose=False)
    train, test, ytr, yte = train_test_split(
        synthetic_df, y, test_size=0.2, random_state=42, stratify=y)
    test = test.copy()
    test.iloc[0, test.columns.get_loc("venue")] = "A Venue Never Seen In Training"
    pipe = build_pipeline(feature_set="A1_all", model_name="logreg")
    pipe.fit(train, ytr)
    pred = pipe.predict(test)  # must not raise
    assert len(pred) == len(test)


def test_dense_transformer_passes_through_dense_input_unchanged():
    dt = DenseTransformer()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = dt.transform(X)
    np.testing.assert_array_equal(out, X)


def test_dense_transformer_densifies_sparse_input():
    from scipy import sparse
    dt = DenseTransformer()
    X = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0]]))
    out = dt.transform(X)
    assert not sparse.issparse(out)
    np.testing.assert_array_equal(out, np.array([[1.0, 0.0], [0.0, 4.0]]))
