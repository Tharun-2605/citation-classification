"""Tests for models.py."""
from __future__ import annotations

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from models import get_model, MODELS, is_sparse_safe, available, NON_NEGATIVE_ONLY

EXPECTED_TYPES = {
    "logreg": LogisticRegression,
    "complement_nb": ComplementNB,
    "linear_svc": LinearSVC,
    "random_forest": RandomForestClassifier,
    "hist_gbm": HistGradientBoostingClassifier,
    "mlp": MLPClassifier,
}


@pytest.mark.parametrize("name,expected_type", EXPECTED_TYPES.items())
def test_get_model_returns_expected_type(name, expected_type):
    clf = get_model(name)
    assert isinstance(clf, expected_type)


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="unknown model"):
        get_model("not_a_real_model")


def test_balanced_class_weight_set_where_supported():
    for name in ("logreg", "linear_svc", "random_forest", "hist_gbm"):
        clf = get_model(name)
        assert "balanced" in str(clf.get_params()["class_weight"])


def test_kwargs_override_defaults():
    clf = get_model("logreg", C=5.0)
    assert clf.C == 5.0


def test_lightgbm_falls_back_to_hist_gbm_when_not_installed():
    # In this environment lightgbm is not installed (see requirements.txt --
    # it's commented out until Day 2/3), so get_model("lightgbm") must fall
    # back to HistGradientBoostingClassifier rather than raising ImportError.
    if available()["lightgbm"]:
        pytest.skip("lightgbm is installed in this environment")
    clf = get_model("lightgbm")
    assert isinstance(clf, HistGradientBoostingClassifier)


def test_is_sparse_safe_tracks_the_lightgbm_fallback():
    # The whole point of is_sparse_safe being a function and not a static
    # dict: when lightgbm isn't installed, "lightgbm" actually resolves to
    # the dense-only HistGradientBoosting, so it must report False.
    if available()["lightgbm"]:
        assert is_sparse_safe("lightgbm") is True
    else:
        assert is_sparse_safe("lightgbm") is False


def test_hist_gbm_is_never_sparse_safe():
    assert is_sparse_safe("hist_gbm") is False


def test_all_models_are_constructible():
    for name in MODELS:
        clf = get_model(name)
        assert clf is not None


def test_non_negative_only_set_is_just_complement_nb():
    assert NON_NEGATIVE_ONLY == {"complement_nb"}
