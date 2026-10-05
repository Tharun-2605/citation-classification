"""Tests for app/demo.py's prediction logic.

Doesn't touch the real demo_model.joblib (not available in this checkout --
it's gitignored, same as the real parquet) or launch the Gradio server.
Instead it builds a tiny in-memory model on synthetic data with the exact
same feature_set the demo uses, and monkeypatches it into the demo module's
globals, so predict() and _top_words() get exercised for real.
"""
from __future__ import annotations

import os
import sys

import gradio as gr
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import demo as demo_app
from binning import make_bands
from features import feature_names
from make_synthetic import make_synthetic
from pipeline import build_pipeline


@pytest.fixture(autouse=True)
def tiny_demo_model(monkeypatch):
    """Swap in a model trained on synthetic data, same feature_set as the
    real demo (B1_title_only), so predict() runs without the real parquet."""
    df = make_synthetic(600, seed=1)
    y, spec = make_bands(df["cites_2yr"], strategy="fixed", verbose=False)

    pipe = build_pipeline(feature_set="B1_title_only", model_name="logreg",
                          model_kwargs={"class_weight": None})
    pipe.fit(df, y)

    monkeypatch.setattr(demo_app, "_MODEL", (pipe, spec))
    monkeypatch.setattr(demo_app, "_PIPE", pipe)
    monkeypatch.setattr(demo_app, "_SPEC", spec)
    monkeypatch.setattr(demo_app, "_NAMES", feature_names(pipe))
    monkeypatch.setattr(demo_app, "_CLF", pipe.named_steps["clf"])
    monkeypatch.setattr(demo_app, "_TFIDF_STEP", pipe.named_steps["features"])
    yield


def test_predict_returns_confidence_for_every_band():
    confidences, blurb, attribution = demo_app.predict(
        "Deep learning approach to neural network optimization")
    assert set(confidences) == {"Uncited", "Low", "Medium", "High"}
    assert all(0.0 <= p <= 1.0 for p in confidences.values())
    assert sum(confidences.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_blurb_names_the_argmax_band():
    confidences, blurb, _ = demo_app.predict("Some evaluation framework")
    predicted = max(confidences, key=confidences.get)
    assert predicted in blurb


def test_predict_rejects_empty_title():
    with pytest.raises(gr.Error, match="Type a paper title first"):
        demo_app.predict("")


def test_predict_rejects_whitespace_only_title():
    with pytest.raises(gr.Error, match="Type a paper title first"):
        demo_app.predict("   ")


def test_predict_raises_clear_error_when_model_missing(monkeypatch):
    monkeypatch.setattr(demo_app, "_MODEL", None)
    with pytest.raises(gr.Error, match="train_demo_model.py"):
        demo_app.predict("some title")


def test_top_words_handles_title_with_no_recognised_words():
    # A title made entirely of characters unlikely to appear in the tiny
    # synthetic vocabulary's n-grams.
    out = demo_app._top_words("qqqqq", predicted_class="Uncited", top_n=5)
    assert isinstance(out, str)
    assert len(out) > 0  # either real attribution or the fallback message


def test_top_words_returns_requested_count_at_most():
    out = demo_app._top_words(
        "Deep learning approach to neural network optimization for graphs",
        predicted_class="Uncited", top_n=3)
    # Each shown word is one "+/-" bulleted line separated by blank lines.
    lines = [l for l in out.split("\n\n") if l.strip()]
    assert len(lines) <= 3


def test_demo_probabilities_are_attached_to_the_right_band_names():
    """Regression test for swapped Uncited/High labels. sklearn stores
    classes_ alphabetically (High, Low, Medium, Uncited), so mapping
    probabilities by position against the band order swaps them. Papers
    really in Uncited must get more Uncited probability than papers really
    in High, and the reverse must hold for High."""
    df = make_synthetic(600, seed=1)
    y, _ = make_bands(df["cites_2yr"], strategy="fixed", verbose=False)
    y = pd.Series(y.astype(str), index=df.index)

    uncited = df[y == "Uncited"].head(20)
    high = df[y == "High"].head(20)
    assert len(uncited) == 20 and len(high) == 20

    def mean_proba(frame, band):
        return sum(demo_app.predict_proba_by_name(t)[band]
                   for t in frame["title"].fillna("untitled")) / len(frame)

    assert mean_proba(uncited, "Uncited") > mean_proba(high, "Uncited")
    assert mean_proba(high, "High") > mean_proba(uncited, "High")


def test_headline_prediction_matches_highest_named_probability():
    confidences, blurb, _ = demo_app.predict("Deep learning survey of graph networks")
    top = max(confidences, key=confidences.get)
    assert blurb.startswith(f"**{top}**")


# --- title + venue panel -------------------------------------------------

@pytest.fixture
def tiny_venue_model(monkeypatch):
    """Venue-aware model on synthetic data, same B4_title_venue feature set
    the real one uses, so the venue panel is exercised without the real parquet."""
    from pipeline import build_pipeline

    df = make_synthetic(600, seed=1)
    y, spec = make_bands(df["cites_2yr"], strategy="fixed", verbose=False)
    pipe = build_pipeline(feature_set="B4_title_venue", model_name="logreg",
                          model_kwargs={"class_weight": None})
    pipe.fit(df, y)
    venues = [v for v in df["venue"].dropna().unique()[:3]]

    monkeypatch.setattr(demo_app, "_VENUE_MODEL", (pipe, spec, venues))
    monkeypatch.setattr(demo_app, "_VPIPE", pipe)
    monkeypatch.setattr(demo_app, "_VSPEC", spec)
    monkeypatch.setattr(demo_app, "_VENUES", venues)
    yield venues


def test_venue_panel_probabilities_sum_to_one(tiny_venue_model):
    confidences, blurb = demo_app.predict_venue("Deep learning survey", tiny_venue_model[0])
    assert set(confidences) == {"Uncited", "Low", "Medium", "High"}
    assert sum(confidences.values()) == pytest.approx(1.0, abs=1e-6)


def test_venue_panel_headline_matches_highest_named_probability(tiny_venue_model):
    confidences, blurb = demo_app.predict_venue("Deep learning survey", tiny_venue_model[0])
    top = max(confidences, key=confidences.get)
    assert blurb.startswith(f"**{top}**")


def test_venue_panel_accepts_unknown_venue(tiny_venue_model):
    confidences, _ = demo_app.predict_venue("Some title", demo_app.UNKNOWN_VENUE)
    assert sum(confidences.values()) == pytest.approx(1.0, abs=1e-6)


def test_venue_panel_probabilities_attached_to_the_right_band_names(tiny_venue_model):
    """Same regression as the title-only panel, for the venue model."""
    df = make_synthetic(600, seed=1)
    y, _ = make_bands(df["cites_2yr"], strategy="fixed", verbose=False)
    y = pd.Series(y.astype(str), index=df.index)
    uncited = df[y == "Uncited"].head(20)
    high = df[y == "High"].head(20)

    def mean_proba(frame, band):
        return sum(demo_app.predict_venue_proba_by_name(t, v)[band]
                   for t, v in zip(frame["title"].fillna("untitled"),
                                   frame["venue"].fillna(demo_app.UNKNOWN_VENUE))) / len(frame)

    assert mean_proba(uncited, "Uncited") > mean_proba(high, "Uncited")
    assert mean_proba(high, "High") > mean_proba(uncited, "High")


def test_venue_panel_raises_clear_error_when_model_missing(monkeypatch):
    monkeypatch.setattr(demo_app, "_VENUE_MODEL", None)
    with pytest.raises(gr.Error, match="train_venue_demo_model.py"):
        demo_app.predict_venue("some title", "IEEE Access")


def test_predict_both_returns_both_panels(tiny_venue_model):
    label, blurb, attr, v_label, v_blurb = demo_app.predict_both(
        "Deep learning survey", tiny_venue_model[0])
    assert label and blurb and v_label and v_blurb


def test_predict_both_keeps_title_panel_when_venue_model_missing(monkeypatch):
    monkeypatch.setattr(demo_app, "_VENUE_MODEL", None)
    label, blurb, attr, v_label, v_blurb = demo_app.predict_both("Deep learning survey", None)
    assert label is not None          # title-only panel still works
    assert v_label is None and "train_venue_demo_model.py" in v_blurb


def test_top_venues_uses_training_split_only_and_skips_nulls():
    import train_venue_demo_model as tv
    train = pd.DataFrame({"venue": ["A", "A", "A", "B", "B", None, "C"]})
    assert tv.top_venues(train, n=2) == ["A", "B"]
