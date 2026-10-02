"""Tests for binning.py.

Two things get tested here: the `fixed` strategy (the one actually used on
the real data, agreed with Person 1 so our numbers are comparable -- see
config.BAND_STRATEGY), and the `zero_plus_quantile` strategy's defence
against silently collapsing bands on the zero-heavy target, which is why
that strategy still exists for synthetic-data dev work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
from binning import make_bands, describe_bands, BandSpec


# --- fixed (the real-data default) -------------------------------------

def test_default_strategy_is_fixed():
    # config.BAND_STRATEGY drives every call site that doesn't pass
    # strategy= explicitly. If this ever changes back, every downstream
    # number silently stops being comparable with Person 1's -- pin it.
    assert C.BAND_STRATEGY == "fixed"


def test_fixed_maps_known_values_to_agreed_bands():
    # The exact mapping Person 1 asked for: [0,1,2,3,7,8,100] ->
    # [Uncited, Low, Low, Medium, Medium, High, High].
    y = pd.Series([0, 1, 2, 3, 7, 8, 100])
    labels, spec = make_bands(y, strategy="fixed", verbose=False)
    assert list(labels) == ["Uncited", "Low", "Low", "Medium", "Medium",
                            "High", "High"]
    assert spec.edges == [0.0, 2.0, 7.0, float("inf")]
    assert spec.labels == ["Uncited", "Low", "Medium", "High"]


def test_fixed_matches_real_data_class_shares(synthetic_df):
    # Not a claim that synthetic data reproduces the real shares -- just
    # that make_bands(df["cites_2yr"]) with the default strategy runs end
    # to end and produces exactly the four agreed labels.
    labels, spec = make_bands(synthetic_df["cites_2yr"], verbose=False)
    assert spec.strategy == "fixed"
    assert spec.labels == ["Uncited", "Low", "Medium", "High"]
    assert set(pd.Series(labels).dropna().unique()) <= set(spec.labels)


def test_fixed_ignores_n_bands():
    y = pd.Series([0, 1, 5, 10])
    _, spec_default = make_bands(y, strategy="fixed", verbose=False)
    _, spec_other_n = make_bands(y, n_bands=2, strategy="fixed", verbose=False)
    assert spec_default.edges == spec_other_n.edges


def test_real_split_csv_band_column_matches_fixed_strategy():
    # Cross-check against the actual frozen split, if it's present in this
    # checkout (it is, once Person 1's data/split.csv is pulled). Skips
    # cleanly if it isn't, e.g. in a minimal checkout.
    if not C.SPLIT_CSV.exists():
        pytest.skip("data/split.csv not present in this checkout")
    split = pd.read_csv(C.SPLIT_CSV)
    label_of = dict(enumerate(["Uncited", "Low", "Medium", "High"]))
    shares = split["band"].map(label_of).value_counts(normalize=True)
    # Loose bounds -- this is a sanity cross-check, not a re-derivation.
    for label in ["Uncited", "Low", "Medium", "High"]:
        assert 0.20 < shares[label] < 0.30


# --- zero_plus_quantile (kept for synthetic-data dev work) -------------

def test_zero_plus_quantile_produces_requested_bands(synthetic_df):
    labels, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4,
                              strategy="zero_plus_quantile", verbose=False)
    assert spec.strategy == "zero_plus_quantile"
    assert len(spec.labels) == 4
    assert set(pd.Series(labels).dropna().unique()) == set(spec.labels)


def test_uncited_is_its_own_band(synthetic_df):
    labels, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4,
                              strategy="zero_plus_quantile", verbose=False)
    assert spec.labels[0] == "Uncited"
    uncited_mask = synthetic_df["cites_2yr"] == 0
    assert (pd.Series(labels)[uncited_mask] == "Uncited").all()
    assert (pd.Series(labels)[~uncited_mask] != "Uncited").all()


def test_n_bands_below_two_raises(synthetic_df):
    with pytest.raises(ValueError, match="n_bands must be at least 2"):
        make_bands(synthetic_df["cites_2yr"], n_bands=1, verbose=False)


def test_negative_target_raises():
    y = pd.Series([-1, 0, 1, 2, 3])
    with pytest.raises(ValueError, match="negative"):
        make_bands(y, n_bands=2, verbose=False)


def test_unknown_strategy_raises(synthetic_df):
    with pytest.raises(ValueError, match="unknown strategy"):
        make_bands(synthetic_df["cites_2yr"], n_bands=4,
                   strategy="not_a_real_strategy", verbose=False)


def test_band_spec_apply_is_deterministic(synthetic_df):
    labels, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4,
                              strategy="zero_plus_quantile", verbose=False)
    reapplied = spec.apply(synthetic_df["cites_2yr"])
    assert list(labels) == list(reapplied)


def test_band_spec_apply_handles_unseen_high_value(synthetic_df):
    _, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4,
                         strategy="zero_plus_quantile", verbose=False)
    huge = pd.Series([synthetic_df["cites_2yr"].max() + 10_000])
    out = spec.apply(huge)
    assert out[0] == spec.labels[-1]  # falls into the top (open-ended) band


def test_too_few_positive_rows_raises():
    # Mostly zero, far too few citing papers for 3 positive bands.
    y = pd.Series([0] * 100 + [1, 2, 3])
    with pytest.raises(ValueError, match="not enough"):
        make_bands(y, n_bands=4, strategy="zero_plus_quantile", verbose=False)


def test_plain_quantile_collapses_on_heavy_zero_mass(capsys):
    # Synthetic-shaped target: mostly zero, like the real thing. The plain
    # quantile strategy should visibly warn rather than silently return
    # fewer classes than asked.
    y = pd.Series([0] * 900 + list(range(1, 101)))
    _, spec = make_bands(y, n_bands=4, strategy="quantile", verbose=False)
    captured = capsys.readouterr()
    assert len(spec.labels) < 4
    assert "WARNING" in captured.out


def test_describe_bands_majority_matches_largest_fraction(synthetic_df):
    labels, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4,
                              strategy="zero_plus_quantile", verbose=False)
    table = describe_bands(labels, spec)
    assert table["fraction"].sum() == pytest.approx(1.0, abs=1e-6)
    assert table["fraction"].max() == pytest.approx(
        pd.Series(labels).value_counts(normalize=True).max(), abs=1e-6)
