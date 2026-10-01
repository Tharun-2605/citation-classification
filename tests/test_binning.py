"""Tests for binning.py. The whole point of this module is to not silently
collapse bands on the zero-heavy target, so that failure mode is tested
directly rather than just the happy path."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from binning import make_bands, describe_bands, BandSpec


def test_zero_plus_quantile_produces_requested_bands(synthetic_df):
    labels, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4, verbose=False)
    assert spec.strategy == "zero_plus_quantile"
    assert len(spec.labels) == 4
    assert set(pd.Series(labels).dropna().unique()) == set(spec.labels)


def test_uncited_is_its_own_band(synthetic_df):
    labels, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4, verbose=False)
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
    labels, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4, verbose=False)
    reapplied = spec.apply(synthetic_df["cites_2yr"])
    assert list(labels) == list(reapplied)


def test_band_spec_apply_handles_unseen_high_value(synthetic_df):
    _, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4, verbose=False)
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
    labels, spec = make_bands(synthetic_df["cites_2yr"], n_bands=4, verbose=False)
    table = describe_bands(labels, spec)
    assert table["fraction"].sum() == pytest.approx(1.0, abs=1e-6)
    assert table["fraction"].max() == pytest.approx(
        pd.Series(labels).value_counts(normalize=True).max(), abs=1e-6)
