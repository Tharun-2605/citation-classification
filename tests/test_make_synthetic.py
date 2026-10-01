"""Tests for make_synthetic.py: shape, reproducibility, and validate()'s
ability to actually catch broken data (not just pass on good data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
from make_synthetic import make_synthetic, validate, REQUIRED


def test_shape_and_columns(synthetic_df):
    assert len(synthetic_df) == 2000
    assert set(synthetic_df.columns) == set(REQUIRED)


def test_dtypes_match_required(synthetic_df):
    # pandas' default string dtype has shifted across versions (plain
    # "object" historically, "str"/StringDtype on newer pandas with pyarrow
    # available) -- accept either for the "str" columns in REQUIRED, since
    # what actually matters is int columns being int and string columns
    # holding strings, not the exact backing dtype.
    for col, dtype in REQUIRED.items():
        actual = synthetic_df[col].dtype
        if dtype == "object":
            assert actual == object or str(actual) == "str" or str(actual).startswith("string")
        else:
            assert str(actual) == dtype


def test_years_within_frozen_range(synthetic_df):
    assert set(synthetic_df["year"].unique()) <= set(C.YEARS)


def test_ids_are_unique(synthetic_df):
    assert not synthetic_df["id"].duplicated().any()


def test_same_seed_is_reproducible():
    a = make_synthetic(500, seed=7)
    b = make_synthetic(500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = make_synthetic(500, seed=7)
    b = make_synthetic(500, seed=8)
    assert not a["cites_2yr"].equals(b["cites_2yr"])


def test_null_rates_roughly_match_request():
    n = 5000
    df = make_synthetic(n, seed=1, null_title_rate=0.1, null_venue_rate=0.2)
    assert df["title"].isna().mean() == pytest.approx(0.1, abs=0.01)
    assert df["venue"].isna().mean() == pytest.approx(0.2, abs=0.01)


def test_cites_2yr_never_exceeds_cites_total(synthetic_df):
    # cites_total = cites_2yr + tail, tail >= 0, so this must always hold.
    assert (synthetic_df["cites_2yr"] <= synthetic_df["cites_total"]).all()


# --- validate() -------------------------------------------------------

def test_validate_passes_on_default_output(synthetic_df):
    result = validate(synthetic_df, verbose=False)
    assert result["problems"] == []
    assert result["year_flat"] is True


def test_validate_raises_on_missing_column(synthetic_df):
    broken = synthetic_df.drop(columns=["venue"])
    with pytest.raises(ValueError, match="missing columns"):
        validate(broken, verbose=False)


def test_validate_flags_negative_counts(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[0, "cites_2yr"] = -1
    result = validate(broken, verbose=False)
    assert any("negative" in p for p in result["problems"])


def test_validate_flags_impossible_cites_2yr_over_total(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[0, "cites_2yr"] = broken.loc[0, "cites_total"] + 100
    result = validate(broken, verbose=False)
    assert any("impossible" in p for p in result["problems"])


def test_validate_flags_year_trend():
    # Construct data where cites_2yr trends hard with year -- target logic
    # broken by construction -- and confirm validate() catches it rather than
    # silently passing.
    n = 2000
    rng = np.random.default_rng(0)
    df = make_synthetic(n, seed=0)
    df = df.sort_values("year").reset_index(drop=True)
    df["cites_2yr"] = np.arange(n)  # perfectly increasing with year order
    result = validate(df, verbose=False)
    assert result["year_flat"] is False
    assert any("trends with year" in p for p in result["problems"])
