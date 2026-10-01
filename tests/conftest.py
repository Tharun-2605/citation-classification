"""Shared fixtures for the unit test suite.

Everything here runs against synthetic data only (`make_synthetic`), per the
schema contract in SCHEMA.md. No test in this suite should touch the real
parquet or network — that keeps the suite fast and runnable before the real
data exists.
"""
from __future__ import annotations

import pytest

from make_synthetic import make_synthetic


@pytest.fixture
def synthetic_df():
    """A reasonably sized synthetic frame, fixed seed for determinism."""
    return make_synthetic(2000, seed=42)


@pytest.fixture
def small_synthetic_df():
    """A smaller frame for tests that don't need volume."""
    return make_synthetic(300, seed=42)
