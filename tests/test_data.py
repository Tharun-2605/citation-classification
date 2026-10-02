"""Tests for data.py -- the one place that joins the real pull against the
frozen split. No code anywhere else should build its own train/val/test
split; these tests exist to catch a regression of that rule as much as to
check the join logic itself.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data import load_split, load_dataset


@pytest.fixture
def split_csv(tmp_path):
    path = tmp_path / "split.csv"
    pd.DataFrame({
        "id": [f"W{i}" for i in range(10)],
        "band": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        "split": ["train"] * 6 + ["val"] * 2 + ["test"] * 2,
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def parquet_path(tmp_path, split_csv):
    # One id (W9) in split.csv has no matching row here, to exercise the
    # "ids in split.csv not found in the parquet" warning path.
    path = tmp_path / "citations.parquet"
    pd.DataFrame({
        "id": [f"W{i}" for i in range(9)],
        "title": [f"title {i}" for i in range(9)],
        "venue": ["Some Venue"] * 9,
    }).to_parquet(path, index=False)
    return path


def test_load_split_reads_required_columns(split_csv):
    split = load_split(split_csv)
    assert set(split.columns) >= {"id", "band", "split"}
    assert len(split) == 10


def test_load_split_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["a"], "band": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_split(path)


def test_load_split_rejects_unexpected_split_values(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["a"], "band": [0], "split": ["training"]}).to_csv(
        path, index=False)
    with pytest.raises(ValueError, match="unexpected split values"):
        load_split(path)


def test_load_split_rejects_bad_band_values(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["a"], "band": [7], "split": ["train"]}).to_csv(
        path, index=False)
    with pytest.raises(ValueError, match="band values outside 0-3"):
        load_split(path)


def test_load_dataset_raises_clear_error_when_parquet_missing(tmp_path, split_csv):
    with pytest.raises(FileNotFoundError, match="isn't committed to git"):
        load_dataset(parquet_path=tmp_path / "nope.parquet", split_path=split_csv)


def test_load_dataset_joins_by_id_and_splits_into_three(parquet_path, split_csv):
    # split.csv: W0-W5 train, W6-W7 val, W8-W9 test. parquet has W0-W8 only
    # (W9 missing), and W9 falls in the test split -- so test comes out
    # as 1 row, not 2, after the inner join.
    out = load_dataset(parquet_path=parquet_path, split_path=split_csv)
    assert set(out) == {"train", "val", "test"}
    assert len(out["train"]) == 6
    assert len(out["val"]) == 2
    assert len(out["test"]) == 1
    assert "split" not in out["train"].columns
    assert "band" in out["train"].columns


def test_load_dataset_warns_on_unmatched_ids(parquet_path, split_csv, capsys):
    load_dataset(parquet_path=parquet_path, split_path=split_csv)
    captured = capsys.readouterr()
    assert "not found in the parquet" in captured.out
