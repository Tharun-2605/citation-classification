"""Load the real pull and join it against the one frozen split.

Per the shared split contract (see config.py): nobody creates their own
train/val/test split. Person 1 froze it once in data/split.csv (columns id,
band, split) and everybody loads that file and joins it onto the pulled
parquet by id. This is the one place that does that join, so train.py,
pipeline.py and anything in notebooks/ all see the same train/val/test rows.

    from data import load_dataset
    splits = load_dataset()
    splits["train"], splits["val"], splits["test"]
"""
from __future__ import annotations

import pandas as pd

import config as C

__all__ = ["load_split", "load_dataset"]


def load_split(path=None) -> pd.DataFrame:
    """The frozen split: columns id, band, split (train/val/test).

    `band` is the integer class index already assigned by the fixed binning
    (0=Uncited, 1=Low, 2=Medium, 3=High -- see binning.py's `fixed`
    strategy). It is not recomputed here; split.csv is the one source of
    truth for it, same as for the split itself.
    """
    path = path or C.SPLIT_CSV
    df = pd.read_csv(path)

    required = {"id", "band", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    bad_splits = set(df["split"].unique()) - {"train", "val", "test"}
    if bad_splits:
        raise ValueError(f"{path} has unexpected split values: {sorted(bad_splits)}")

    bad_bands = set(df["band"].unique()) - {0, 1, 2, 3}
    if bad_bands:
        raise ValueError(f"{path} has band values outside 0-3: {sorted(bad_bands)}")

    return df


def load_dataset(parquet_path=None, split_path=None) -> dict[str, pd.DataFrame]:
    """Load the real parquet, join the frozen split onto it by id, and
    return {"train": df, "val": df, "test": df}.

    Each frame is on the frozen schema (SCHEMA.md) plus a `band` column
    (the int class index from split.csv -- see binning.BandSpec for turning
    that back into Uncited/Low/Medium/High labels).

    Raises FileNotFoundError with a clear message if the parquet hasn't been
    downloaded yet -- it's not committed to git, it lives on Kaggle (see
    README.md for the dataset name).
    """
    parquet_path = parquet_path or C.DATASET_PARQUET
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. The real parquet isn't committed to "
            "git (see README.md for the Kaggle dataset name) -- download it "
            "into data/, or run this on Kaggle with the dataset attached.")

    df = pd.read_parquet(parquet_path)
    split = load_split(split_path)

    merged = df.merge(split[["id", "band", "split"]], on="id", how="inner")
    if len(merged) != len(split):
        print(f"  warning: {len(split) - len(merged)} ids in {split_path or C.SPLIT_CSV} "
              "were not found in the parquet")

    return {
        name: merged[merged["split"] == name].drop(columns=["split"]).reset_index(drop=True)
        for name in ("train", "val", "test")
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    split = load_split()
    print(f"split.csv: {len(split):,} rows")
    print(split["split"].value_counts().to_string())
    print()
    print("band distribution:")
    print(split["band"].value_counts(normalize=True).sort_index().round(4).to_string())

    try:
        splits = load_dataset()
        for name, d in splits.items():
            print(f"{name:>5}: {len(d):,} rows")
    except FileNotFoundError as e:
        print(f"\n(parquet not downloaded locally -- {e})")
