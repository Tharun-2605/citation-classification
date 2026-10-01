"""Single source of truth for constants both team members depend on.

If the two of us use different seeds, different band counts or different
splits, the comparison table in the report is meaningless. Everything that
must match lives here and nowhere else.

Do not hardcode a seed, a path or a column name anywhere but this file.
"""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42

YEARS = [2019, 2020, 2021, 2022, 2023]
CITATION_WINDOW = 2
TARGET = "cites_2yr"

N_BANDS = 4
BAND_STRATEGY = "zero_plus_quantile"

ID = "id"
TEXT_COLS = ["title"]
CATEGORICAL_COLS = ["subfield", "topic"]
VENUE_COL = "venue"
NUMERIC_COLS = ["n_references", "year"]

ALL_COLUMNS = [ID, "title", "year", "subfield", "topic", "venue",
               "n_references", "cites_total", "cites_2yr"]

TEST_SIZE = 0.20
VAL_SIZE = 0.10
STRATIFY = True

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

DATASET_PARQUET = DATA_DIR / "openalex_cs_2019_2023.parquet"
EMBEDDINGS_NPY = DATA_DIR / "title_embeddings.npy"
EMBEDDING_IDS_NPY = DATA_DIR / "title_embedding_ids.npy"
SPLIT_JSON = DATA_DIR / "split_indices.json"
RESULTS_CSV = Path(os.environ.get("RESULTS_CSV", RESULTS_DIR / "results.csv"))

KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = KAGGLE_INPUT.exists()

TFIDF_WORD = dict(ngram_range=(1, 2), min_df=3, max_features=50_000,
                  sublinear_tf=True, strip_accents="unicode")
TFIDF_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                  max_features=50_000, sublinear_tf=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


FEATURE_SETS = {
    "A1_all":            dict(use_title=True,  use_meta=True,  use_venue=True),
    "A2_no_venue":       dict(use_title=True,  use_meta=True,  use_venue=False),
    "A3_venue_only":     dict(use_title=False, use_meta=False, use_venue=True),
    "B1_title_only":     dict(use_title=True,  use_meta=False, use_venue=False),
    "B2_title_plus_meta": dict(use_title=True, use_meta=True,  use_venue=False),
    "B3_meta_only":      dict(use_title=False, use_meta=True,  use_venue=False),
}


def ensure_dirs() -> None:
    """Create the output directories. Safe to call repeatedly."""
    for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """One-line summary for the top of a notebook, so a run is self-documenting."""
    return (f"seed={SEED} bands={N_BANDS}({BAND_STRATEGY}) "
            f"years={YEARS[0]}-{YEARS[-1]} window={CITATION_WINDOW}y "
            f"test={TEST_SIZE} val={VAL_SIZE} kaggle={ON_KAGGLE}")


if __name__ == "__main__":
    print(describe())
    print(f"ROOT          {ROOT}")
    print(f"parquet       {DATASET_PARQUET}")
    print(f"results       {RESULTS_CSV}")
    print(f"feature sets  {list(FEATURE_SETS)}")