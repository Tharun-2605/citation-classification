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
BAND_STRATEGY = "fixed"

ID = "id"
TEXT_COLS = ["title"]
CATEGORICAL_COLS = ["subfield", "topic"]
VENUE_COL = "venue"
NUMERIC_COLS = ["n_references", "year"]

ALL_COLUMNS = [ID, "title", "year", "subfield", "topic", "venue",
               "n_references", "cites_total", "cites_2yr"]

# No code anywhere creates its own train/val/test split. The split is frozen
# once, by Person 1, in data/split.csv (columns: id, band, split) and
# everybody loads it and joins onto it by id -- see data.py. There used to
# be a TEST_SIZE/VAL_SIZE/SPLIT_JSON here for an independently-drawn split;
# removed, since two different splits would make every number incomparable.
STRATIFY = True

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = KAGGLE_INPUT.exists()

# The real pull. Not committed to git (see .gitignore) -- it lives as the
# Kaggle Dataset tharunganesh172/citations-openalex-cs-2019-2023 (file
# citations_openalex.parquet, 43,997 rows). On Kaggle with that dataset
# attached it's under /kaggle/input; locally, download it from the Kaggle
# dataset page into data/.
KAGGLE_PARQUET = (KAGGLE_INPUT / "datasets" / "tharunganesh172"
                  / "citations-openalex-cs-2019-2023" / "citations_openalex.parquet")
DATASET_PARQUET = KAGGLE_PARQUET if ON_KAGGLE else DATA_DIR / "citations_openalex.parquet"

SPLIT_CSV = DATA_DIR / "split.csv"

EMBEDDINGS_NPY = DATA_DIR / "title_embeddings.npy"
EMBEDDING_IDS_NPY = DATA_DIR / "title_embedding_ids.npy"
RESULTS_CSV = Path(os.environ.get("RESULTS_CSV", RESULTS_DIR / "results.csv"))

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
            f"split=data/split.csv kaggle={ON_KAGGLE}")


if __name__ == "__main__":
    print(describe())
    print(f"ROOT          {ROOT}")
    print(f"parquet       {DATASET_PARQUET}")
    print(f"split         {SPLIT_CSV}")
    print(f"results       {RESULTS_CSV}")
    print(f"feature sets  {list(FEATURE_SETS)}")