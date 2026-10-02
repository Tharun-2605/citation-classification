"""Feature pipeline.

One `build_features()` call returns a fitted-in-place sklearn transformer for
whichever feature set you ask for. The ablations on Day 4 are the same function
called with different flags — no separate code paths, which is what makes the
comparison honest.

    from features import build_features
    from config import FEATURE_SETS

    pipe = build_features(**FEATURE_SETS["A2_no_venue"])
    X = pipe.fit_transform(df_train)

Nulls: `title` and `venue` come back null from OpenAlex. Everything here
handles that; do not pre-fill them in the notebook.

Embeddings: computed once (GPU, on Kaggle), cached to .npy, then looked up by
paper id inside the pipeline. Unsupervised, so there is no train/test leakage
in precomputing them over the whole dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

__all__ = ["build_features", "TextSelector", "EmbeddingLookup", "Log1pColumns",
           "compute_embeddings", "load_embeddings", "feature_names"]

class TextSelector(BaseEstimator, TransformerMixin):
    """Pull one text column out as a 1-D array of strings, nulls -> "".

    TfidfVectorizer needs a flat iterable of strings, not a DataFrame column,
    and it throws on NaN. This does both jobs.
    """

    def __init__(self, column: str = "title"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            s = X[self.column]
        else:
            s = pd.Series(np.asarray(X).ravel())
        return s.fillna("").astype(str).values

    def get_feature_names_out(self, input_features=None):
        # Passes a single text column through to the TfidfVectorizer step
        # that follows it in the pipeline. Without this, ColumnTransformer's
        # get_feature_names_out() raises AttributeError on any feature set
        # with use_title=True -- which is what features.feature_names() and
        # the demo's attribution panel both call.
        return np.asarray([self.column])


class Log1pColumns(BaseEstimator, TransformerMixin):
    """log1p a subset of numeric columns, by position, before scaling.

    `n_references` is heavy-tailed; log1p-ing it before StandardScaler
    measurably helped the linear models on the real data (confirmed by
    Person 1). Trees don't care either way. `year` is left alone -- it's
    already a small, roughly-linear range.
    """

    def __init__(self, column_indices: tuple[int, ...] = ()):
        self.column_indices = column_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for i in self.column_indices:
            X[:, i] = np.log1p(np.clip(X[:, i], a_min=0, a_max=None))
        return X

    def get_feature_names_out(self, input_features=None):
        # Passes names through unchanged -- same column count and order
        # in and out, just values transformed. Without this,
        # ColumnTransformer.get_feature_names_out() raises AttributeError
        # on any feature set with use_meta=True (see the same fix on
        # TextSelector above, which hit the same sklearn requirement).
        return np.asarray(input_features)


class EmbeddingLookup(BaseEstimator, TransformerMixin):
    """Map each row's paper id to its precomputed sentence embedding.

    Unknown ids get a zero vector, so a row that was added after the
    embeddings were computed degrades quietly instead of crashing the demo.
    """

    def __init__(self, table: dict[str, np.ndarray] | None = None,
                 dim: int = C.EMBEDDING_DIM, id_col: str = C.ID):
        self.table = table or {}
        self.dim = dim
        self.id_col = id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ids = X[self.id_col] if isinstance(X, pd.DataFrame) else pd.Series(X).astype(str)
        zero = np.zeros(self.dim, dtype=np.float32)
        return np.vstack([self.table.get(str(i), zero) for i in ids])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"emb_{i}" for i in range(self.dim)])

def build_features(
    use_title: bool = True,
    use_meta: bool = True,
    use_venue: bool = True,
    text_mode: str = "tfidf",
    embeddings: dict[str, np.ndarray] | None = None,
    char_ngrams: bool = True,
) -> Pipeline:
    """Assemble the feature transformer.

    Parameters
    ----------
    use_title : include the title text block
    use_meta  : include subfield, topic, n_references, year
    use_venue : include venue. The Day 4 ablation flips exactly this.
    text_mode : "tfidf", "embeddings", or "both". "embeddings" and "both"
        require `embeddings`.
    embeddings : {paper id -> vector}, from `load_embeddings()`
    char_ngrams : add character n-grams alongside word n-grams. Helps a little
        on short text; drop it if memory is tight.

    Returns
    -------
    An unfitted sklearn Pipeline. Call `.fit_transform(df_train)` then
    `.transform(df_test)` — never fit on test.
    """
    if text_mode not in {"tfidf", "embeddings", "both"}:
        raise ValueError(f"unknown text_mode: {text_mode!r}")
    if text_mode in {"embeddings", "both"} and not embeddings:
        raise ValueError(
            f"text_mode={text_mode!r} needs embeddings — run compute_embeddings() "
            "on Kaggle first, then load_embeddings()")
    if not (use_title or use_meta or use_venue):
        raise ValueError("no feature blocks enabled — nothing to build")

    blocks: list[tuple] = []

    if use_title:
        if text_mode in {"tfidf", "both"}:
            blocks.append((
                "title_word",
                Pipeline([("sel", TextSelector("title")),
                          ("tfidf", TfidfVectorizer(**C.TFIDF_WORD))]),
                C.TEXT_COLS,
            ))
            if char_ngrams:
                blocks.append((
                    "title_char",
                    Pipeline([("sel", TextSelector("title")),
                              ("tfidf", TfidfVectorizer(**C.TFIDF_CHAR))]),
                    C.TEXT_COLS,
                ))
        if text_mode in {"embeddings", "both"}:
            blocks.append((
                "title_emb",
                EmbeddingLookup(embeddings),
                [C.ID],
            ))

    if use_meta:
        blocks.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=5, sparse_output=True)),
            ]),
            C.CATEGORICAL_COLS,
        ))
        log_cols = tuple(i for i, c in enumerate(C.NUMERIC_COLS) if c == "n_references")
        blocks.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log1p", Log1pColumns(log_cols)),
                ("scale", StandardScaler()),
            ]),
            C.NUMERIC_COLS,
        ))

    if use_venue:
        blocks.append((
            "venue",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=5, sparse_output=True)),
            ]),
            [C.VENUE_COL],
        ))

    ct = ColumnTransformer(blocks, remainder="drop",
                           sparse_threshold=0.3, n_jobs=None)
    return Pipeline([("features", ct)])


def feature_names(pipe: Pipeline) -> np.ndarray:
    """Feature names from a FITTED pipeline. Needed for the demo's attribution
    panel and for reading model coefficients."""
    return pipe.named_steps["features"].get_feature_names_out()

def compute_embeddings(df: pd.DataFrame, batch_size: int = 256,
                       model_name: str = C.EMBEDDING_MODEL,
                       save: bool = True) -> dict[str, np.ndarray]:
    """Encode every title once. Run this on Kaggle with the GPU on.

    Saves two .npy files (vectors and their ids) so the lookup survives a
    kernel restart. Do not call this in a loop.
    """
    from sentence_transformers import SentenceTransformer

    C.ensure_dirs()
    model = SentenceTransformer(model_name)

    ids = df[C.ID].astype(str).tolist()
    titles = df["title"].fillna("").astype(str).tolist()

    vecs = model.encode(titles, batch_size=batch_size,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True).astype(np.float32)

    if save:
        np.save(C.EMBEDDINGS_NPY, vecs)
        np.save(C.EMBEDDING_IDS_NPY, np.array(ids, dtype=object),
                allow_pickle=True)
        print(f"saved {vecs.shape} to {C.EMBEDDINGS_NPY}")

    return dict(zip(ids, vecs))


def load_embeddings(vectors_path=None, ids_path=None) -> dict[str, np.ndarray]:
    """Load the cached embeddings as an id -> vector dict."""
    vectors_path = vectors_path or C.EMBEDDINGS_NPY
    ids_path = ids_path or C.EMBEDDING_IDS_NPY
    vecs = np.load(vectors_path)
    ids = np.load(ids_path, allow_pickle=True)
    if len(vecs) != len(ids):
        raise ValueError(f"embedding/id length mismatch: {len(vecs)} vs {len(ids)}")
    return dict(zip([str(i) for i in ids], vecs))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_synthetic import make_synthetic

    df = make_synthetic(1500, seed=C.SEED)
    print(C.describe())
    print()

    for name, kwargs in C.FEATURE_SETS.items():
        pipe = build_features(**kwargs)
        X = pipe.fit_transform(df)
        kind = "sparse" if hasattr(X, "toarray") else "dense"
        print(f"  {name:<20} {X.shape[0]:>5} x {X.shape[1]:<7} {kind}")

    print("\nnull handling check (50% of titles and venues nulled):")
    holed = df.copy()
    holed.loc[holed.sample(frac=0.5, random_state=1).index, "title"] = None
    holed.loc[holed.sample(frac=0.5, random_state=2).index, "venue"] = None
    X = build_features(**C.FEATURE_SETS["A1_all"]).fit_transform(holed)
    print(f"  built {X.shape} without error")