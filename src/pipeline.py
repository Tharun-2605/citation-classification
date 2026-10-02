"""The single sklearn Pipeline: build_features() wired into a classifier.

This is the one module that is allowed to know about both features.py and
models.py. Keeping that coupling in exactly one place is what makes the
ablations (Day 4) honest: every comparison in the report is the same
`build_pipeline` call with different `feature_set` / `model_name` arguments,
never a hand-rolled one-off pipeline that quietly differs from the others.

    from pipeline import build_pipeline

    pipe = build_pipeline(feature_set="A2_no_venue", model_name="logreg")
    pipe.fit(df_train, y_train)
    pipe.predict(df_test)

Train on df_train, never fit on df_test -- same rule as build_features().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import config as C
from features import build_features
from models import get_model, is_sparse_safe, NON_NEGATIVE_ONLY

__all__ = ["build_pipeline", "DenseTransformer"]


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Densifies a sparse feature matrix.

    Needed in front of models that cannot take sparse input -- today that is
    HistGradientBoosting, which is also what `get_model("lightgbm")` silently
    falls back to when LightGBM isn't installed. Ask `models.is_sparse_safe`,
    never a static table, since that fallback is exactly what makes a static
    answer wrong (see models.py).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else np.asarray(X)


def build_pipeline(
    feature_set: str = "A1_all",
    model_name: str = "logreg",
    feature_overrides: dict | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
    model_kwargs: dict | None = None,
) -> Pipeline:
    """Assemble one fit/predict-ready Pipeline for an (feature_set, model)
    pair.

    Parameters
    ----------
    feature_set : a key from config.FEATURE_SETS -- this is what the Day 4
        ablations vary.
    model_name : a key from models.MODELS.
    feature_overrides : override individual build_features() flags on top of
        the named feature_set, e.g. {"char_ngrams": False}.
    embeddings : passed through to build_features() for text_mode in
        {"embeddings", "both"}.
    model_kwargs : passed through to get_model(), e.g. {"C": 0.5}.

    Returns
    -------
    An unfitted sklearn Pipeline: ColumnTransformer -> [dense conversion if
    the model needs it] -> classifier.
    """
    if feature_set not in C.FEATURE_SETS:
        raise KeyError(f"unknown feature set {feature_set!r}. "
                       f"Available: {sorted(C.FEATURE_SETS)}")

    fkwargs = dict(C.FEATURE_SETS[feature_set])
    fkwargs.update(feature_overrides or {})

    if model_name in NON_NEGATIVE_ONLY and fkwargs.get("use_meta", False):
        raise ValueError(
            f"{model_name!r} requires non-negative features, but "
            f"{feature_set!r} (possibly after feature_overrides) has "
            "use_meta=True -- its numeric block is StandardScaler'd and can "
            "go negative. Use a feature set with use_meta=False, or pick a "
            "different model.")

    features_pipe = build_features(embeddings=embeddings, **fkwargs)
    clf = get_model(model_name, **(model_kwargs or {}))

    steps = list(features_pipe.steps)  # [("features", ColumnTransformer)]
    if not is_sparse_safe(model_name):
        steps.append(("to_dense", DenseTransformer()))
    steps.append(("clf", clf))

    return Pipeline(steps)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from binning import make_bands, BandSpec
    from metrics import evaluate
    from data import load_dataset

    try:
        # The real thing: the frozen split, joined by id. No ad-hoc split.
        splits = load_dataset()
        spec = BandSpec("fixed", [0.0, 2.0, 7.0, float("inf")],
                        ["Uncited", "Low", "Medium", "High"])
        train, test = splits["train"], splits["test"]
        ytr = pd.Categorical(train["band"].map(dict(enumerate(spec.labels))),
                             categories=spec.labels, ordered=True)
        yte = pd.Categorical(test["band"].map(dict(enumerate(spec.labels))),
                             categories=spec.labels, ordered=True)
    except FileNotFoundError as e:
        # Dev fallback only -- the parquet isn't downloaded locally. This is
        # NOT the frozen split (synthetic rows have no entry in split.csv to
        # join against), it's here purely so this file stays runnable for
        # pipeline-mechanics smoke testing without the real data.
        print(f"(no real data locally -- {e})")
        print("falling back to synthetic data for a mechanics smoke test only\n")
        from sklearn.model_selection import train_test_split
        from make_synthetic import make_synthetic

        df = make_synthetic(3000, seed=C.SEED)
        y, spec = make_bands(df["cites_2yr"], strategy="fixed", verbose=False)
        train, test, ytr, yte = train_test_split(
            df, y, test_size=0.2, random_state=C.SEED, stratify=y)

    for feature_set, model_name in [
        ("A1_all", "logreg"),
        ("A2_no_venue", "random_forest"),
        ("A3_venue_only", "hist_gbm"),       # exercises the dense conversion
        ("B1_title_only", "complement_nb"),  # exercises the non-negative check
    ]:
        pipe = build_pipeline(feature_set=feature_set, model_name=model_name)
        pipe.fit(train, ytr)
        pred = pipe.predict(test)
        evaluate(yte, pred, y_train=ytr,
                 model_name=f"{model_name} / {feature_set}", labels=spec.labels)
