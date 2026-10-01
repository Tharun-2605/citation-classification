"""Model constructors. No training logic — that lives in train.py.

Keeping the zoo in one file means a reviewer can see every model and every
hyperparameter in one screen, and means both of us are instantiating them
identically.

`class_weight="balanced"` is on by default everywhere it is supported. With a
target where one band holds half the papers, an unweighted model collapses to
predicting that band and reports a high accuracy that means nothing. Weighting
trades accuracy for macro F1, which is our headline metric.

    from models import get_model, MODELS
    clf = get_model("random_forest")
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

import config as C

__all__ = ["get_model", "MODELS", "SPARSE_SAFE", "is_sparse_safe", "available"]

def get_logreg(**kw):
    """Strong, fast, interpretable baseline on sparse TF-IDF. Its coefficients
    are what the demo's attribution panel reads."""
    params = dict(max_iter=2000, class_weight="balanced",
                  random_state=C.SEED, n_jobs=-1, C=1.0)
    params.update(kw)
    return LogisticRegression(**params)


def get_complement_nb(**kw):
    """ComplementNB, not MultinomialNB. Complement is the variant designed for
    imbalanced text data, which is exactly what we have. Needs non-negative
    features, so TF-IDF only — never embeddings."""
    params = dict(alpha=0.5)
    params.update(kw)
    return ComplementNB(**params)


def get_linear_svc(**kw):
    params = dict(class_weight="balanced", random_state=C.SEED,
                  max_iter=5000, dual="auto")
    params.update(kw)
    return LinearSVC(**params)


def get_random_forest(**kw):
    """Watch memory here. On dense 384-dim embeddings with 150k rows this is
    the step most likely to exhaust 8GB — run it on Kaggle."""
    params = dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
                  class_weight="balanced_subsample", n_jobs=-1,
                  random_state=C.SEED)
    params.update(kw)
    return RandomForestClassifier(**params)


def get_lightgbm(**kw):
    """LightGBM if installed, otherwise sklearn's HistGradientBoosting.

    Both are gradient boosting on histogram-binned features and they score
    within noise of each other here, so the fallback costs nothing. Say in the
    report which one actually ran.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return get_hist_gbm(**kw)

    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                  class_weight="balanced", random_state=C.SEED,
                  n_jobs=-1, verbose=-1)
    params.update(kw)
    return LGBMClassifier(**params)


def get_hist_gbm(**kw):
    """Always available. Requires dense input — see SPARSE_SAFE."""
    params = dict(max_iter=400, learning_rate=0.05,
                  class_weight="balanced", random_state=C.SEED)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def get_mlp(**kw):
    """Small MLP. Do not reach for PyTorch here — it buys nothing on this
    problem and costs an afternoon. No class_weight support in sklearn's MLP,
    so read its macro F1 with that in mind."""
    params = dict(hidden_layer_sizes=(128, 64), max_iter=60,
                  early_stopping=True, n_iter_no_change=5,
                  random_state=C.SEED)
    params.update(kw)
    return MLPClassifier(**params)


MODELS = {
    "logreg": get_logreg,
    "complement_nb": get_complement_nb,
    "linear_svc": get_linear_svc,
    "random_forest": get_random_forest,
    "lightgbm": get_lightgbm,
    "hist_gbm": get_hist_gbm,
    "mlp": get_mlp,
}

_SPARSE_SAFE_STATIC = {
    "logreg": True,
    "complement_nb": True,
    "linear_svc": True,
    "random_forest": True,
    "lightgbm": True,       
    "hist_gbm": False,      
    "mlp": True,
}


def is_sparse_safe(name: str) -> bool:
    """Resolved at runtime, not from a static table.

    `get_lightgbm` silently falls back to HistGradientBoosting when LightGBM
    is missing, and that fallback is dense-only. A static lookup would tell the
    harness sparse input is fine and the model would then blow up inside
    `fit`. Ask this function instead.
    """
    if name == "lightgbm" and not available()["lightgbm"]:
        return False        # we will actually get HistGradientBoosting
    return _SPARSE_SAFE_STATIC[name]


class _SparseSafeView(dict):
    """Keeps `SPARSE_SAFE[name]` working while routing through the runtime
    check, so existing call sites do not need to change."""

    def __getitem__(self, name):
        return is_sparse_safe(name)

    def get(self, name, default=None):
        try:
            return is_sparse_safe(name)
        except KeyError:
            return default


SPARSE_SAFE = _SparseSafeView(_SPARSE_SAFE_STATIC)

NON_NEGATIVE_ONLY = {"complement_nb"}


def get_model(name: str, **kw):
    """Look up a model by name."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}. Available: {sorted(MODELS)}")
    return MODELS[name](**kw)


def available() -> dict[str, bool]:
    """Which optional dependencies are actually installed here."""
    out = {}
    try:
        import lightgbm  # noqa: F401
        out["lightgbm"] = True
    except ImportError:
        out["lightgbm"] = False
    try:
        import sentence_transformers  # noqa: F401
        out["sentence_transformers"] = True
    except ImportError:
        out["sentence_transformers"] = False
    return out


if __name__ == "__main__":
    print("optional dependencies:", available())
    print()
    for name in MODELS:
        clf = get_model(name)
        flags = []
        if not is_sparse_safe(name):
            flags.append("dense only")
        if name in NON_NEGATIVE_ONLY:
            flags.append("non-negative features only")
        note = f"   [{', '.join(flags)}]" if flags else ""
        print(f"  {name:<16} {type(clf).__name__}{note}")