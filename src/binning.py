"""Turn `cites_2yr` into class labels.

The hard part is not the binning, it is the zero mass. Citation counts are
power-law distributed with a large spike at zero, so a naive
`pd.qcut(y, 4)` puts several bin edges at 0, collapses them, and silently
returns fewer classes than asked for. That failure is quiet, which makes it
dangerous.

Three strategies here:

`fixed` (default, agreed with Person 1 on the real data)
    Fixed edges [0, 2, 7, inf] -> Uncited / Low / Medium / High, classes
    0/1/2/3. Not fitted from the data at all, which is the point: both of us
    get exactly the same bands on exactly the same rows (92.4% of rows
    disagreed with `zero_plus_quantile`'s data-fitted edges, which made the
    two codebases' numbers incomparable). Matches the `band` column already
    in data/split.csv.

`zero_plus_quantile`
    "Uncited" is its own class. The papers with at least one citation are then
    split into equal-sized quantile bands. Kept for synthetic-data dev work
    and as a documented alternative, but NOT what gets used for the real
    comparison table any more -- use `fixed` for that.

`quantile`
    Plain quantile binning over the whole target. Provided for comparison and
    because a reviewer may ask what the naive approach gives. It reports
    honestly when bins collapse instead of hiding it.

Whichever is used, ALWAYS report the resulting class distribution alongside
model scores. A band holding 60% of papers means the majority baseline is 0.60
and any accuracy near that is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

__all__ = ["make_bands", "describe_bands", "BandSpec"]


class BandSpec:
    """The fitted binning. Keep it — the demo app needs it to turn a predicted
    class index back into a human-readable band."""

    def __init__(self, strategy: str, edges: list[float], labels: list[str]):
        self.strategy = strategy
        self.edges = edges
        self.labels = labels

    def apply(self, y) -> pd.Series:
        """Label a new target vector using the already-fitted edges."""
        y = pd.Series(np.asarray(y, dtype=float))
        out = pd.Series(pd.NA, index=y.index, dtype="object")
        lo = -np.inf
        for edge, label in zip(self.edges, self.labels):
            out[(y > lo) & (y <= edge)] = label
            lo = edge
        out[y > self.edges[-1]] = self.labels[-1]
        return pd.Categorical(out, categories=self.labels, ordered=True)

    def __repr__(self):
        parts = []
        lo = 0
        for edge, label in zip(self.edges, self.labels):
            hi = "inf" if np.isinf(edge) else f"{edge:g}"
            parts.append(f"{label}=({lo:g}, {hi}]")
            lo = edge if not np.isinf(edge) else lo
        return f"BandSpec[{self.strategy}]({', '.join(parts)})"


def make_bands(
    y,
    n_bands: int = C.N_BANDS,
    strategy: str = C.BAND_STRATEGY,
    verbose: bool = True,
):
    """Bin the target into ordered class labels.

    Parameters
    ----------
    y : array-like of non-negative integers (`cites_2yr`)
    n_bands : total number of classes, including the uncited class when
        `strategy="zero_plus_quantile"`. Must be >= 2. Ignored by
        `strategy="fixed"`, which always produces its 4 fixed bands.
    strategy : "fixed", "zero_plus_quantile", or "quantile"

    Returns
    -------
    (labels, spec) : a pandas Categorical and the BandSpec used to make it
    """
    if n_bands < 2:
        raise ValueError("n_bands must be at least 2")

    y = pd.Series(np.asarray(y, dtype=float))
    if (y < 0).any():
        raise ValueError("target contains negative values")

    if strategy == "fixed":
        spec = _fixed(y)
    elif strategy == "zero_plus_quantile":
        spec = _zero_plus_quantile(y, n_bands)
    elif strategy == "quantile":
        spec = _quantile(y, n_bands)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    labels = spec.apply(y)

    if verbose:
        describe_bands(labels, spec)

    return labels, spec


def _fixed(y: pd.Series) -> BandSpec:
    """Fixed edges [0, 2, 7, inf], agreed with Person 1 so both pipelines'
    numbers are directly comparable on the real data. Matches the `band`
    column already frozen in data/split.csv: 0=Uncited, 1=Low, 2=Medium,
    3=High. `y` is accepted (for a consistent call signature with the other
    strategies) but not used to fit anything -- these edges don't move."""
    return BandSpec("fixed", [0.0, 2.0, 7.0, float("inf")],
                    ["Uncited", "Low", "Medium", "High"])


def _zero_plus_quantile(y: pd.Series, n_bands: int) -> BandSpec:
    positive = y[y > 0]
    n_positive_bands = n_bands - 1

    if len(positive) < n_positive_bands * 10:
        raise ValueError(
            f"only {len(positive)} papers with any citations — not enough for "
            f"{n_positive_bands} positive bands. Reduce n_bands.")

    qs = np.linspace(0, 1, n_positive_bands + 1)[1:-1]
    cuts = sorted(set(np.quantile(positive, qs)))

    edges = [0.0] + [float(c) for c in cuts] + [float("inf")]

    names = ["Uncited"]
    n_upper = len(edges) - 1
    if n_upper == 3:
        names += ["Low", "Medium", "High"]
    elif n_upper == 2:
        names += ["Low", "High"]
    elif n_upper == 4:
        names += ["Low", "Medium", "High", "Very high"]
    else:
        names += [f"Band {i}" for i in range(1, n_upper + 1)]

    if len(cuts) < n_positive_bands - 1:
        print(f"  note: requested {n_bands} bands, produced {len(edges)} — "
              "quantile edges collided on repeated values. This is reported, "
              "not hidden.")

    return BandSpec("zero_plus_quantile", edges, names[:len(edges)])


def _quantile(y: pd.Series, n_bands: int) -> BandSpec:
    qs = np.linspace(0, 1, n_bands + 1)[1:-1]
    cuts = sorted(set(np.quantile(y, qs)))
    edges = [float(c) for c in cuts] + [float("inf")]

    if len(edges) < n_bands:
        print(f"  WARNING: asked for {n_bands} bands, got {len(edges)}. "
              f"Quantile edges collapsed because "
              f"{float((y == 0).mean()):.0%} of the target is zero. "
              "Use strategy='zero_plus_quantile' instead.")

    names = [f"Q{i + 1}" for i in range(len(edges))]
    return BandSpec("quantile", edges, names)


def describe_bands(labels, spec: BandSpec) -> pd.DataFrame:
    """Print and return the class distribution plus the implied majority
    baseline. Always look at this before trusting any score."""
    counts = pd.Series(labels).value_counts().reindex(spec.labels).fillna(0).astype(int)
    frac = counts / counts.sum()

    table = pd.DataFrame({"count": counts, "fraction": frac.round(4)})
    print(f"\n{spec}")
    print(table.to_string())
    print(f"majority-class baseline accuracy: {frac.max():.4f}")
    print(f"random-guess accuracy           : {1 / len(spec.labels):.4f}")
    return table


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from make_synthetic import make_synthetic

    df = make_synthetic(2000)

    print("=== fixed, the default used on the real data ===")
    make_bands(df["cites_2yr"], strategy="fixed")

    print("\n=== zero_plus_quantile, 4 bands ===")
    make_bands(df["cites_2yr"], n_bands=4, strategy="zero_plus_quantile")

    print("\n=== plain quantile, 4 bands (shown so the failure is visible) ===")
    make_bands(df["cites_2yr"], n_bands=4, strategy="quantile")