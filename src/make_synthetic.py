"""Synthetic data on the frozen schema, so the pipeline can be built before the
real OpenAlex pull lands.

The point is not realism of content. It is realism of *shape*: a power-law
target with a large zero mass, nulls in `title` and `venue`, venue carrying
genuine signal (so the ablation has something to find), and a flat median
target across years (so the year-flatness check passes on good data and we can
tell when it is catching something real).

Usage
-----
    from make_synthetic import make_synthetic, validate
    df = make_synthetic(500)
    validate(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

YEARS = C.YEARS

SUBFIELDS = [
    "Artificial Intelligence",
    "Computer Vision and Pattern Recognition",
    "Computer Networks and Communications",
    "Software",
    "Information Systems",
    "Human-Computer Interaction",
    "Computational Theory and Mathematics",
    "Signal Processing",
]

VENUES = {
    "Nature Machine Intelligence": 6.0,
    "NeurIPS": 4.5,
    "CVPR": 4.0,
    "ICML": 4.0,
    "ACM Computing Surveys": 3.5,
    "IEEE Transactions on Pattern Analysis": 3.0,
    "ICLR": 3.0,
    "AAAI": 2.0,
    "IEEE Access": 1.0,
    "Journal of Computer Science": 0.6,
    "International Journal of Computer Applications": 0.4,
    "Proceedings of a Regional Workshop": 0.3,
}

HOT_WORDS = ["deep", "transformer", "attention", "graph", "self-supervised",
             "foundation", "diffusion", "benchmark", "large-scale", "neural"]
COLD_WORDS = ["a", "note", "preliminary", "case", "survey", "brief",
              "towards", "some", "remarks", "an"]
NEUTRAL_WORDS = ["learning", "model", "analysis", "framework", "approach",
                 "system", "method", "evaluation", "network", "data",
                 "optimization", "detection", "classification", "prediction",
                 "representation", "architecture", "inference", "training"]


def _make_title(rng: np.random.Generator, heat: float) -> str:
    """Build a title whose word mix reflects `heat` in [0, 1]."""
    n = int(rng.integers(4, 12))
    words = []
    for _ in range(n):
        r = rng.random()
        if r < 0.25 * heat:
            words.append(rng.choice(HOT_WORDS))
        elif r < 0.25 * heat + 0.20 * (1 - heat):
            words.append(rng.choice(COLD_WORDS))
        else:
            words.append(rng.choice(NEUTRAL_WORDS))
    return " ".join(words).capitalize()


def make_synthetic(
    n: int = 500,
    seed: int = C.SEED,
    null_title_rate: float = 0.01,
    null_venue_rate: float = 0.08,
) -> pd.DataFrame:
    """Return `n` rows on the frozen schema.

    Parameters
    ----------
    n : number of rows
    seed : random seed, so the frame is reproducible
    null_title_rate, null_venue_rate : fraction of rows with those fields null,
        matching what OpenAlex actually returns
    """
    rng = np.random.default_rng(seed)

    venue_names = list(VENUES)
    # Realistic: a few venues hold most papers.
    venue_p = np.array([0.02, 0.03, 0.04, 0.03, 0.02, 0.05,
                        0.03, 0.08, 0.25, 0.18, 0.15, 0.12])
    venue_p = venue_p / venue_p.sum()

    venue = rng.choice(venue_names, size=n, p=venue_p)
    subfield = rng.choice(SUBFIELDS, size=n)
    year = rng.choice(YEARS, size=n)

    venue_mult = np.array([VENUES[v] for v in venue])
    heat = np.clip((venue_mult - 0.3) / 5.7, 0, 1)

    titles = [_make_title(rng, float(h)) for h in heat]

    intensity = rng.lognormal(mean=np.log(venue_mult) - 0.5, sigma=1.1)
    cites_2yr = rng.poisson(intensity)

    years_since = 2026 - year
    tail = rng.poisson(intensity * 0.35 * np.maximum(years_since - 2, 0))
    cites_total = cites_2yr + tail

    n_references = rng.poisson(28) + rng.integers(0, 12, size=n)

    df = pd.DataFrame({
        "id": [f"https://openalex.org/W{9000000 + i}" for i in range(n)],
        "title": titles,
        "year": year.astype(int),
        "subfield": subfield,
        "topic": [f"{s} topic {rng.integers(1, 6)}" for s in subfield],
        "venue": venue,
        "n_references": n_references.astype(int),
        "cites_total": cites_total.astype(int),
        "cites_2yr": cites_2yr.astype(int),
    })

    # Inject nulls the way OpenAlex does.
    if null_title_rate:
        idx = rng.choice(n, size=int(n * null_title_rate), replace=False)
        df.loc[idx, "title"] = None
    if null_venue_rate:
        idx = rng.choice(n, size=int(n * null_venue_rate), replace=False)
        df.loc[idx, "venue"] = None

    return df


def _year_flatness(values, years, tol: float = 0.10):
    """Spearman rank correlation between year and the target.

    Robust to the zero mass and the heavy tail, unlike a comparison of medians.
    Returns (passes, rho). |rho| under `tol` counts as flat.
    """
    v = pd.Series(np.asarray(values), dtype=float)
    y = pd.Series(np.asarray(years), dtype=float)
    if v.nunique() < 2 or y.nunique() < 2:
        return True, 0.0
    rho = float(v.rank().corr(y.rank()))
    if np.isnan(rho):
        return True, 0.0
    return abs(rho) <= tol, rho


REQUIRED = {
    "id": "object", "title": "object", "year": "int64", "subfield": "object",
    "topic": "object", "venue": "object", "n_references": "int64",
    "cites_total": "int64", "cites_2yr": "int64",
}


def validate(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Check the schema invariants. Returns a dict of results; raises on a
    hard structural failure. Run this on the REAL data too, not just synthetic."""
    problems, warnings = [], []

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    extra = set(df.columns) - set(REQUIRED)
    if extra:
        warnings.append(f"unexpected columns (harmless): {sorted(extra)}")

    if df["id"].duplicated().any():
        problems.append(f"{int(df['id'].duplicated().sum())} duplicate ids")

    bad_years = set(df["year"].unique()) - set(YEARS)
    if bad_years:
        problems.append(f"years outside 2019-2023: {sorted(bad_years)}")

    if (df["cites_2yr"] < 0).any() or (df["cites_total"] < 0).any():
        problems.append("negative citation counts")

    n_impossible = int((df["cites_2yr"] > df["cites_total"]).sum())
    if n_impossible:
        problems.append(
            f"{n_impossible} rows where cites_2yr > cites_total (impossible)")

    zero_frac = float((df["cites_2yr"] == 0).mean())
    skew = float(df["cites_2yr"].skew())
    if skew < 1.0:
        warnings.append(
            f"target skew is {skew:.2f} — expected > 1. Distribution looks "
            "wrong; check the pull.")

    flat, rho = _year_flatness(df["cites_2yr"], df["year"])
    med = df.groupby("year")["cites_2yr"].median()
    mean = df.groupby("year")["cites_2yr"].mean()

    if not flat:
        problems.append(
            f"cites_2yr trends with year (Spearman rho = {rho:+.3f}, "
            "expected near 0). The target logic is broken — fix before "
            "modelling.")
    elif abs(rho) > 0.05:
        warnings.append(
            f"mild year trend in cites_2yr (rho = {rho:+.3f}). Tolerable, but "
            "check the counts_by_year window logic.")

    _, rho_total = _year_flatness(df["cites_total"], df["year"])
    if rho_total >= -0.02:
        warnings.append(
            f"cites_total does not fall with year (rho = {rho_total:+.3f}). "
            "Expected clearly negative — older papers have had longer to "
            "accumulate. If this is flat too, the pull may be wrong.")

    result = {
        "rows": len(df),
        "zero_fraction": zero_frac,
        "skew": skew,
        "median_by_year": med.to_dict(),
        "mean_by_year": mean.round(2).to_dict(),
        "year_rho": round(float(rho), 4),
        "year_rho_cites_total": round(float(rho_total), 4),
        "year_flat": flat,
        "null_title": int(df["title"].isna().sum()),
        "null_venue": int(df["venue"].isna().sum()),
        "problems": problems,
        "warnings": warnings,
    }

    if verbose:
        print(f"rows              : {result['rows']:,}")
        print(f"zero fraction     : {zero_frac:.1%}")
        print(f"target skew       : {skew:.2f}")
        print(f"nulls title/venue : {result['null_title']} / {result['null_venue']}")
        print("cites_2yr by year (median / mean):")
        for y in sorted(med.index):
            print(f"    {y}:  {med[y]:>5.1f}  /  {mean[y]:>6.2f}")
        print(f"year trend rho    : {rho:+.4f}  (cites_2yr, want ~0)")
        print(f"                    {rho_total:+.4f}  (cites_total, want negative)")
        print(f"year-flatness     : {'PASS' if flat else 'FAIL'}")
        for w in warnings:
            print(f"  warning: {w}")
        for p in problems:
            print(f"  PROBLEM: {p}")
        if not problems:
            print("all invariants hold")

    return result


if __name__ == "__main__":
    frame = make_synthetic(500)
    print(frame.head(3).to_string())
    print()
    validate(frame)