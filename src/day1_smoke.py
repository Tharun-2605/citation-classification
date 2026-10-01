"""End-to-end smoke test for the Day 1 layer.

Runs the whole measurement stack on synthetic data: generate -> validate ->
bin -> baselines -> evaluate -> log. If this passes, the contract layer works
and the real parquet can be dropped in unchanged on Day 2.

    python src/day1_smoke.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from make_synthetic import make_synthetic, validate
from binning import make_bands
from baselines import fit_baselines, baseline_scores
from metrics import evaluate, log_result, results_table

SEED = 42


def main() -> int:
    print("1. generate\n" + "-" * 62)
    df = make_synthetic(5000, seed=SEED)
    print(f"   {len(df):,} rows, {df.shape[1]} columns")

    print("\n2. validate\n" + "-" * 62)
    report = validate(df)
    if report["problems"]:
        print("\nFAILED: fix the problems above before going further.")
        return 1

    print("\n3. bin\n" + "-" * 62)
    y, spec = make_bands(df["cites_2yr"], n_bands=4)

    print("\n4. split\n" + "-" * 62)
    # Placeholder features only. The real feature pipeline is Day 2 work;
    # the point here is that the measurement layer runs end to end.
    X = df[["n_references"]].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"   train {len(X_tr):,}   test {len(X_te):,}   stratified, seed {SEED}")

    print("\n5. baselines\n" + "-" * 62)
    base = fit_baselines(X_tr, y_tr, seed=SEED)
    baseline_scores(base, X_te, y_te)

    print("\n6. a real (but deliberately weak) model")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(X_tr, y_tr)
    res = evaluate(y_te, clf.predict(X_te), y_train=y_tr,
                   model_name="LogReg on n_references only",
                   labels=spec.labels)

    print("\n7. log\n" + "-" * 62)
    log_result(res, notes="day-1 smoke test, synthetic data, placeholder feature")
    print()
    print(results_table().to_string())

    print("\n" + "=" * 62)
    print("Day 1 layer OK. n_references alone should barely beat the")
    print("baseline — that is the expected result, not a bug.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())