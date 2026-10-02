# citation-classification

UE24CS352A Machine Learning mini-project — Problem Statement 66, "Paper Citation Classification".

We classify a computer-science paper into a **citation-impact band** (Uncited / Low /
Medium / High) using its title and metadata, where the target is the number of
citations a paper receives in the two full calendar years after publication. This is
multi-class classification, not citation-network node classification and not
citation-intent classification — see `SCHEMA.md` for the exact target definition and
why it's built this way (age-invariance, the two-year window, why 2019–2023).

The headline result is an ablation: how much signal sits in the **title** alone versus
**venue** and other **metadata**. Short answer — title alone is weak, venue and
reference count each add real signal, and the full feature set is best but modest.
See [Results](#results) below.

## Setup

```bash
git clone git@github.com:Tharun-2605/citation-classification.git
cd citation-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get the real data

The pulled dataset (43,997 rows) is **not committed to this repository** — it lives as
a Kaggle Dataset and is regenerated locally, the same way `models/` (trained model
files) is:

- Kaggle dataset: `tharunganesh172/citations-openalex-cs-2019-2023`
- File: `citations_openalex.parquet`
- Pulled with a fixed seed (`SEED = 42` in `src/config.py`), stratified by
  year × CS subfield — see `notebooks/00_pull_and_experiments.ipynb.ipynb` for the
  exact OpenAlex API calls

Download `citations_openalex.parquet` from the Kaggle dataset page, or with the
Kaggle CLI:

```bash
kaggle datasets download -d tharunganesh172/citations-openalex-cs-2019-2023 \
    -p data --unzip
```

and place it at `data/citations_openalex.parquet` (gitignored).

`data/split.csv` (the frozen train/val/test split, **is** committed) doesn't need any
extra step — it's already in the repo.

## Running things

```bash
# Unit tests (synthetic data only, no real data needed) -- 104 tests
python -m pytest -q

# Day-1 measurement-layer smoke test, synthetic data
python src/day1_smoke.py

# Confirm the real pull + frozen split load and join correctly
python src/data.py

# Train and serialise the model the demo app loads (needs the real parquet)
python src/train_demo_model.py

# Launch the demo -- type a title, get a predicted band + confidence +
# word-level attribution
python app/demo.py
```

`notebooks/00_pull_and_experiments.ipynb.ipynb` (the OpenAlex pull, EDA, baselines,
first ablations) and `notebooks/01-train-harness.ipynb` (the shared 5-fold CV + final
test harness) are meant to be run on Kaggle, with the dataset above attached and
internet access on for the pull notebook only.

## Repository map

```
citation-classification/
├── README.md                              this file
├── SCHEMA.md                               frozen column contract + target definition
├── requirements.txt
├── pytest.ini
├── notebooks/
│   ├── 00_pull_and_experiments.ipynb.ipynb  OpenAlex pull, EDA, baselines, first ablations
│   ├── 01-train-harness.ipynb              shared 5-fold CV + final test harness
│   └── 02_models.ipynb                     (placeholder)
├── src/
│   ├── config.py                           single source of truth: seed, paths, band strategy, feature sets
│   ├── make_synthetic.py                   synthetic data on the frozen schema + validate()
│   ├── data.py                             loads the real parquet, joins the frozen split by id
│   ├── binning.py                          cites_2yr -> band labels (fixed edges [0, 2, 7, inf])
│   ├── features.py                         TF-IDF / embeddings / encoders, one build_features() per ablation
│   ├── models.py                           model constructors (LogReg, RF, LightGBM/HistGBM fallback, MLP, ...)
│   ├── pipeline.py                         build_features() + a classifier as one sklearn Pipeline
│   ├── baselines.py                        majority / stratified / uniform dummy baselines
│   ├── metrics.py                          evaluate() against baselines, log_result() to results.csv
│   ├── train_demo_model.py                 trains + serialises the title-only model the demo loads
│   └── day1_smoke.py                       end-to-end smoke test on synthetic data
├── app/
│   └── demo.py                             Gradio demo: title -> predicted band + confidence + attribution
├── tests/                                  pytest suite, synthetic data only (no real data required)
├── results/
│   ├── results.csv                         shared run log (model, feature_set, eval_split, metrics, per-class F1)
│   ├── baselines.csv, ablation_A.csv, ablation_B.csv, cv_results.csv, test_results.csv
│   └── figures/                            target_checks.png, confusion_matrices*.png
└── data/                                   gitignored -- citations_openalex.parquet, split.csv (split.csv IS committed)
```

## Results

Band edges are **fixed**, not data-fitted: `cites_2yr` in `[0, 2, 7, inf)` →
Uncited / Low / Medium / High, agreed on so every run's numbers are directly
comparable. See `src/binning.py`. Majority-class baseline: 0.265 accuracy, 0.105
macro-F1.

Final test-set scores (fit on train+val, scored once on the held-out test split,
LogisticRegression):

| Features | Accuracy | Macro-F1 |
|---|---|---|
| title only | 0.368 | 0.360 |
| n_references only | 0.418 | 0.395 |
| venue only | 0.431 | 0.403 |
| metadata only (no venue) | 0.443 | 0.422 |
| title + metadata (no venue) | 0.430 | 0.419 |
| **everything (incl. venue)** | **0.462** | **0.452** |

Full comparison table, 5-fold CV numbers, and per-class F1: `results/results.csv` and
`results/test_results.csv`. Confusion matrices: `results/figures/`.

The demo app necessarily uses the **title-only** model (0.360 macro-F1 on test) — a
typed title is the only input it ever has, so it can't use venue or metadata even
though those carry more signal. The app says this on screen.

## References

Problem statement: PS 66, sourced from the CS229 (Stanford, Fall 2019) project
archive — [report
PDF](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26647373.pdf).
No original source code exists for that project; this implementation is written from
scratch.

Dataset: [OpenAlex](https://openalex.org/) via its public
[API](https://docs.openalex.org/api-entities/works/work-object) and the
[`pyalex`](https://github.com/J535D165/pyalex) Python client.

Repositories read for feature-engineering ideas (none of these is the CS229 project;
we wrote our own code and cite whichever influenced our approach):

| Repository | Why it's relevant |
|---|---|
| [RUCAIBox/Citation-Count-Prediction](https://github.com/RUCAIBox/Citation-Count-Prediction) | EMNLP 2019 neural citation count prediction, closest in spirit to this task |
| [Lucaweihs/impact-prediction](https://github.com/Lucaweihs/impact-prediction) | Citation/h-index prediction on Semantic Scholar data, handling the paper-age confound |
| [fhopp/citation-prediction](https://github.com/fhopp/citation-prediction) | Smaller, closest to our scale |
| [zzeiidann/Paper-Citation-Prediction](https://github.com/zzeiidann/Paper-Citation-Prediction) | A different task (link prediction), but the embedding setup was a useful reference |
