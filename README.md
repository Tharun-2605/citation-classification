# citation-classification

UE24CS352A Machine Learning mini-project, Problem Statement 66, "Paper Citation Classification".

**Team:** Tharun Ganesh S (PES2UG24AM172), Vansh Sharma (PES2UG24AM179)
**Write-up:** [`report/writeup.pdf`](report/writeup.pdf)

We classify a computer-science paper into a **citation-impact band** (Uncited / Low /
Medium / High) using its title and metadata. The target is the number of citations a paper
receives in the two full calendar years after its publication year. This is multi-class
classification, not citation-network node classification and not citation-intent
classification. See `SCHEMA.md` for the exact target definition and why it is built this
way (age-invariance, the two-year window, why 2019 to 2023).

The headline result is an ablation: how much signal sits in the **title** alone versus
**venue** and other **metadata**. Short answer: the title alone is weak, venue and reference
count each add real signal, and the full feature set is best but modest (about 0.45
macro-F1). See [Results](#results) below.

## Setup

Python 3.10 or newer. The project was tested with Python 3.14 on Windows and with the
Python that Kaggle provides.

```bash
git clone https://github.com/Tharun-2605/citation-classification.git
cd citation-classification
```

**Windows (Command Prompt):**

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Get the real data

The pulled dataset (43,997 rows) is **not committed to this repository**. It is stored as a
Kaggle Dataset, and you download it once into `data/`:

- Kaggle dataset: `tharunganesh172/citations-openalex-cs-2019-2023`
  (https://www.kaggle.com/datasets/tharunganesh172/citations-openalex-cs-2019-2023)
- File: `citations_openalex.parquet` (about 3 MB)
- Pulled from the OpenAlex API with a fixed seed (`SEED = 42` in `src/config.py`),
  stratified by year x CS subfield. See `notebooks/00_pull_and_experiments.ipynb.ipynb`
  for the exact API calls.

Download the file from the Kaggle dataset page (the **Download** button gives a zip; unzip
it), or use the Kaggle CLI:

```bash
kaggle datasets download -d tharunganesh172/citations-openalex-cs-2019-2023 -p data --unzip
```

Either way, the file must end up at `data/citations_openalex.parquet` (it is gitignored).

`data/split.csv` (the frozen train/val/test split) **is** committed and needs no extra step.

## Running things

```bash
# Unit tests (synthetic data only, no real data needed)
python -m pytest -q

# Day-1 measurement-layer smoke test, synthetic data
python src/day1_smoke.py

# Confirm the real pull and the frozen split load and join correctly
python src/data.py

# Train and save the model the demo app loads (needs the real parquet; takes a few minutes)
python src/train_demo_model.py

# Launch the demo, then open http://127.0.0.1:7860 in a browser.
# Type a title, get a predicted band + confidence + the words behind the prediction.
python app/demo.py
```

The commands above work on Windows as written (Python accepts forward slashes). Run
`python src/train_demo_model.py` before the first launch of the demo.

`notebooks/00_pull_and_experiments.ipynb.ipynb` (the OpenAlex pull, EDA, baselines, first
ablations) and `notebooks/01-train-harness.ipynb` (the shared 5-fold CV and final test
harness) are meant to be run on Kaggle, with the dataset above attached. Only the pull
notebook needs internet access.

## Repository map

```
citation-classification/
├── README.md                              this file
├── SCHEMA.md                              frozen column contract + target definition
├── requirements.txt
├── pytest.ini
├── report/
│   └── writeup.pdf                        two-page write-up
├── notebooks/
│   ├── 00_pull_and_experiments.ipynb.ipynb  OpenAlex pull, EDA, baselines, first ablations
│   ├── 01-train-harness.ipynb             shared 5-fold CV + final test harness
│   └── 02_models.ipynb                    RandomForest / HistGradientBoosting vs LogReg (validation split)
├── src/
│   ├── config.py                          single source of truth: seed, paths, band strategy, feature sets
│   ├── make_synthetic.py                  synthetic data on the frozen schema + validate()
│   ├── data.py                            loads the real parquet, joins the frozen split by id
│   ├── binning.py                         cites_2yr -> band labels (fixed edges [0, 2, 7, inf])
│   ├── features.py                        TF-IDF / embeddings / encoders, one build_features() per ablation
│   ├── models.py                          model constructors (LogReg, RF, LightGBM/HistGBM fallback, MLP, ...)
│   ├── pipeline.py                        build_features() + a classifier as one sklearn Pipeline
│   ├── baselines.py                       majority / stratified / uniform dummy baselines
│   ├── metrics.py                         evaluate() against baselines, log_result() to results.csv
│   ├── train_demo_model.py                trains + saves the title-only model the demo loads
│   └── day1_smoke.py                      end-to-end smoke test on synthetic data
├── app/
│   └── demo.py                            Gradio demo: title -> predicted band + confidence + attribution
├── tests/                                 pytest suite, synthetic data only (no real data required)
├── results/
│   ├── results.csv                        shared run log (model, feature_set, eval_split, metrics, per-class F1)
│   ├── results_rf_hgb.csv                 RandomForest / HistGradientBoosting runs (validation split)
│   ├── baselines.csv, ablation_A.csv, ablation_B.csv, cv_results.csv, test_results.csv
│   └── figures/                           target_checks.png, confusion_matrices*.png
└── data/                                  citations_openalex.parquet (gitignored), split.csv (committed)
```

## Results

Band edges are **fixed**, not fitted to the data: `cites_2yr` in `[0, 2, 7, inf)` gives
Uncited / Low / Medium / High (0, 1-2, 3-7, 8+ citations), so every run's numbers are
directly comparable. See `src/binning.py`. Majority-class baseline: 0.265 accuracy,
0.105 macro-F1.

All scores below are for Logistic Regression. Test scores come from fitting on train +
validation and scoring once on the held-out test split. They are the `eval_split = test`
rows of `results/results.csv`; the cross-validation column is the 5-fold CV on train +
validation (`eval_split = cv` rows).

| Features | 5-fold CV macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| majority-class baseline | n/a | 0.265 | 0.105 |
| title only | 0.372 | 0.368 | 0.361 |
| venue only | 0.385 | 0.420 | 0.390 |
| metadata only (no venue) | 0.427 | 0.443 | 0.423 |
| title + metadata (no venue) | 0.424 | 0.429 | 0.420 |
| **everything (title + metadata + venue)** | **0.448** | **0.459** | **0.450** |

What the ablation shows:

- The title is the weakest source of signal, and it adds nothing once metadata is known
  (0.423 without the title, 0.420 with it).
- Venue and reference count each carry more signal than the title, and adding venue to
  title + metadata lifts macro-F1 from 0.420 to 0.450.
- A different model does not help. On the validation split, RandomForest and
  HistGradientBoosting (`notebooks/02_models.ipynb`, `results/results_rf_hgb.csv`) did not
  beat Logistic Regression on the same features, so the ceiling comes from the features.

Per-class F1, the full run log and the confusion matrices are in `results/results.csv` and
`results/figures/`.

The demo app necessarily uses the **title-only** model (about 0.36 macro-F1 on test). A typed
title is the only input it ever has, so it cannot use venue or metadata even though those
carry more signal. The app says this on screen.

## References

Problem statement: PS 66, sourced from the CS229 (Stanford, Fall 2019) project archive,
[report PDF](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26647373.pdf).
No original source code exists for that project; this implementation is written from
scratch.

Dataset: [OpenAlex](https://openalex.org/) via its public
[API](https://docs.openalex.org/api-entities/works/work-object), queried with plain HTTP
requests.
