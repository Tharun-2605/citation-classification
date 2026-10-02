"""Demo app: type a paper title, get a predicted citation-impact band.

Deliberately title-only. The demo's only input is a typed title, so it can
never use venue/subfield/reference count -- the model behind it (B1,
title-only TF-IDF + LogReg) is honestly the weakest of the ablations
(~0.36 macro-F1 on the held-out test set, see results/results.csv), not the
project's best number. The write-up and slides report the strongest
ablation (everything incl. venue, ~0.45); this app reports what a bare
title alone can actually support, and says so on screen rather than
implying otherwise.

Run
---
    python src/train_demo_model.py   # once, needs the real parquet locally
    python app/demo.py

Needs models/demo_model.joblib (gitignored, regenerate with the line above
-- same reason the real parquet isn't committed either).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import joblib
import gradio as gr
import numpy as np
import pandas as pd

import config as C
from features import feature_names

MODEL_PATH = C.MODELS_DIR / "demo_model.joblib"

BAND_BLURB = {
    "Uncited": "0 citations in the first two years",
    "Low": "1-2 citations in the first two years",
    "Medium": "3-7 citations in the first two years",
    "High": "8+ citations in the first two years",
}


def _load_model():
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["spec"]


_PIPE = _SPEC = _NAMES = _CLF = _TFIDF_STEP = None  # always defined, even cold
_MODEL = _load_model()
if _MODEL is not None:
    _PIPE, _SPEC = _MODEL
    _NAMES = feature_names(_PIPE)
    _CLF = _PIPE.named_steps["clf"]
    _TFIDF_STEP = _PIPE.named_steps["features"]


def _top_words(title: str, predicted_idx: int, top_n: int = 8):
    """Which words in this title pushed the prediction toward the predicted
    band, and which pushed away from it. coef_[predicted_idx] is the
    predicted class's weight on each TF-IDF feature; multiplying by this
    title's own TF-IDF value gives each word's actual contribution here,
    not just its weight in the abstract."""
    row = pd.DataFrame({"title": [title]})
    x = _TFIDF_STEP.transform(row)
    x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()

    coef = _CLF.coef_[predicted_idx]
    contribution = x * coef
    nonzero = np.nonzero(x)[0]
    if len(nonzero) == 0:
        return "(no recognised words -- falling back to the model's prior for this band)"

    order = nonzero[np.argsort(-contribution[nonzero])]
    lines = []
    for i in order[:top_n]:
        sign = "+" if contribution[i] > 0 else "-"
        lines.append(f"{sign} `{_NAMES[i]}`  ({contribution[i]:+.3f})")
    return "\n\n".join(lines)


def predict(title: str):
    if _MODEL is None:
        raise gr.Error(
            "models/demo_model.joblib not found. Run "
            "`python src/train_demo_model.py` first (needs the real parquet "
            "in data/, see README.md for the Kaggle dataset name).")
    if not title or not title.strip():
        raise gr.Error("Type a paper title first.")

    row = pd.DataFrame({"title": [title]})
    proba = _PIPE.predict_proba(row)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = _SPEC.labels[pred_idx]

    confidences = {label: float(p) for label, p in zip(_SPEC.labels, proba)}
    attribution = _top_words(title, pred_idx)

    blurb = (f"**{pred_label}** -- {BAND_BLURB[pred_label]}\n\n"
            "Title-only model (~0.36 macro-F1 on held-out test). The "
            "project's strongest result (~0.45) needs venue and metadata "
            "this demo doesn't have.")

    return confidences, blurb, attribution


with gr.Blocks(title="Citation band predictor") as demo:
    gr.Markdown(
        "# Citation-impact band predictor\n"
        "Type a paper title. The model predicts which citation band it "
        "would likely fall into, based on two-year citation counts for CS "
        "papers (2019-2023, OpenAlex). **Title only** -- no venue, no field, "
        "no reference count. See the note below the prediction for what "
        "that costs in accuracy.")

    with gr.Row():
        title_in = gr.Textbox(
            label="Paper title", lines=2,
            placeholder="e.g. A transformer-based approach to few-shot learning")
        predict_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        label_out = gr.Label(label="Predicted band (confidence)", num_top_classes=4)
        with gr.Column():
            blurb_out = gr.Markdown(label="What this means")
            attribution_out = gr.Markdown(label="Which words drove this prediction")

    predict_btn.click(predict, inputs=title_in, outputs=[label_out, blurb_out, attribution_out])
    title_in.submit(predict, inputs=title_in, outputs=[label_out, blurb_out, attribution_out])

    gr.Examples(
        examples=[
            ["A Survey of Deep Learning Approaches to Graph Neural Networks"],
            ["Towards a preliminary note on some remarks"],
            ["Attention-based Transformer Architecture for Large-scale Foundation Models"],
        ],
        inputs=title_in,
    )


if __name__ == "__main__":
    if _MODEL is None:
        print(f"warning: {MODEL_PATH} not found -- the app will load but "
              "every prediction will show an error until you run "
              "`python src/train_demo_model.py`.")
    demo.launch()
