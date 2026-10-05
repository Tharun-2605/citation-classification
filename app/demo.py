"""Demo app: type a paper title, get a predicted citation-impact band.

Two panels, side by side.

* Title only (the original panel): the model behind it, B1 title-only
  TF-IDF + LogReg, is the weakest ablation (about 0.36 macro-F1 on the test
  set). It is shown as the honest "title alone" result.
* Title + venue: pick a venue from a dropdown of the ten most frequent
  training venues, or "Unknown venue". The model is B4_title_venue (title
  TF-IDF + venue one-hot, LogReg). Its test score is in
  results/results_title_venue.csv.

Venue and reference count are not in the title-only panel, so that panel
still shows what a title alone supports.

Run
---
    python src/train_demo_model.py         # title-only model, once
    python src/train_venue_demo_model.py   # title + venue model, once
    python app/demo.py

Both need data/citations_openalex.parquet. The models are gitignored, like
the parquet. Without the venue model the second panel shows a message and
the title-only panel still works.
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
VENUE_MODEL_PATH = C.MODELS_DIR / "demo_model_venue.joblib"
UNKNOWN_VENUE = "Unknown venue"

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


def _load_venue_model():
    if not VENUE_MODEL_PATH.exists():
        return None
    bundle = joblib.load(VENUE_MODEL_PATH)
    return bundle["pipeline"], bundle["spec"], bundle["venues"]


_PIPE = _SPEC = _NAMES = _CLF = _TFIDF_STEP = None  # always defined, even cold
_MODEL = _load_model()
if _MODEL is not None:
    _PIPE, _SPEC = _MODEL
    _NAMES = feature_names(_PIPE)
    _CLF = _PIPE.named_steps["clf"]
    _TFIDF_STEP = _PIPE.named_steps["features"]

_VPIPE = _VSPEC = _VENUES = None  # always defined, even cold
_VENUE_MODEL = _load_venue_model()
if _VENUE_MODEL is not None:
    _VPIPE, _VSPEC, _VENUES = _VENUE_MODEL


def predict_proba_by_name(title: str) -> dict:
    """Probability per band NAME. Maps through the fitted model's own
    classes_, so it stays correct whatever order the classes were stored in."""
    row = pd.DataFrame({"title": [title]})
    proba = _PIPE.predict_proba(row)[0]
    return {str(c): float(p) for c, p in zip(_CLF.classes_, proba)}


def _clean_name(raw: str) -> str:
    """'title_word__survey of' -> 'survey of'; char n-grams keep their text."""
    return raw.split("__", 1)[-1]


def _top_words(title: str, predicted_class: str, top_n: int = 8):
    """Which words in this title pushed the prediction toward the predicted
    band, and which pushed away from it. The coefficient row is looked up by
    class NAME via clf.classes_ -- never by position, because classes_ is
    alphabetical (High, Low, Medium, Uncited), not the order of the bands."""
    row = pd.DataFrame({"title": [title]})
    x = _TFIDF_STEP.transform(row)
    x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()

    class_idx = list(_CLF.classes_).index(predicted_class)
    coef = _CLF.coef_[class_idx]
    contribution = x * coef
    nonzero = np.nonzero(x)[0]
    if len(nonzero) == 0:
        return "(no recognised words -- falling back to the model's prior for this band)"

    order = nonzero[np.argsort(-contribution[nonzero])]
    lines = []
    for i in order[:top_n]:
        sign = "+" if contribution[i] > 0 else "-"
        lines.append(f"{sign} `{_clean_name(_NAMES[i])}`  ({contribution[i]:+.3f})")
    return "\n\n".join(lines)


def predict(title: str):
    if _MODEL is None:
        raise gr.Error(
            "models/demo_model.joblib not found. Run "
            "`python src/train_demo_model.py` first (needs the real parquet "
            "in data/, see README.md for the Kaggle dataset name).")
    if not title or not title.strip():
        raise gr.Error("Type a paper title first.")

    proba_by_name = predict_proba_by_name(title)
    pred_label = max(proba_by_name, key=proba_by_name.get)

    # Display in band order (Uncited -> High), keyed by name.
    confidences = {label: proba_by_name[label] for label in _SPEC.labels}
    attribution = _top_words(title, pred_label)

    blurb = (f"**{pred_label}** -- {BAND_BLURB[pred_label]}\n\n"
             "Title-only model (about 0.36 macro-F1 on held-out test). The "
             "project's strongest result (about 0.45) needs venue and metadata "
             "this demo doesn't have.")

    return confidences, blurb, attribution


def predict_venue_proba_by_name(title: str, venue: str | None) -> dict:
    """Probability per band NAME from the title + venue model. An unknown
    venue is passed as missing, which the pipeline's imputer handles the
    same way it handles missing venues in the real data."""
    venue_value = np.nan if (venue is None or venue == UNKNOWN_VENUE) else venue
    row = pd.DataFrame({"title": [title], "venue": [venue_value]})
    proba = _VPIPE.predict_proba(row)[0]
    classes = _VPIPE.named_steps["clf"].classes_
    return {str(c): float(p) for c, p in zip(classes, proba)}


def predict_venue(title: str, venue: str | None):
    """Headline, blurb and confidences for the title + venue panel."""
    if _VENUE_MODEL is None:
        raise gr.Error(
            "models/demo_model_venue.joblib not found. Run "
            "`python src/train_venue_demo_model.py` first.")
    if not title or not title.strip():
        raise gr.Error("Type a paper title first.")

    proba_by_name = predict_venue_proba_by_name(title, venue)
    pred_label = max(proba_by_name, key=proba_by_name.get)
    confidences = {label: proba_by_name[label] for label in _VSPEC.labels}
    blurb = (f"**{pred_label}** -- {BAND_BLURB[pred_label]}\n\n"
             "Title and venue model. Its test score is in "
             "`results/results_title_venue.csv`.")
    return confidences, blurb


def predict_both(title: str, venue: str | None):
    """Single handler for the UI: title-only panel plus title+venue panel.
    If one model is missing, its panel shows the message and the other
    panel still works."""
    try:
        label, blurb, attribution = predict(title)
    except gr.Error as err:
        label, blurb, attribution = None, f"**Error:** {err}", ""

    try:
        v_label, v_blurb = predict_venue(title, venue)
    except gr.Error as err:
        v_label, v_blurb = None, f"**Error:** {err}"

    return label, blurb, attribution, v_label, v_blurb


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
        venue_choices = (list(_VENUES) + [UNKNOWN_VENUE]) if _VENUES else [UNKNOWN_VENUE]
        venue_in = gr.Dropdown(
            label="Venue (used by the title + venue panel only)",
            choices=venue_choices, value=venue_choices[0])
        predict_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Title only")
            label_out = gr.Label(label="Predicted band (confidence)", num_top_classes=4)
            blurb_out = gr.Markdown()
            attribution_out = gr.Markdown(label="Which words drove this prediction")
        with gr.Column():
            gr.Markdown("### Title + venue")
            venue_label_out = gr.Label(label="Predicted band (confidence)", num_top_classes=4)
            venue_blurb_out = gr.Markdown()

    outputs = [label_out, blurb_out, attribution_out, venue_label_out, venue_blurb_out]
    predict_btn.click(predict_both, inputs=[title_in, venue_in], outputs=outputs)
    title_in.submit(predict_both, inputs=[title_in, venue_in], outputs=outputs)

    gr.Examples(
        examples=[
            ["A Survey of Deep Learning Approaches to Graph Neural Networks", None],
            ["Towards a preliminary note on some remarks", None],
            ["Attention-based Transformer Architecture for Large-scale Foundation Models", None],
        ],
        inputs=[title_in, venue_in],
    )


if __name__ == "__main__":
    if _MODEL is None:
        print(f"warning: {MODEL_PATH} not found -- the title-only panel will show an "
              "error until you run `python src/train_demo_model.py`.")
    if _VENUE_MODEL is None:
        print(f"warning: {VENUE_MODEL_PATH} not found -- the title + venue panel will "
              "show an error until you run `python src/train_venue_demo_model.py`.")
    demo.launch()
