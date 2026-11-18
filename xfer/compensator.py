"""Post-hoc compensator for the base model V (section 3.7): a detector that predicts when V is wrong (Equation 3.10),
a corrector trained on V's failures only (Equation 3.11), and the gate that runs the corrector only when the detector
fires (Equation 3.12). This module holds the feature blocks and the CPU (tabular) arm; the image arms are in
scripts/13_compensator_image.py.

Feature blocks, all about V and the image, never the other sources' predictions:
  pred  V's calibrated probabilities (log), confidence, margin, entropy and one-hot prediction
  expl  V's region SHAP and LIME values, their rank agreement, and the region support of V's prediction
  face  measurements of the face itself from source D: 8 geometric ratios and 14 action-unit activations
"""
import os

import numpy as np
import pandas as pd

from .io import pred_path, read_predictions
from .labels import K, prob_columns
from .reasoner import au_scores, region_support
from .regions import GEOMETRY, REGIONS

BLOCKS = {"pred": ["pred"], "pred+expl": ["pred", "expl"], "pred+face": ["pred", "face"],
          "pred+expl+face": ["pred", "expl", "face"]}


def load_split(out, split, ids=None):
    m = pd.read_csv(os.path.join(out, "manifest.csv"))
    sub = m[m.split == split]
    if ids is None:
        ids = sub.image_id.astype(str).sort_values().values
    p = read_predictions(pred_path(out, "VT", split))
    p["image_id"] = p.image_id.astype(str)
    p = p.set_index("image_id").loc[ids]
    e = pd.read_csv(os.path.join(out, "explanations", f"V_{split}.csv"))
    e["image_id"] = e.image_id.astype(str)
    e = e.set_index("image_id").reindex(ids)
    g = pd.read_csv(os.path.join(out, "results", "D_geometry.csv"))
    g["image_id"] = g.image_id.astype(str)
    g = g.set_index("image_id").reindex(ids)
    paths = sub.assign(image_id=sub.image_id.astype(str)).set_index("image_id").loc[ids].path.values
    P = p[prob_columns()].values
    ev = {"P_V": P, "phi_V": e[[f"shap_{r}" for r in REGIONS]].values}
    sup, gate, pred = region_support(ev, "V")
    return dict(ids=np.asarray(ids), y=p.label.values.astype(int), P=P, pred=P.argmax(1), paths=paths,
                shap=e[[f"shap_{r}" for r in REGIONS]].values, lime=e[[f"lime_{r}" for r in REGIONS]].values,
                spearman=e.spearman.values, support=sup, geo=g[GEOMETRY].values, au=au_scores(g).values,
                found=g.found.fillna(False).values.astype(float))


def features(d, block):
    P = np.clip(d["P"], 1e-6, 1)
    s = np.sort(P, 1)
    parts = []
    if "pred" in BLOCKS[block]:
        ent = -(P * np.log(P)).sum(1, keepdims=True)
        parts += [np.log(P), s[:, -1:], s[:, -1:] - s[:, -2:-1], ent, np.eye(K)[d["pred"]]]
    if "expl" in BLOCKS[block]:
        parts += [np.nan_to_num(d["shap"]), np.nan_to_num(d["lime"]), np.nan_to_num(d["spearman"])[:, None],
                  d["support"][:, None]]
    if "face" in BLOCKS[block]:
        parts += [np.nan_to_num(d["geo"]), np.nan_to_num(d["au"]), d["found"][:, None]]
    return np.hstack(parts).astype(float)


def gate_threshold(det_score, base_pred, corr_pred, y):
    """Choose the detector threshold on validation that maximises the net number of corrections."""
    best = (0, 1.01)                                          # (net, threshold); threshold > 1 means never fire
    for t in np.unique(np.round(det_score, 4)):
        fire = det_score >= t
        final = np.where(fire, corr_pred, base_pred)
        net = int(((final == y) & (base_pred != y)).sum() - ((final != y) & (base_pred == y)).sum())
        if net > best[0]:
            best = (net, float(t))
    return best[1]


def correction_counts(base_pred, final, y):
    fixed = int(((final == y) & (base_pred != y)).sum())
    broken = int(((final != y) & (base_pred == y)).sum())
    return fixed, broken, (fixed - broken) / len(y)
