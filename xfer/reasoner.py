"""Explanation-to-symbol translation and the ontology-grounded reasoner (sections 3.5 and 3.6).

Evidence per image
  sources m in {V, A, B, C}: calibrated class probabilities p_m, the predicted class c_m and its region SHAP values.
    Region support s_m = share of the positive SHAP mass that lies in the regions moved by the action units of c_m
    (Table 11 via xfer.ontology); 0.5 for a neutral prediction, which moves no action unit.
    Sanity gate: a source whose explanation has no positive mass, or puts most of it on the forehead, is
    down-weighted by GATE.
  source D: action-unit activations read from MediaPipe blendshapes. Each action unit is "active" relative to its
    75th percentile on the training split, either as a 0/1 indicator or graded (min(1, score / threshold)).
    Support S_D(c) = mean activation over the action units of c; for neutral, 1 - mean activation of all units.
Score (Equation 3.8 in spirit):
  score(c) = sum_m alpha_m g_m [ lam * p_m(c) + (1 - lam) * s_m * 1[c = c_m] ] + alpha_D * S_D(c) / sum_c S_D(c)
Decision (Figure 16): label if the top score reaches tau and leads the second by delta; a blended label if the two
top classes are a confusable pair and together reach tau; otherwise abstain. Weights, lam, tau and delta are chosen
on the validation split only.
"""
import os

import numpy as np
import pandas as pd

from .io import pred_path, read_predictions
from .labels import CLASSES, K, prob_columns
from . import ontology as ONT
from .regions import REGIONS

SOURCES = ["V", "A", "B", "C"]
GATE = 0.5


def aus():
    return list(ONT.AU_BLENDSHAPES)


# ----------------------------------------------------------------------------- loading
def au_scores(g):
    cols = {}
    for au, groups in ONT.AU_BLENDSHAPES.items():
        vals = [g[[c for c in grp if c in g.columns]].mean(axis=1) for grp in groups]
        cols[au] = pd.concat(vals, axis=1).max(axis=1)
    return pd.DataFrame(cols)


def load(out, split, prob_tags=None):
    """Aligned evidence for one split. prob_tags maps a source to the prediction file used for it (e.g. V -> VT)."""
    tags = {s: s + "T" for s in SOURCES}
    tags.update(prob_tags or {})
    m = pd.read_csv(os.path.join(out, "manifest.csv"))
    ids = m[m.split == split].image_id.astype(str).sort_values().values
    ev = dict(ids=ids)
    for s in SOURCES:
        p = read_predictions(pred_path(out, tags[s], split))
        p["image_id"] = p.image_id.astype(str)
        p = p.set_index("image_id").loc[ids]
        ev["y"] = p.label.values.astype(int)
        ev[f"P_{s}"] = p[prob_columns()].values
        e = pd.read_csv(os.path.join(out, "explanations", f"{s}_{split}.csv"))
        e["image_id"] = e.image_id.astype(str)
        e = e.set_index("image_id").reindex(ids)
        ev[f"phi_{s}"] = e[[f"shap_{r}" for r in REGIONS]].values          # NaN where not explained
        ev[f"target_{s}"] = e.target.values
    g = pd.read_csv(os.path.join(out, "results", "D_geometry.csv"))
    g["image_id"] = g.image_id.astype(str)
    g = g.set_index("image_id").reindex(ids)
    ev["found"] = g.found.fillna(False).values.astype(bool)
    ev["au"] = au_scores(g).values
    return ev


def au_thresholds(out):
    g = pd.read_csv(os.path.join(out, "results", "D_geometry.csv"))
    g = g[(g.split == "train") & g.found]
    s = au_scores(g)
    thr = s.quantile(0.75)
    measurable = thr >= 0.01
    return thr.values, measurable.values


# ----------------------------------------------------------------------------- translation
def d_support(ev, thr, measurable, mode):
    s = np.nan_to_num(ev["au"])
    t = np.where(thr > 0, thr, 1.0)
    act = (s >= t).astype(float) if mode == "hard" else np.clip(s / t, 0, 1)
    act[:, ~measurable] = np.nan
    idx = {a: i for i, a in enumerate(aus())}
    S = np.zeros((len(s), K))
    for k, c in enumerate(CLASSES):
        cols = []
        for au in ONT.AU_CONFIG[c]:
            if not measurable[idx[au]] and au in ONT.SUBSTITUTE:
                au = ONT.SUBSTITUTE[au]
            if measurable[idx[au]] and idx[au] not in cols:
                cols.append(idx[au])
        if c == "neutral":
            S[:, k] = 1 - np.nanmean(act, axis=1)
        else:
            S[:, k] = act[:, cols].mean(axis=1)
    S = np.clip(S, 1e-6, None)
    S[~ev["found"]] = 1.0                                              # no face found: no evidence
    return S / S.sum(1, keepdims=True), act


def reg_mask():
    return np.array([[r in ONT.class_regions(c) for r in REGIONS] for c in CLASSES], float)   # (K, M)


def region_support(ev, s):
    phi = ev[f"phi_{s}"]
    pred = ev[f"P_{s}"].argmax(1)
    pos = np.clip(np.nan_to_num(phi), 0, None)
    tot = pos.sum(1)
    inside = (pos * reg_mask()[pred]).sum(1)
    sup = np.where(tot > 0, inside / np.where(tot > 0, tot, 1), 0.0)
    sup[CLASSES.index("neutral") == pred] = 0.5
    explained = ~np.isnan(phi).any(1)
    sup[~explained] = 0.5
    fore = pos[:, REGIONS.index("forehead")] / np.where(tot > 0, tot, 1)
    gate = np.where(explained & ((tot <= 0) | (fore > 0.5)), GATE, 1.0)
    return sup, gate, pred


def translate(ev, thr, measurable, mode):
    tr = dict(qD=d_support(ev, thr, measurable, mode)[0])
    for s in SOURCES:
        tr[f"sup_{s}"], tr[f"gate_{s}"], tr[f"pred_{s}"] = region_support(ev, s)
    return tr


# ----------------------------------------------------------------------------- scoring and decisions
def brackets(ev, tr, lam):
    """Per-source class scores before weighting: dict name -> (n, K)."""
    out = {}
    n = len(ev["y"])
    for s in SOURCES:
        b = lam * ev[f"P_{s}"]
        b[np.arange(n), tr[f"pred_{s}"]] += (1 - lam) * tr[f"sup_{s}"]
        out[s] = b * tr[f"gate_{s}"][:, None]
    out["D"] = tr["qD"]
    return out


def score(br, alpha):
    names = SOURCES + ["D"]
    sc = sum(alpha[i] * br[n] for i, n in enumerate(names))
    return sc / sc.sum(1, keepdims=True)


def decide(sc, tau, delta, conf=None):
    """Returns kind (0 label, 1 blend, 2 abstain), c1, c2."""
    conf = conf or ONT.confusable_index()
    o = np.argsort(-sc, 1)
    c1, c2 = o[:, 0], o[:, 1]
    s1, s2 = sc[np.arange(len(sc)), c1], sc[np.arange(len(sc)), c2]
    pair = np.array([frozenset((a, b)) in conf for a, b in zip(c1, c2)])
    kind = np.full(len(sc), 2)
    kind[pair & (s1 + s2 >= tau)] = 1
    kind[(s1 >= tau) & (s1 - s2 >= delta)] = 0
    return kind, c1, c2


def selective(y, kind, c1, c2):
    acc = kind < 2
    cov = acc.mean()
    err = np.where(kind == 0, c1 != y, np.where(kind == 1, (c1 != y) & (c2 != y), False))
    risk = err[acc].mean() if acc.any() else 0.0
    return cov, risk, (kind == 0).mean(), (kind == 1).mean()


def consistency(br, c_final):
    own = np.stack([br[n].argmax(1) for n in br], 1)
    return (own == c_final[:, None]).mean(1)


def trace(i, ev, tr, br, sc, kind, c1, c2, alpha):
    names = SOURCES + ["D"]
    lines = []
    for j, s in enumerate(SOURCES):
        p = tr[f"pred_{s}"][i]
        lines.append(f"{s}: {CLASSES[p]} (p={ev[f'P_{s}'][i, p]:.2f}, region support {tr[f'sup_{s}'][i]:.2f}"
                     f"{', gated' if tr[f'gate_{s}'][i] < 1 else ''}, weight {alpha[j]:.2f})")
    d = tr["qD"][i]
    top = np.argsort(-d)[:2]
    lines.append(f"D: action-unit evidence favours {CLASSES[top[0]]} ({d[top[0]]:.2f}) then {CLASSES[top[1]]} "
                 f"({d[top[1]]:.2f}), weight {alpha[4]:.2f}" if ev["found"][i] else "D: no face landmarks, no evidence")
    verdict = {0: f"LABEL {CLASSES[c1[i]]}", 1: f"BLEND {CLASSES[c1[i]]}-{CLASSES[c2[i]]} (confusable pair)",
               2: "ABSTAIN (refer to a person)"}[kind[i]]
    lines.append(f"scores: {CLASSES[c1[i]]} {sc[i, c1[i]]:.2f}, {CLASSES[c2[i]]} {sc[i, c2[i]]:.2f} -> {verdict}")
    return " | ".join(lines)
