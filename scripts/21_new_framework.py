"""Steps 3, 5 and 6 of the new design on the CPU: claim checking, XAI fusion and symbolic decision, plus the
interim structured-evidence compensator for the new VLM (the SLM arm is trained separately on a GPU).

Inputs in OUT: manifest.csv, predictions/{V2,A,B,C}_{train,val,test}.csv, descriptions/V2_*_{describe,given}.csv,
explanations/occ_{V2,A,B,C}_{val,test}.npz, results/D_geometry.csv.
Everything is fitted on the validation split (or the 1,501-image training subset for the compensator); the test set
is used once. Writes results/new_framework.md, results/new_framework.json and predictions/NF_*_test.csv.
Usage:  python scripts/21_new_framework.py --out OUT
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from xfer import ontology as ONT
from xfer.claims import FIELDS, FIELD_REGION, claim_table
from xfer.io import pred_path, read_predictions
from xfer.labels import CLASSES, K, prob_columns
from xfer.metrics import accuracy, bootstrap_ci, macro_f1, mcnemar
from xfer.reasoner import au_scores, au_thresholds, aus, decide, selective
from xfer.regions import COARSE_OF, FINE, REGIONS

EXPERTS = ["V2", "A", "B", "C"]
NAMES = {"V2": "InternVL3-8B (VLM)", "V2c": "VLM + compensator", "A": "CLIP", "B": "POSTER++", "C": "HOG/LBP + SVM"}
RNG = np.random.default_rng(42)


# ============================================================================ loading
def load_split(out, split, thr, measurable):
    m = pd.read_csv(os.path.join(out, "manifest.csv"))
    ids = m[m.split == split].image_id.astype(str).sort_values().values
    ev = dict(ids=ids, n=len(ids))
    for s in EXPERTS:
        p = read_predictions(pred_path(out, s, split))
        p["image_id"] = p.image_id.astype(str)
        p = p.set_index("image_id").reindex(ids)
        ev["y"] = p.label.values.astype(int)
        ev[f"P_{s}"] = np.clip(p[prob_columns()].values.astype(float), 1e-6, 1)
    g = pd.read_csv(os.path.join(out, "results", "D_geometry.csv"))
    g["image_id"] = g.image_id.astype(str)
    g = g.set_index("image_id").reindex(ids)
    ev["found"] = g.found.fillna(False).values.astype(bool)
    au = np.nan_to_num(au_scores(g).values)
    ev["au_graded"] = np.clip(au / np.where(thr > 0, thr, 1), 0, 1)
    ev["au_active"] = au >= np.where(thr > 0, thr, np.inf)
    ev["geom"] = np.nan_to_num(g[["inner_brow_raise", "outer_brow_raise", "brow_gap", "eye_opening", "mouth_opening",
                                  "mouth_width", "lip_corner_lift", "nose_to_lip"]].values.astype(float))
    # occlusion explanations (fine regions); the VLM has them only for its stratified sample
    for s in EXPERTS:
        f = os.path.join(out, "explanations", f"occ_{s}_{split}.npz")
        ev[f"occ_{s}"] = np.zeros(ev["n"], bool)
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        pos = {i: j for j, i in enumerate(z["image_id"].astype(str))}
        idx = np.array([pos.get(i, -1) for i in ids])
        have = idx >= 0
        ev[f"occ_{s}"] = have
        for key in ["P_full", "P_none", "P_drop", "del_P", "ins_P"]:
            arr = z[key]
            full = np.full((ev["n"],) + arr.shape[1:], np.nan)
            full[have] = arr[idx[have]]
            ev[f"{key}_{s}"] = full
    # the VLM's structured description (emotion not given)
    f = os.path.join(out, "descriptions", f"V2_{split}_describe.csv")
    ev["has_desc"] = np.zeros(ev["n"], bool)
    if os.path.exists(f):
        d = pd.read_csv(f).drop_duplicates("image_id", keep="last")
        d["image_id"] = d.image_id.astype(str)
        d = d.set_index("image_id").reindex(ids)
        ev["has_desc"] = d.text.notna().values
        ev["desc"] = d.reset_index()
        cons, ncl, posf, flags = claim_table(ev["desc"], ev["au_active"], aus(), measurable)
        ev["claim_cons"], ev["n_claims"], ev["claim_fields"], ev["claim_flags"] = cons, ncl, posf, flags
        ev["desc_emotion"] = d.emotion_idx.fillna(-1).values.astype(int)
    return ev


# ============================================================================ calibration and reliability
def fit_temperature(P, y):
    L = np.log(P)
    def nll(t):
        z = L / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def apply_t(P, t):
    z = np.log(P) / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def class_reliability(P, y):
    """Precision of each predicted class on validation, with add-one smoothing."""
    pred = P.argmax(1)
    return np.array([((pred == c) & (y == c)).sum() + 1 for c in range(K)]) / \
        np.array([(pred == c).sum() + 2 for c in range(K)])


# ============================================================================ step 3/4: interim compensator (CPU)
def comp_features(ev):
    desc_onehot = np.zeros((ev["n"], K))
    ok = ev.get("desc_emotion", np.full(ev["n"], -1)) >= 0
    desc_onehot[np.where(ok)[0], ev["desc_emotion"][ok]] = 1
    flags = np.nan_to_num(ev.get("claim_flags", np.full((ev["n"], 5), np.nan)), nan=-1)
    cons = np.nan_to_num(ev.get("claim_cons", np.full(ev["n"], np.nan)), nan=-1)[:, None]
    return np.hstack([np.log(ev["P_V2"]), ev["au_graded"], ev["geom"], flags, cons, desc_onehot])


def train_compensator(tr, va):
    from sklearn.ensemble import HistGradientBoostingClassifier
    keep = tr["has_desc"]
    X, y, pred = comp_features(tr)[keep], tr["y"][keep], tr["P_V2"][keep].argmax(1)
    wrong = (pred != y).astype(int)
    det = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         class_weight="balanced", random_state=42).fit(X, wrong)
    cor = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         class_weight="balanced", random_state=42).fit(X[wrong == 1], y[wrong == 1])
    def apply(ev, tau_d):
        Xe = comp_features(ev)
        d = det.predict_proba(Xe)[:, 1]
        pc = np.zeros((ev["n"], K))
        pc[:, cor.classes_] = cor.predict_proba(Xe)
        P = ev["P_V2"].copy()
        fire = d >= tau_d
        P[fire] = np.clip(pc[fire], 1e-6, 1)
        return P / P.sum(1, keepdims=True), fire, d
    best = max(((accuracy(va["y"], apply(va, t)[0].argmax(1)), t) for t in np.linspace(0.3, 0.9, 25)))
    return apply, best[1], len(X), int(wrong.sum())


# ============================================================================ step 5: XAI fusion
CLASS_REG = np.array([[r in ONT.class_regions(c) for r in REGIONS] for c in CLASSES], float)       # (K, 7)
FINE2COARSE = np.array([[COARSE_OF[f] == r for r in REGIONS] for f in FINE], float)              # (10, 7)


def translate(ev, s):
    """Occlusion -> per-class region support (n, K), forehead gate (n,), 1-step faithfulness (n,) for expert s.
    Images without occlusion get support 0.5, gate 1 and faithfulness NaN."""
    n = ev["n"]
    sup, gate, faith = np.full((n, K), 0.5), np.ones(n), np.full(n, np.nan)
    have = ev[f"occ_{s}"]
    if not have.any():
        return sup, gate, faith
    Pf, Pd, Pn = ev[f"P_full_{s}"][have], ev[f"P_drop_{s}"][have], ev[f"P_none_{s}"][have]
    drop = np.clip(Pf[:, None, :] - Pd, 0, None)                                  # (h, 10, K)
    coarse = np.einsum("hrk,rc->hck", drop, FINE2COARSE)                           # (h, 7, K)
    tot = coarse.sum(1)
    inside = np.einsum("hck,kc->hk", coarse, CLASS_REG)
    sp = np.where(tot > 1e-6, inside / np.where(tot > 1e-6, tot, 1), 0.0)
    sp[:, CLASSES.index("neutral")] = 0.5
    sup[have] = sp
    t = Pf.argmax(1)
    h = np.arange(len(t))
    fore = coarse[h, REGIONS.index("forehead"), t] / np.where(tot[h, t] > 1e-6, tot[h, t], 1)
    gate[have] = np.where((tot[h, t] <= 1e-6) | (fore > 0.5), 0.5, 1.0)
    d1 = ev[f"del_P_{s}"][have][:, 0, :][h, t]
    i1 = ev[f"ins_P_{s}"][have][:, 0, :][h, t]
    del_n = np.clip((Pf[h, t] - d1) / np.maximum(Pf[h, t], 1e-6), 0, 1)
    span = Pf[h, t] - Pn[h, t]
    ins_n = np.where(span > 0.02, np.clip((i1 - Pn[h, t]) / np.where(span > 0.02, span, 1), 0, 1), 0.5)
    faith[have] = 0.5 * (del_n + ins_n)
    return sup, gate, faith


def d_support(ev):
    idx = {a: i for i, a in enumerate(aus())}
    act = ev["au_graded"]
    S = np.zeros((ev["n"], K))
    for k, c in enumerate(CLASSES):
        cols = sorted({idx[ONT.SUBSTITUTE.get(a, a)] for a in ONT.AU_CONFIG[c]})
        S[:, k] = act[:, cols].mean(1) if cols else 1 - act.mean(1)
    S = np.clip(S, 1e-6, None)
    S[~ev["found"]] = 1.0
    return S / S.sum(1, keepdims=True)


def prepare(ev, cal, vlm="V2"):
    """Everything the fusion needs, for one split. vlm = 'V2' or 'V2c' (compensated probabilities in ev)."""
    f = dict(y=ev["y"], n=ev["n"], qD=d_support(ev))
    for s in EXPERTS:
        key = vlm if s == "V2" else s
        f[f"P_{s}"] = apply_t(ev[f"P_{key}"], cal["T"][key])
        f[f"rel_{s}"] = cal["rel"][key]
        f[f"sup_{s}"], f[f"gate_{s}"], fa = translate(ev, s)
        if s == "V2":   # the VLM's faithfulness: occlusion where available, its verified claims everywhere
            cc = ev.get("claim_cons", np.full(ev["n"], np.nan))
            fa = np.where(np.isnan(fa), cc, np.where(np.isnan(cc), fa, 0.5 * (fa + cc)))
            f["claim_cons"] = cc
        f[f"faith_{s}"] = np.nan_to_num(fa, nan=0.5)
    return f


def evidence(f, s, lam, rho, gam):
    """Per-class evidence of expert s: calibrated confidence x per-class reliability x region support x sanity gate
    x faithfulness (Stage 2, steps 1 to 3)."""
    E = f[f"P_{s}"] * (f[f"rel_{s}"][None, :] ** rho)
    E = E * (lam + (1 - lam) * f[f"sup_{s}"])
    E = E * f[f"gate_{s}"][:, None] * ((0.5 + 0.5 * f[f"faith_{s}"]) ** gam)[:, None]
    return E / np.clip(E.sum(1, keepdims=True), 1e-9, None)


def fuse(f, prm, symbolic=None):
    """Stage 2, step 4 (aggregate), optionally with the Stage 3 precedence rule. Returns fused scores (n, K)."""
    a = prm["alpha"]
    F = np.zeros((f["n"], K))
    w_tot = np.zeros(f["n"])
    for j, s in enumerate(EXPERTS):
        w = np.full(f["n"], a[j])
        if symbolic and s == "V2":                     # precedence: geometry over the VLM's contradicted claims
            w = np.where(np.nan_to_num(f["claim_cons"], nan=1) < 0.5, w * symbolic["pi"], w)
        F += w[:, None] * evidence(f, s, prm["lam"], prm["rho"], prm["gam"])
        w_tot += w
    F += a[4] * f["qD"]
    return F / (w_tot + a[4])[:, None]


def agreement(f, prm, F):
    c = F.argmax(1)
    tops = np.stack([evidence(f, s, prm["lam"], prm["rho"], prm["gam"]).argmax(1) for s in EXPERTS], 1)
    w = np.array(prm["alpha"][:4])
    agree = ((tops == c[:, None]) * w).sum(1) / max(w.sum(), 1e-9)
    faith = np.stack([f[f"faith_{s}"] for s in EXPERTS], 1)
    outlier = (tops != c[:, None]) & (faith < 0.3)
    return agree, outlier


def search_fusion(fv, n_iter=600):
    """Random search on validation (macro-F1), starting from the probability-only weighted average."""
    cand = [dict(alpha=[0.25, 0.25, 0.25, 0.25, 0.0], lam=1.0, rho=0.0, gam=0.0)]
    for _ in range(n_iter):
        a = RNG.dirichlet(np.ones(5)).tolist()
        if RNG.random() < 0.3:
            a[4] = 0.0
        cand.append(dict(alpha=a, lam=float(RNG.choice([1.0, RNG.random()])), rho=float(RNG.choice([0, 0.5, 1])),
                         gam=float(RNG.choice([0, 1, 2]))))
    scored = [(macro_f1(fv["y"], fuse(fv, c).argmax(1)), i) for i, c in enumerate(cand)]
    return cand[max(scored)[1]], max(scored)[0]


def search_prob_only(fv, n_iter=400):
    cand = [dict(alpha=a, lam=1.0, rho=0.0, gam=0.0) for a in
            [[0.25, 0.25, 0.25, 0.25, 0.0]] + [RNG.dirichlet(np.ones(4)).tolist() + [0.0] for _ in range(n_iter)]]
    scored = [(macro_f1(fv["y"], fuse(fv, c).argmax(1)), i) for i, c in enumerate(cand)]
    return cand[max(scored)[1]]


# ============================================================================ step 6: symbolic decision
def symbolic_scores(f, prm, sym):
    """Precedence rule (inside fuse) and the AU tie-break for confusable pairs: when the two leading classes are a
    confusable pair and D's action-unit support favours one by at least the ratio kappa, that class leads."""
    F = fuse(f, prm, symbolic=sym)
    flip = np.zeros(f["n"], bool)
    if sym["kappa"]:
        conf = ONT.confusable_index()
        o = np.argsort(-F, 1)
        c1, c2 = o[:, 0], o[:, 1]
        pair = np.array([frozenset((a, b)) in conf for a, b in zip(c1, c2)])
        r = np.arange(f["n"])
        flip = pair & (f["qD"][r, c2] >= sym["kappa"] * f["qD"][r, c1])
        F = F.copy()
        F[r[flip], c1[flip]], F[r[flip], c2[flip]] = F[r[flip], c2[flip]], F[r[flip], c1[flip]]
    return F, flip


def tune_thresholds(F, y, risk=0.05):
    best = (0, 0.5, 0.2)
    for tau in np.linspace(0.2, 0.9, 36):
        for delta in np.linspace(0, 0.6, 31):
            k, c1, c2 = decide(F, tau, delta)
            cov, r, *_ = selective(y, k, c1, c2)
            if r <= risk and cov > best[0]:
                best = (cov, tau, delta)
    return best


def rules_fired(i, f, r, ev):
    """Names of the ontology rules that fired for image i (Stage 3, step 2), for the justification trace."""
    out = []
    q = f["qD"][i]
    if ev["found"][i]:
        out.append(f"AU-pattern: action units favour {CLASSES[int(q.argmax())]}")
    for s in EXPERTS:
        if f[f"gate_{s}"][i] < 1:
            out.append(f"sanity-gate: {s} explanation lies on the forehead")
    cc = f["claim_cons"][i]
    if r["sym"]["pi"] < 1 and not np.isnan(cc) and cc < 0.5:
        out.append(f"precedence: geometry over VLM claims ({cc:.0%} verified)")
    if r["flip"][i]:
        out.append(f"confusable-pair tie-break by action units: {CLASSES[r['c1'][i]]} over {CLASSES[r['c2'][i]]}")
    if r["outl"][i].any():
        out.append("outlier: " + ", ".join(s for s, o in zip(EXPERTS, r["outl"][i]) if o))
    out.append({0: "decide: label (top score and margin pass)", 1: "decide: confusable pair, compound label",
                2: "decide: abstain, refer to a person"}[int(r["kind"][i])])
    return out


def fused_explanation(r, ev):
    """Stage 2 OUT, saved per image: candidate emotions, consensus region and AU weights for the leading emotion,
    per-source support or conflict, agreement score, and the rules that fired."""
    f, prm, F = r["f"], r["prm"], r["F"]
    rows = []
    idx = {a: j for j, a in enumerate(aus())}
    for i in range(f["n"]):
        c = int(r["c1"][i])
        top3 = np.argsort(-F[i])[:3]
        reg, wsum = np.zeros(len(REGIONS)), 0.0
        for j, s in enumerate(EXPERTS):
            if ev[f"occ_{s}"][i]:
                d = np.clip(ev[f"P_full_{s}"][i, c] - ev[f"P_drop_{s}"][i, :, c], 0, None) @ FINE2COARSE
                if d.sum() > 0:
                    w = prm["alpha"][j] * f[f"faith_{s}"][i]
                    reg += w * d / d.sum(); wsum += w
        reg = reg / wsum if wsum > 0 else reg
        au_w = {a: round(float(ev["au_graded"][i, idx[ONT.SUBSTITUTE.get(a, a)]]), 2) for a in ONT.AU_CONFIG[CLASSES[c]]}
        stance = {s: ("support" if evidence(f, s, prm["lam"], prm["rho"], prm["gam"])[i].argmax() == c else "conflict")
                  for s in EXPERTS}
        rows.append(dict(image_id=ev["ids"][i], label=CLASSES[ev["y"][i]],
                         decision={0: "label", 1: "compound", 2: "abstain"}[int(r["kind"][i])],
                         emotion=CLASSES[c], second=CLASSES[int(r["c2"][i])],
                         candidates="; ".join(f"{CLASSES[k]} {F[i, k]:.2f}" for k in top3),
                         **{f"region_{g}": round(float(v), 3) for g, v in zip(REGIONS, reg)},
                         au_weights=json.dumps(au_w), agreement=round(float(r["agree"][i]), 3),
                         **{f"{s}_stance": stance[s] for s in EXPERTS},
                         **{f"{s}_faith": round(float(f[f"faith_{s}"][i]), 3) for s in EXPERTS},
                         vlm_claims_verified=None if np.isnan(f["claim_cons"][i]) else round(float(f["claim_cons"][i]), 2),
                         rules=" | ".join(rules_fired(i, f, r, ev))))
    return pd.DataFrame(rows)


def trace(i, f, prm, F, kind, c1, c2, ev, agree):
    parts = []
    for j, s in enumerate(EXPERTS):
        E = evidence(f, s, prm["lam"], prm["rho"], prm["gam"])[i]
        c = int(E.argmax())
        parts.append(f"{s}: {CLASSES[c]} (p={f[f'P_{s}'][i, c]:.2f}, support {f[f'sup_{s}'][i, c]:.2f}, "
                     f"faith {f[f'faith_{s}'][i]:.2f}{', gated' if f[f'gate_{s}'][i] < 1 else ''})")
    if ev["has_desc"][i]:
        d = ev["desc"].iloc[i]
        parts.append(f"VLM says: {d.get('reason', '')} [claims verified {f['claim_cons'][i]:.0%}]")
    q = f["qD"][i]
    top = np.argsort(-q)[:2]
    parts.append(f"D (action units): {CLASSES[top[0]]} {q[top[0]]:.2f}, {CLASSES[top[1]]} {q[top[1]]:.2f}")
    verdict = {0: f"LABEL {CLASSES[c1[i]]}", 1: f"BLEND {CLASSES[c1[i]]}-{CLASSES[c2[i]]}",
               2: "ABSTAIN (refer to a person)"}[kind[i]]
    parts.append(f"fused {CLASSES[c1[i]]} {F[i, c1[i]]:.2f}, {CLASSES[c2[i]]} {F[i, c2[i]]:.2f}; agreement "
                 f"{agree[i]:.2f} -> {verdict}")
    return " | ".join(parts)


# ============================================================================ main
def ci(y, p):
    a, f = bootstrap_ci(accuracy, y, p), bootstrap_ci(macro_f1, y, p)
    return f"{100 * a[0]:.1f} ({100 * a[1]:.1f}-{100 * a[2]:.1f})", f"{f[0]:.3f} ({f[1]:.3f}-{f[2]:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--risk", type=float, default=0.05)
    ap.add_argument("--owl", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "ontology", "xfer_emonto.owl"))
    a = ap.parse_args()
    if a.owl and os.path.exists(a.owl):
        ONT.use_owl(a.owl)   # the reasoner's knowledge comes from the EMONTO extension itself
        global CLASS_REG
        CLASS_REG = np.array([[r in ONT.class_regions(c) for r in REGIONS] for c in CLASSES], float)
    thr, measurable = au_thresholds(a.out)
    tr, va, te = (load_split(a.out, s, thr, measurable) for s in ("train", "val", "test"))
    L, J = [], {}

    # ---------------------------------------------------------------- step 3: claims
    L += ["# New framework: claims, fusion and symbolic decision (CPU)", "", "## Step 3: the VLM's claims checked "
          "against MediaPipe", ""]
    rows = []
    for name, ev in [("val", va), ("test", te)]:
        ok = ~np.isnan(ev["claim_cons"])
        fl = ev["claim_flags"]
        rows.append(dict(split=name, images=int(ok.sum()), mean_claims=round(ev["n_claims"][ok].mean(), 2),
                         verified=f"{100 * np.nanmean(ev['claim_cons']):.1f}%",
                         **{f"{fld}": f"{100 * np.nanmean(fl[:, j]):.0f}%" for j, fld in enumerate(FIELDS)},
                         correct_when_verified=f"{100 * (ev['desc_emotion'][ev['claim_cons'] >= 0.8] == ev['y'][ev['claim_cons'] >= 0.8]).mean():.1f}%",
                         correct_when_contradicted=f"{100 * (ev['desc_emotion'][ev['claim_cons'] < 0.5] == ev['y'][ev['claim_cons'] < 0.5]).mean():.1f}%"))
    L += [pd.DataFrame(rows).to_markdown(index=False), "",
          "Columns brows to mouth: share of that field's claims that MediaPipe confirms. Last two columns: accuracy of "
          "the VLM's described emotion when at least 80% of its claims are verified, and when fewer than half are.", ""]
    # reference explanations for the SLM (emotion given in the prompt; train subset and val only)
    for split in ("train", "val"):
        gf = os.path.join(a.out, "descriptions", f"V2_{split}_given.csv")
        if os.path.exists(gf):
            g = pd.read_csv(gf).drop_duplicates("image_id", keep="last")
            ev = tr if split == "train" else va
            pos = {i: j for j, i in enumerate(ev["ids"])}
            idx = np.array([pos[str(i)] for i in g.image_id])
            cons, *_ = claim_table(g, ev["au_active"][idx], aus(), measurable)
            L.append(f"Reference explanations ({split}, true emotion given): {len(g)} images, "
                     f"{100 * np.nanmean(cons):.1f}% of claims verified; {100 * np.mean(cons >= 0.8):.1f}% of images "
                     f"have at least 80% verified (these are kept as SLM training targets).")
    L.append("")
    # does blurring the region a claim talks about change the VLM's answer? (stratified sample only)
    for name, ev in [("test", te)]:
        have = ev["occ_V2"] & ev["has_desc"]
        hits, tot = 0, 0
        for i in np.where(have)[0]:
            t = int(ev["P_full_V2"][i].argmax())
            drop = ev["P_full_V2"][i, t] - ev["P_drop_V2"][i, :, t]
            coarse = drop @ FINE2COARSE
            for fld in ev["claim_fields"][i]:
                tot += 1
                hits += coarse[REGIONS.index(FIELD_REGION[fld])] > 0.01
        L += [f"Claim-occlusion check ({name}, {int(have.sum())} sampled images): blurring the region named in a "
              f"positive claim lowers the VLM's confidence in {100 * hits / max(tot, 1):.1f}% of {tot} claims.", ""]

    # ---------------------------------------------------------------- calibration, reliability, compensator
    cal = dict(T={}, rel={})
    for s in EXPERTS:
        cal["T"][s] = fit_temperature(va[f"P_{s}"], va["y"])
    apply_comp, tau_d, n_comp, n_fail = train_compensator(tr, va)
    for ev in (tr, va, te):
        ev["P_V2c"], ev["fire"], _ = apply_comp(ev, tau_d)
    cal["T"]["V2c"] = fit_temperature(va["P_V2c"], va["y"])
    for s in EXPERTS + ["V2c"]:
        cal["rel"][s] = class_reliability(apply_t(va[f"P_{s}"], cal["T"][s]), va["y"])
    J["temperature"] = cal["T"]

    # ---------------------------------------------------------------- explanation quality per expert
    L += ["## Step 2 summary: the shared explanation, on one scale (1-step, confidence-normalised)", ""]
    rows = []
    for s in EXPERTS:
        sup, gate, faith = translate(te, s)
        have = te[f"occ_{s}"]
        t = te[f"P_{s}"].argmax(1)
        rows.append(dict(expert=NAMES[s], images=int(have.sum()), faithfulness=round(np.nanmean(faith), 3),
                         forehead_gated=f"{100 * (gate[have] < 1).mean():.1f}%",
                         region_support_pred=round(sup[have][np.arange(have.sum()), t[have]].mean(), 3)))
    L += [pd.DataFrame(rows).to_markdown(index=False), "",
          "Faithfulness: mean of (drop when the top region is blurred) and (recovery when only it is restored), each "
          "relative to the expert's own confidence; 1 = perfectly faithful. Region support: share of the evidence "
          "for the predicted emotion that lies in the regions its action units move (ontology).", ""]

    # ---------------------------------------------------------------- fusion and symbolic stage, two VLM variants
    results, systems = {}, {}
    y = te["y"]
    for s in EXPERTS + ["V2c"]:
        systems[NAMES[s]] = apply_t(te[f"P_{s}"], cal["T"][s]).argmax(1)
    votes = np.stack([te[f"P_{s}"].argmax(1) for s in EXPERTS], 1)
    systems["Majority vote"] = np.array([np.bincount(v, minlength=K).argmax() for v in votes])
    J["params"] = {}
    for vlm in ("V2", "V2c"):
        fv, ft = prepare(va, cal, vlm), prepare(te, cal, vlm)
        wavg = search_prob_only(fv)
        prm, f1v = search_fusion(fv)
        best_sym, best_f1 = None, -1
        for pi in (1.0, 0.5, 0.25):
            for kappa in (0, 1.25, 1.5, 2.0):
                sym = dict(pi=pi, kappa=kappa)
                f1 = macro_f1(fv["y"], symbolic_scores(fv, prm, sym)[0].argmax(1))
                if f1 > best_f1 + 1e-9:
                    best_sym, best_f1 = sym, f1
        tag = "" if vlm == "V2" else " (VLM compensated)"
        systems["Weighted average" + tag] = fuse(ft, wavg).argmax(1)
        systems["XAI fusion" + tag] = fuse(ft, prm).argmax(1)
        Ft, flip_t = symbolic_scores(ft, prm, best_sym)
        systems["Full framework" + tag] = Ft.argmax(1)
        Fv, _ = symbolic_scores(fv, prm, best_sym)
        cov_v, tau, delta = tune_thresholds(Fv, fv["y"], a.risk)
        kind, c1, c2 = decide(Ft, tau, delta)
        cov, risk, lab, blend = selective(y, kind, c1, c2)
        agree, outl = agreement(ft, prm, Ft)
        blends_ok = ((c1 == y) | (c2 == y))[kind == 1].mean() if (kind == 1).any() else float("nan")
        results[vlm] = dict(cov=cov, risk=risk, lab=lab, blend=blend, blends_ok=blends_ok, tau=tau, delta=delta,
                            agree_acc=agree[kind < 2].mean(), agree_abst=agree[kind == 2].mean(),
                            outlier=outl.any(1).mean(), kind=kind, c1=c1, c2=c2, F=Ft, f=ft, prm=prm, agree=agree,
                            outl=outl, flip=flip_t, sym=best_sym)
        J["params"][vlm] = dict(fusion=prm, symbolic=best_sym, weighted_average=wavg, tau=tau, delta=delta,
                                val_macro_f1_fusion=f1v, val_macro_f1_symbolic=best_f1)
        pd.DataFrame(dict(image_id=te["ids"], label=y, pred=Ft.argmax(1), kind=kind, c1=c1, c2=c2,
                          agreement=agree)).to_csv(pred_path(a.out, f"NF{'c' if vlm == 'V2c' else ''}", "test"),
                                                   index=False)

    L += ["## Recognition on the RAF-DB test set (3,068 images)", ""]
    rows = []
    B = systems[NAMES["B"]]
    W = systems["Weighted average (VLM compensated)"]
    for name, p in systems.items():
        acc, f1 = ci(y, p)
        mb, mw = mcnemar(y, B, p), mcnemar(y, W, p)
        rows.append(dict(system=name, accuracy=acc, macro_F1=f1,
                         vs_POSTER=f"{mb['n01']}/{mb['n10']}, p={mb['p_value']:.3g}",
                         vs_weighted_avg=f"{mw['n01']}/{mw['n10']}, p={mw['p_value']:.3g}"))
    L += [pd.DataFrame(rows).to_markdown(index=False), "",
          "vs columns: images fixed/broken relative to POSTER++ and to the weighted average (VLM compensated), with "
          "McNemar's p.", "",
          f"Interim compensator (gradient-boosted trees on the VLM's probabilities, claims and face measurements; "
          f"the SLM replaces it): trained on {n_comp} explained training images ({n_fail} VLM failures); fires on "
          f"{100 * te['fire'].mean():.1f}% of test images.", ""]
    s_ = (te["occ_V2"])
    if s_.any():
        L += [f"On the {int(s_.sum())} test images where the VLM also has region occlusion: full framework (VLM "
              f"compensated) accuracy {100 * accuracy(y[s_], systems['Full framework (VLM compensated)'][s_]):.1f}% vs "
              f"{100 * accuracy(y[s_], systems['Full framework'][s_]):.1f}% (raw VLM); weighted average "
              f"{100 * accuracy(y[s_], W[s_]):.1f}%; POSTER++ {100 * accuracy(y[s_], B[s_]):.1f}%.", ""]

    L += ["## Symbolic decision (label, blend or abstain) on test", ""]
    rows = []
    for vlm, r in results.items():
        rows.append(dict(VLM=NAMES[vlm], tau=round(r["tau"], 3), delta=round(r["delta"], 3),
                         coverage=f"{100 * r['cov']:.1f}%", selective_risk=f"{100 * r['risk']:.1f}%",
                         labels=f"{100 * r['lab']:.1f}%", blends=f"{100 * r['blend']:.1f}%",
                         blends_with_truth=f"{100 * r['blends_ok']:.1f}%",
                         agreement_accepted=round(r["agree_acc"], 2), agreement_abstained=round(r["agree_abst"], 2),
                         images_with_outlier=f"{100 * r['outlier']:.1f}%"))
    L += [pd.DataFrame(rows).to_markdown(index=False), "",
          f"POSTER++ error on all test images: {100 * (1 - accuracy(y, B)):.1f}%.", ""]
    r = results["V2c"]
    L += ["## Example justification traces (test)", ""]
    for kind_ in (0, 1, 2):
        idx = np.where((r["kind"] == kind_) & te["has_desc"])[0]
        if len(idx):
            i = int(idx[0])
            L.append(f"- true {CLASSES[y[i]]}: " + trace(i, r["f"], r["prm"], r["F"], r["kind"], r["c1"], r["c2"],
                                                          te, r["agree"]) + " | rules: " + "; ".join(rules_fired(i, r["f"], r, te)))
    L.append("")
    os.makedirs(os.path.join(a.out, "explanations"), exist_ok=True)
    fe = fused_explanation(results["V2c"], te)
    fe.to_csv(os.path.join(a.out, "explanations", "fused_test.csv"), index=False)
    L += [f"Fused explanations for all {len(fe)} test images: explanations/fused_test.csv (candidates, consensus "
          f"region and AU weights, per-source support or conflict, agreement, rules fired). Knowledge read from "
          f"{ONT.SOURCE_OF_KNOWLEDGE}.", ""]
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    open(os.path.join(a.out, "results", "new_framework.md"), "w").write("\n".join(L))
    json.dump(J, open(os.path.join(a.out, "results", "new_framework.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
