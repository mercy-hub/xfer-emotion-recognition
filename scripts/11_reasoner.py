"""Step 11: experiments E2 and E3 with the ontology-grounded reasoner. CPU, a few minutes.

Everything is fitted on the validation split; the test split is used once at the end.
Systems compared on the test split (forced decisions, E2):
  B alone; majority vote of V, A, B, C; confidence-weighted average (weights = validation macro-F1);
  tuned weighted average (probabilities only: lam = 1, no source D; weights searched on validation) - the ablation;
  reasoner without compensator (explanations and source D, V temperature-scaled);
  reasoner with V and A prior-recalibrated (the E4 baseline standing in for the compensator; exploratory).
E3: thresholds tau and delta chosen on validation as the largest coverage whose selective risk is at most --risk;
coverage, selective risk, blends and abstentions reported on test; risk-coverage curves for B and the reasoner.
Writes results/e2_test.csv/.md, results/e3_test.csv, results/reasoner_params.json, results/traces_test.csv,
figures/risk_coverage.png.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xfer.labels import CLASSES
from xfer.metrics import accuracy, bootstrap_ci, macro_f1, mcnemar
from xfer import ontology as ONT
from xfer.reasoner import (SOURCES, au_thresholds, brackets, consistency, decide, load, score, selective, trace,
                           translate)


def search(ev, tr_by_mode, n, rng, probs_only=False, start=None):
    """Random search on the validation split, then local refinement around the best point. `start` (for the full
    reasoner, the probabilities-only optimum) is evaluated first, so the full model can never do worse on
    validation than the special case it contains."""
    def evaluate(p):
        a = np.array(p["alpha"])
        return macro_f1(ev["y"], score(brackets(ev, tr_by_mode[p["mode"]], p["lam"]), a).argmax(1))

    cands = [] if start is None else [dict(start)]
    for _ in range(n):
        a = rng.dirichlet(np.ones(5))
        if probs_only:
            a[4] = 0; a = a / a.sum()
        cands.append(dict(mode="soft" if probs_only else str(rng.choice(["soft", "hard"])),
                          lam=1.0 if probs_only else float(rng.uniform(0, 1)), alpha=a.tolist()))
    best = max(((evaluate(p), p) for p in cands), key=lambda t: t[0])
    for _ in range(n):                                   # local refinement
        p = dict(best[1])
        a = rng.dirichlet(np.array(p["alpha"]) * 300 + 0.05)
        if probs_only:
            a[4] = 0; a = a / a.sum()
        lam = p["lam"] if probs_only else float(np.clip(p["lam"] + rng.normal(0, 0.05), 0, 1))
        q = dict(mode=p["mode"], lam=lam, alpha=a.tolist())
        f = evaluate(q)
        if f > best[0]:
            best = (f, q)
    return best


def risk_at(conf_s, wrong, cov):
    """Selective risk when the `cov` most confident images are accepted."""
    k = max(1, int(round(cov * len(conf_s))))
    return float(wrong[np.argsort(-conf_s)[:k]].mean())


def majority(ev):
    votes = np.stack([ev[f"P_{s}"].argmax(1) for s in SOURCES], 1)
    tot = sum(ev[f"P_{s}"] for s in SOURCES)
    out = []
    for v, t in zip(votes, tot):
        cnt = np.bincount(v, minlength=7).astype(float) + t * 1e-3       # ties broken by summed probability
        out.append(cnt.argmax())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_search", type=int, default=3000)
    ap.add_argument("--risk", type=float, default=0.05, help="target selective risk on validation (E3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--owl", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "ontology", "xfer_emonto.owl"),
                    help="the ontology the rules are read from ('' to use xfer/ontology.py directly)")
    a = ap.parse_args()
    if a.owl:
        print("rules read from the ontology:", ONT.use_owl(a.owl))
    rng = np.random.default_rng(a.seed)
    res = os.path.join(a.out, "results"); os.makedirs(res, exist_ok=True)
    thr, meas = au_thresholds(a.out)
    print("action units measurable on these faces:",
          [au for au, ok in zip(list(ONT.AU_BLENDSHAPES), meas) if ok])

    variants = {"raw": {}, "recal": {"V": "Vcal", "A": "Acal"}}
    ev = {(v, s): load(a.out, s, tags) for v, tags in variants.items() for s in ("val", "test")}
    tr = {(v, s, mo): translate(ev[(v, s)], thr, meas, mo) for v in variants for s in ("val", "test")
          for mo in ("soft", "hard")}

    params, rows, preds = {}, [], {}
    yv, yt = ev[("raw", "val")]["y"], ev[("raw", "test")]["y"]
    # baselines on temperature-scaled probabilities
    preds["B: POSTER++ alone"] = ev[("raw", "test")]["P_B"].argmax(1)
    preds["Majority vote"] = majority(ev[("raw", "test")])
    w = np.array([macro_f1(yv, ev[("raw", "val")][f"P_{s}"].argmax(1)) for s in SOURCES])
    w = w / w.sum()
    preds["Confidence-weighted average"] = sum(wi * ev[("raw", "test")][f"P_{s}"] for wi, s in zip(w, SOURCES)).argmax(1)
    params["confidence_weights"] = dict(zip(SOURCES, w.round(4).tolist()))

    for v in variants:
        trv = {mo: tr[(v, "val", mo)] for mo in ("soft", "hard")}
        f, p = search(ev[(v, "val")], trv, a.n_search, rng, probs_only=True)
        params["probabilities_only_" + v] = p
        params["probabilities_only_" + v + "_val_macro_f1"] = f
    p = params["probabilities_only_raw"]
    sc = score(brackets(ev[("raw", "test")], tr[("raw", "test", p["mode"])], 1.0), np.array(p["alpha"]))
    preds["Tuned weighted average (probabilities only)"] = sc.argmax(1)

    final = {}
    for v, name in (("raw", "Reasoner without compensator"), ("recal", "Reasoner, V and A prior-recalibrated")):
        trv = {mo: tr[(v, "val", mo)] for mo in ("soft", "hard")}
        f, p = search(ev[(v, "val")], trv, a.n_search, rng, start=params["probabilities_only_" + v])
        params[name] = dict(val_macro_f1=f, **p)
        print(f"{name}: validation macro-F1 {f:.4f} (probabilities only: {params['probabilities_only_' + v + '_val_macro_f1']:.4f})")
        alpha = np.array(p["alpha"])
        brv = brackets(ev[(v, "val")], tr[(v, "val", p["mode"])], p["lam"])
        brt = brackets(ev[(v, "test")], tr[(v, "test", p["mode"])], p["lam"])
        scv, sct = score(brv, alpha), score(brt, alpha)
        preds[name] = sct.argmax(1)
        final[name] = (scv, sct, brt, p, v)

    # confusable-pair subset: the two top classes of the averaged probabilities form a pair in the ontology
    avg = sum(ev[("raw", "test")][f"P_{s}"] for s in SOURCES)
    top2 = np.argsort(-avg, 1)[:, :2]
    conf = ONT.confusable_index()
    subset = np.array([frozenset(t) in conf for t in top2])
    for name, pr in preds.items():
        acc = bootstrap_ci(accuracy, yt, pr, 1000)
        f1 = bootstrap_ci(macro_f1, yt, pr, 1000)
        mc = mcnemar(yt, pr, preds["B: POSTER++ alone"]) if name != "B: POSTER++ alone" else None
        rows.append(dict(system=name, accuracy=acc[0], acc_low=acc[1], acc_high=acc[2], macro_f1=f1[0],
                         f1_low=f1[1], f1_high=f1[2], confusable_subset_acc=accuracy(yt[subset], pr[subset]),
                         vs_B_n01=mc["n01"] if mc else None, vs_B_n10=mc["n10"] if mc else None,
                         vs_B_p=mc["p_value"] if mc else None))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(res, "e2_test.csv"), index=False)
    with open(os.path.join(res, "e2_test.md"), "w") as fh:
        fh.write(f"Confusable-pair subset: {subset.sum()} test images\n\n| System | Accuracy (95% CI) | Macro-F1 (95% CI) | "
                 "Confusable-pair acc. | vs B: n01 / n10, p |\n|---|---|---|---|---|\n")
        for r in rows:
            vb = f"{r['vs_B_n01']} / {r['vs_B_n10']}, {r['vs_B_p']:.3g}" if r["vs_B_n01"] is not None else "-"
            fh.write(f"| {r['system']} | {r['accuracy']:.4f} ({r['acc_low']:.4f}-{r['acc_high']:.4f}) | "
                     f"{r['macro_f1']:.4f} ({r['f1_low']:.4f}-{r['f1_high']:.4f}) | {r['confusable_subset_acc']:.4f} | {vb} |\n")
    print(open(os.path.join(res, "e2_test.md")).read())

    # ---------------- E3: abstention and blending
    e3, curves = [], {}
    for name, (scv, sct, brt, p, v) in final.items():
        best = None
        for tau in np.linspace(0.2, 0.9, 36):
            for delta in np.linspace(0, 0.6, 31):
                kv, c1v, c2v = decide(scv, tau, delta)
                cov, risk, lab, bl = selective(yv, kv, c1v, c2v)
                if risk <= a.risk and (best is None or cov > best[0]):
                    best = (cov, tau, delta)
        cov_v, tau, delta = best
        kt, c1t, c2t = decide(sct, tau, delta)
        cov, risk, lab, bl = selective(yt, kt, c1t, c2t)
        blend_hit = ((c1t == yt) | (c2t == yt))[kt == 1].mean() if (kt == 1).any() else float("nan")
        cons = consistency(brt, c1t)
        e3.append(dict(system=name, tau=tau, delta=delta, val_coverage=cov_v, coverage=cov, selective_risk=risk,
                       single_labels=lab, blends=bl, abstentions=1 - cov, blend_contains_truth=blend_hit,
                       error_best_single=1 - accuracy(yt, preds["B: POSTER++ alone"]),
                       mean_consistency_accepted=cons[kt < 2].mean(), mean_consistency_abstained=cons[kt == 2].mean()))
        params[name].update(tau=float(tau), delta=float(delta))
        margin = np.sort(sct, 1)[:, -1] - np.sort(sct, 1)[:, -2]
        curves[name] = (margin, c1t != yt)
        pb = ev[("raw", "test")]["P_B"]
        e3[-1]["B_risk_same_coverage"] = risk_at(pb.max(1), pb.argmax(1) != yt, cov)
        e3[-1]["reasoner_margin_risk_same_coverage"] = risk_at(margin, c1t != yt, cov)
        if v == "raw":
            ev_t = ev[("raw", "test")]
            trt = tr[("raw", "test", p["mode"])]
            tr_rows = [dict(image_id=ev_t["ids"][i], label=CLASSES[yt[i]], decision=int(kt[i]),
                            consistency=float(cons[i]),
                            trace=trace(i, ev_t, trt, brt, sct, kt, c1t, c2t, np.array(p["alpha"])))
                       for i in range(len(yt))]
            pd.DataFrame(tr_rows).to_csv(os.path.join(res, "traces_test.csv"), index=False)
    e3 = pd.DataFrame(e3)
    e3.to_csv(os.path.join(res, "e3_test.csv"), index=False)
    print(e3.round(4).T.to_string())

    pb = ev[("raw", "test")]["P_B"]
    curves["B: POSTER++ alone (max probability)"] = (pb.max(1), pb.argmax(1) != yt)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    styles = ["-", "--", ":"]
    for (name, (conf_s, wrong)), ls in zip(curves.items(), styles):
        o = np.argsort(-conf_s)
        cov = np.arange(1, len(o) + 1) / len(o)
        risk = np.cumsum(wrong[o]) / np.arange(1, len(o) + 1)
        ax.plot(cov, risk, ls, color="black", lw=1.2, label=name)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (error among accepted)")
    ax.legend(fontsize=6.5, frameon=False); ax.set_ylim(0, None)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    fig.savefig(os.path.join(a.out, "figures", "risk_coverage.png"), dpi=220); plt.close(fig)
    with open(os.path.join(res, "reasoner_params.json"), "w") as fh:
        json.dump(params, fh, indent=2)
    print(json.dumps(params, indent=1))


if __name__ == "__main__":
    main()
