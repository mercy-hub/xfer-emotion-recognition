"""Step 12 (E4, CPU arm): detector and corrector built from V's own evidence, with input ablations. Minutes.

Training images: the training-split images that have V's region explanations (a stratified subset, step 9), so that
every ablation is trained on the same images. The detector learns from all of them whether V is wrong; the corrector
learns from V's failures only which class is right. Model type (logistic regression or gradient-boosted trees) and
the detector threshold are chosen on the validation split; the test split is used once.
Ablations (feature blocks): pred, pred+expl, pred+face, pred+expl+face (xfer/compensator.py).
Reference rows: V alone, V with prior recalibration (the E4 baseline), and each corrector applied to every image
without the detector.
Writes results/e4_tabular_test.csv/.md and predictions/Vcomp-<block>_<split>.csv.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xfer.compensator import BLOCKS, correction_counts, features, gate_threshold, load_split
from xfer.io import append_predictions, pred_path, read_predictions
from xfer.labels import K
from xfer.metrics import accuracy, bootstrap_ci, macro_f1, mcnemar


def models(seed):
    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000,
                                                                               class_weight="balanced")),
        "boosted": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                          l2_regularization=1.0, class_weight="balanced",
                                                          random_state=seed),
    }


def proba_full(clf, X):
    """Probabilities over all K classes even if some class never appeared among the failures."""
    p = np.zeros((len(X), K))
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    res = os.path.join(a.out, "results")
    os.makedirs(res, exist_ok=True)
    tr_ids = pd.read_csv(os.path.join(a.out, "explanations", "V_train.csv")).image_id.astype(str).sort_values().values
    D = {"train": load_split(a.out, "train", tr_ids), "val": load_split(a.out, "val"), "test": load_split(a.out, "test")}
    for s, d in D.items():
        print(f"{s}: {len(d['y'])} images, V wrong on {np.mean(d['pred'] != d['y']):.3f}")
    yt, base_t = D["test"]["y"], D["test"]["pred"]
    rows, params = [], {}

    def add(name, final, det_auc=None, fire=None):
        fx, br, net = correction_counts(base_t, final, yt)
        acc = bootstrap_ci(accuracy, yt, final, 1000)
        f1 = bootstrap_ci(macro_f1, yt, final, 1000)
        rows.append(dict(system=name, accuracy=acc[0], acc_low=acc[1], acc_high=acc[2], macro_f1=f1[0],
                         f1_low=f1[1], f1_high=f1[2], fixed=fx, broken=br, net_correction=net,
                         detector_auroc=det_auc, share_corrected=fire))
        return final

    add("V alone", base_t)
    cal = read_predictions(pred_path(a.out, "Vcal", "test"))
    cal["image_id"] = cal.image_id.astype(str)
    cal_pred = add("V with prior recalibration (baseline)", cal.set_index("image_id").loc[D["test"]["ids"]].pred.values)

    finals = {}
    for block in BLOCKS:
        X = {s: features(d, block) for s, d in D.items()}
        z = {s: (d["pred"] != d["y"]).astype(int) for s, d in D.items()}
        # detector: chosen by validation AUROC
        det_best = None
        for name, mk in models(a.seed).items():
            det = mk().fit(X["train"], z["train"])
            auc = roc_auc_score(z["val"], det.predict_proba(X["val"])[:, 1])
            if det_best is None or auc > det_best[0]:
                det_best = (auc, name, det)
        # corrector: trained on failures only, chosen by validation accuracy on validation failures
        f_tr, f_va = z["train"] == 1, z["val"] == 1
        cor_best = None
        for name, mk in models(a.seed).items():
            cor = mk().fit(X["train"][f_tr], D["train"]["y"][f_tr])
            acc_f = accuracy(D["val"]["y"][f_va], proba_full(cor, X["val"][f_va]).argmax(1))
            if cor_best is None or acc_f > cor_best[0]:
                cor_best = (acc_f, name, cor)
        det, cor = det_best[2], cor_best[2]
        dv, dt = det.predict_proba(X["val"])[:, 1], det.predict_proba(X["test"])[:, 1]
        cv, ct = proba_full(cor, X["val"]).argmax(1), proba_full(cor, X["test"]).argmax(1)
        thr = gate_threshold(dv, D["val"]["pred"], cv, D["val"]["y"])
        fire = dt >= thr
        final = np.where(fire, ct, base_t)
        auc_t = roc_auc_score(z["test"], dt)
        finals[block] = add(f"Compensator ({block})", final, auc_t, float(fire.mean()))
        add(f"Corrector on every image, no detector ({block})", ct)
        params[block] = dict(detector=det_best[1], detector_val_auroc=det_best[0], corrector=cor_best[1],
                             corrector_val_failure_accuracy=cor_best[0], threshold=thr)
        for s in ("val", "test"):                       # uniform files for step 15
            sc_s = dv if s == "val" else dt
            pd.DataFrame(dict(image_id=D[s]["ids"], label=D[s]["y"], score=sc_s)).to_csv(
                pred_path(a.out, f"det-{block}", s), index=False)
            p = pred_path(a.out, f"corr-tab-{block}", s)
            if os.path.exists(p):
                os.remove(p)
            append_predictions(p, D[s]["ids"], D[s]["y"], proba_full(cor, X[s]))
        for s in ("val", "test"):
            fire_s = (dv if s == "val" else dt) >= thr
            fin = np.where(fire_s, cv if s == "val" else ct, D[s]["pred"])
            p = pred_path(a.out, f"Vcomp-{block}", s)
            if os.path.exists(p):
                os.remove(p)
            append_predictions(p, D[s]["ids"], D[s]["y"], np.eye(K)[fin], extra={"detector": dv if s == "val" else dt})
        print(f"{block}: detector {det_best[1]} (val AUROC {det_best[0]:.3f}), corrector {cor_best[1]} "
              f"(val failure acc {cor_best[0]:.3f}), threshold {thr:.3f}")

    t = pd.DataFrame(rows)
    full = finals["pred+expl+face"]
    t["vs_recal_p"] = [mcnemar(yt, r_final, cal_pred)["p_value"] if r_final is not None else None
                       for r_final in [None] * len(t)]
    for block, fin in finals.items():
        r = mcnemar(yt, fin, cal_pred)
        t.loc[t.system == f"Compensator ({block})", "vs_recal_p"] = r["p_value"]
        t.loc[t.system == f"Compensator ({block})", "vs_recal_n01_n10"] = f"{r['n01']} / {r['n10']}"
    t.to_csv(os.path.join(res, "e4_tabular_test.csv"), index=False)
    with open(os.path.join(res, "e4_tabular_test.md"), "w") as fh:
        fh.write("| System | Accuracy (95% CI) | Macro-F1 (95% CI) | Fixed | Broken | Net correction | Detector AUROC | "
                 "Share corrected |\n|---|---|---|---|---|---|---|---|\n")
        for r in t.itertuples():
            da = f"{r.detector_auroc:.3f}" if r.detector_auroc == r.detector_auroc and r.detector_auroc is not None else "-"
            sc = f"{r.share_corrected:.3f}" if r.share_corrected == r.share_corrected and r.share_corrected is not None else "-"
            fh.write(f"| {r.system} | {r.accuracy:.4f} ({r.acc_low:.4f}-{r.acc_high:.4f}) | {r.macro_f1:.4f} "
                     f"({r.f1_low:.4f}-{r.f1_high:.4f}) | {r.fixed} | {r.broken} | {r.net_correction:.4f} | {da} | {sc} |\n")
    with open(os.path.join(res, "e4_tabular_params.json"), "w") as fh:
        json.dump(params, fh, indent=2)
    print(open(os.path.join(res, "e4_tabular_test.md")).read())
    print(t[["system", "vs_recal_n01_n10", "vs_recal_p"]].dropna().to_string(index=False))


if __name__ == "__main__":
    main()
