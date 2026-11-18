"""Step 15 (E4): evaluate every compensator arm the same way and write Table 18. CPU, a minute.

Inputs (from steps 12 to 14):
  predictions/det-<block>_<split>.csv           detector scores (probability that V is wrong)
  predictions/corr-<arm>-<ablation>_<split>.csv corrector class probabilities (images it was not run on are missing)
Tabular arms use the detector built from the same feature block; image arms share the full detector
(pred+expl+face). For each arm the detector threshold is chosen on validation to maximise net corrections; where the
corrector was not run, V's answer is kept. Test metrics: accuracy, macro-F1, fixed, broken, net correction rate with a
95% bootstrap interval, detector AUROC, share corrected, and McNemar against the prior-recalibration baseline.
Writes results/e4_test.csv and results/e4_test.md, and predictions/Vcomp-<arm>-<ablation>_<split>.csv (the
compensated V, one-hot) for the image arms, so that step 16 can run the full framework with any arm.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from xfer.compensator import correction_counts, gate_threshold
from xfer.io import append_predictions, pred_path, read_predictions
from xfer.labels import prob_columns
from xfer.metrics import accuracy, bootstrap_ci, macro_f1, mcnemar

SHARED_DET = "pred+expl+face"


def frame(out, tag, split, ids):
    p = read_predictions(pred_path(out, tag, split))
    p["image_id"] = p.image_id.astype(str)
    return p.set_index("image_id").reindex(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    data = {}
    for s in ("val", "test"):
        v = read_predictions(pred_path(a.out, "VT", s))
        v["image_id"] = v.image_id.astype(str)
        v = v.sort_values("image_id")
        data[s] = dict(ids=v.image_id.values, y=v.label.values.astype(int), base=v[prob_columns()].values.argmax(1))
    yt, bt = data["test"]["y"], data["test"]["base"]
    cal = frame(a.out, "Vcal", "test", data["test"]["ids"]).pred.values

    arms = sorted({re.sub(r"_(val|test)\.csv$", "", os.path.basename(f))[5:]
                   for f in glob.glob(os.path.join(a.out, "predictions", "corr-*_test.csv"))})
    rows = [dict(system="V alone", arm="-", ablation="-", final=bt),
            dict(system="V with prior recalibration (baseline)", arm="-", ablation="-", final=cal)]
    for arm_abl in arms:
        arm, abl = arm_abl.split("-", 1)
        det_block = abl if arm == "tab" else SHARED_DET
        fin, det_t, fire_t = {}, None, None
        for s in ("val", "test"):
            d = data[s]
            det = pd.read_csv(pred_path(a.out, f"det-{det_block}", s))
            det["image_id"] = det.image_id.astype(str)
            score = det.set_index("image_id").reindex(d["ids"]).score.values
            c = frame(a.out, f"corr-{arm_abl}", s, d["ids"])
            have = ~c[prob_columns()].isna().any(axis=1).values
            corr = np.where(have, np.nan_to_num(c[prob_columns()].values).argmax(1), d["base"])
            score_g = np.where(have, score, -1.0)            # never fire where the corrector was not run
            if s == "val":
                thr = gate_threshold(score_g, d["base"], corr, d["y"])
            fire = score_g >= thr
            fin[s] = np.where(fire, corr, d["base"])
            if arm != "tab":                                   # step 12 already wrote Vcomp-<block> for the tabular arm
                fp = pred_path(a.out, f"Vcomp-{arm_abl}", s)
                if os.path.exists(fp):
                    os.remove(fp)
                append_predictions(fp, d["ids"], d["y"], np.eye(len(prob_columns()))[fin[s]])
            if s == "test":
                det_t, fire_t = score, fire
        rows.append(dict(system=f"{arm} corrector, {abl}", arm=arm, ablation=abl, final=fin["test"], threshold=thr,
                         detector_auroc=roc_auc_score((bt != yt).astype(int), det_t), share_corrected=fire_t.mean()))

    out_rows = []
    rng = np.random.default_rng(0)
    for r in rows:
        f = r["final"]
        fx, br, net = correction_counts(bt, f, yt)
        nets = []
        for _ in range(a.B):
            i = rng.integers(0, len(yt), len(yt))
            nets.append(correction_counts(bt[i], f[i], yt[i])[2])
        lo, hi = np.percentile(nets, [2.5, 97.5])
        acc = bootstrap_ci(accuracy, yt, f, a.B)
        f1 = bootstrap_ci(macro_f1, yt, f, a.B)
        mc = mcnemar(yt, f, cal)
        out_rows.append(dict(system=r["system"], arm=r["arm"], ablation=r["ablation"], accuracy=acc[0],
                             acc_low=acc[1], acc_high=acc[2], macro_f1=f1[0], f1_low=f1[1], f1_high=f1[2], fixed=fx,
                             broken=br, net=net, net_low=lo, net_high=hi, detector_auroc=r.get("detector_auroc"),
                             share_corrected=r.get("share_corrected"), threshold=r.get("threshold"),
                             vs_recal=f"{mc['n01']} / {mc['n10']}, p={mc['p_value']:.3g}"))
    t = pd.DataFrame(out_rows)
    t.to_csv(os.path.join(a.out, "results", "e4_test.csv"), index=False)
    with open(os.path.join(a.out, "results", "e4_test.md"), "w") as fh:
        fh.write("| System | Accuracy (95% CI) | Macro-F1 (95% CI) | Fixed | Broken | Net correction (95% CI) | "
                 "Detector AUROC | Share corrected | vs recalibration: n01 / n10, p |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in t.itertuples():
            da = "-" if pd.isna(r.detector_auroc) else f"{r.detector_auroc:.3f}"
            sc = "-" if pd.isna(r.share_corrected) else f"{r.share_corrected:.3f}"
            fh.write(f"| {r.system} | {r.accuracy:.4f} ({r.acc_low:.4f}-{r.acc_high:.4f}) | {r.macro_f1:.4f} "
                     f"({r.f1_low:.4f}-{r.f1_high:.4f}) | {r.fixed} | {r.broken} | {r.net:.4f} ({r.net_low:.4f}-"
                     f"{r.net_high:.4f}) | {da} | {sc} | {r.vs_recal} |\n")
    print(open(os.path.join(a.out, "results", "e4_test.md")).read())


if __name__ == "__main__":
    main()
