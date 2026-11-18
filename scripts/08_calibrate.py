"""Step 8: class-prior recalibration baseline (E4) and temperature scaling for the reasoner. CPU, minutes.

For each source, two maps from its log-probabilities z to new probabilities are fitted on the validation split by
minimising the negative log-likelihood (Guo et al., 2017):
  temperature:  softmax(z / T)                 - one parameter; keeps the ranking, so accuracy is unchanged
  vector:       softmax(w * z + b)             - one weight and one bias per class; can change the predicted class,
                                                 so it can undo a systematic class bias such as never answering neutral
The vector map applied to V is the "prior recalibration" baseline of E4: a compensator must beat it.
Writes predictions/<S>cal_<split>.csv (vector map) and predictions/<S>T_<split>.csv (temperature), and
results/calibration.csv with validation and test metrics before and after.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax

from xfer.io import append_predictions, pred_path, read_predictions
from xfer.labels import K, prob_columns
from xfer.metrics import accuracy, ece, macro_f1


def logp(df):
    return np.log(np.clip(df[prob_columns()].values, 1e-8, 1))


def nll(z, y):
    return -log_softmax(z, axis=1)[np.arange(len(y)), y].mean()


def fit_temperature(z, y):
    r = minimize(lambda t: nll(z / np.exp(t[0]), y), [0.0], method="L-BFGS-B")
    return dict(T=float(np.exp(r.x[0])))


def fit_vector(z, y, l2=1e-3):
    def f(p):
        w, b = p[:K], p[K:]
        return nll(z * w + b, y) + l2 * (np.sum((w - 1) ** 2) + np.sum(b ** 2))
    r = minimize(f, np.r_[np.ones(K), np.zeros(K)], method="L-BFGS-B")
    return dict(w=r.x[:K].tolist(), b=r.x[K:].tolist())


def apply(params, z):
    if "T" in params:
        return np.exp(log_softmax(z / params["T"], axis=1))
    return np.exp(log_softmax(z * np.array(params["w"]) + np.array(params["b"]), axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="V,A,B,C")
    a = ap.parse_args()
    rows, params = [], {}
    for s in a.sources.split(","):
        if not os.path.exists(pred_path(a.out, s, "val")):
            print(f"{s}: no validation predictions, skipped"); continue
        val = read_predictions(pred_path(a.out, s, "val"))
        zv, yv = logp(val), val.label.values.astype(int)
        params[s] = dict(temperature=fit_temperature(zv, yv), vector=fit_vector(zv, yv))
        for split in ("train", "val", "test"):
            p = pred_path(a.out, s, split)
            if not os.path.exists(p):
                continue
            df = read_predictions(p)
            z, y = logp(df), df.label.values.astype(int)
            for kind, tag in (("vector", "cal"), ("temperature", "T")):
                P = apply(params[s][kind], z)
                out = pred_path(a.out, s + tag, split)
                if os.path.exists(out):
                    os.remove(out)
                append_predictions(out, df.image_id, y, P)
                if split in ("val", "test"):
                    rows.append(dict(source=s, map=kind, split=split, accuracy=accuracy(y, P.argmax(1)),
                                     macro_f1=macro_f1(y, P.argmax(1)), ece=ece(y, P)))
            if split in ("val", "test"):
                P0 = df[prob_columns()].values
                rows.append(dict(source=s, map="none", split=split, accuracy=accuracy(y, P0.argmax(1)),
                                 macro_f1=macro_f1(y, P0.argmax(1)), ece=ece(y, P0)))
    t = pd.DataFrame(rows).sort_values(["source", "split", "map"])
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
    t.to_csv(os.path.join(a.out, "results", "calibration.csv"), index=False)
    with open(os.path.join(a.out, "models", "calibration.json"), "w") as fh:
        json.dump(params, fh, indent=2)
    print(t.round(4).to_string(index=False))
    print("Validation numbers are fitted on validation (optimistic); judge the maps on the test rows.")


if __name__ == "__main__":
    main()
