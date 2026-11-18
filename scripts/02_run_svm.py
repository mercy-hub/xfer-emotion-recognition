"""Step 2 (source C): train the HOG/LBP + SVM on the train split, choose C on val, predict val and test.

Runs on CPU (Kaggle CPU session is enough). Usage:
  python scripts/02_run_svm.py --out /kaggle/working/outputs
Writes predictions/C_val.csv, predictions/C_test.csv (and C_train.csv for the compensator later),
models/svm_C.joblib and features/C_<split>.npy.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np

from xfer.io import append_predictions, load_manifest, pred_path, set_seed
from xfer.metrics import accuracy, macro_f1
from xfer.sources.hog_svm import build_svm, feature_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--C_grid", default="1,3,10,30")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    m = load_manifest(a.out)
    os.makedirs(os.path.join(a.out, "features"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)

    X = {}
    for split in ("train", "val", "test"):
        f = os.path.join(a.out, "features", f"C_{split}.npy")
        if os.path.exists(f):
            X[split] = np.load(f)
        else:
            t = time.time()
            X[split] = feature_matrix(m[m.split == split].path.tolist())
            np.save(f, X[split])
            print(f"features {split}: {X[split].shape} in {time.time() - t:.0f}s")
    y = {s: m[m.split == s].label.values for s in X}

    results = {}
    best = None
    for C in [float(c) for c in a.C_grid.split(",")]:
        t = time.time()
        clf = build_svm(C=C, seed=a.seed).fit(X["train"], y["train"])
        pv = clf.predict_proba(X["val"]).argmax(1)
        results[C] = dict(val_acc=accuracy(y["val"], pv), val_macro_f1=macro_f1(y["val"], pv), secs=time.time() - t)
        print(f"C={C}: val acc {results[C]['val_acc']:.4f}, macro-F1 {results[C]['val_macro_f1']:.4f}")
        if best is None or results[C]["val_macro_f1"] > results[best[0]]["val_macro_f1"]:
            best = (C, clf)
    C, clf = best
    print(f"chosen C={C} (by validation macro-F1)")
    joblib.dump(clf, os.path.join(a.out, "models", "svm_C.joblib"))
    with open(os.path.join(a.out, "models", "svm_C_selection.json"), "w") as fh:
        json.dump(dict(chosen_C=C, grid={str(k): v for k, v in results.items()}), fh, indent=2)

    for split in ("train", "val", "test"):
        path = pred_path(a.out, "C", split)
        if os.path.exists(path):
            os.remove(path)
        sub = m[m.split == split]
        append_predictions(path, sub.image_id, sub.label, clf.predict_proba(X[split]))
    print("done: predictions/C_{train,val,test}.csv")


if __name__ == "__main__":
    main()
