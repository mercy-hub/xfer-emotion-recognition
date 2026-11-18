"""CPU tests for stage 2 on a fake RAF-DB copy: calibration, region explanations (with the real SVM as the model),
merging, landmark geometry helpers and the surrogate tree. Run: python tests/test_stage2.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from xfer.explain import coalitions, kernel_shap  # noqa: E402
from xfer.io import append_predictions, pred_path  # noqa: E402
from xfer.labels import prob_columns  # noqa: E402
from xfer.regions import GEOMETRY, M, REGIONS, geometry, region_map, region_masks  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
    return r.stdout


def test_all():
    tmp = tempfile.mkdtemp()
    raw, out = os.path.join(tmp, "raf"), os.path.join(tmp, "out")
    run("tests/make_fake_raf.py", "--out", raw, "--layout", "folders")
    run("scripts/01_prepare_data.py", "--root", raw, "--out", out)
    run("scripts/02_run_svm.py", "--out", out, "--C_grid", "10")

    # a fake V that never answers neutral (index 6), like the real one in E1
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        c = pd.read_csv(pred_path(out, "C", split))
        P = c[prob_columns()].values.copy()
        P[:, 6] *= 0.05
        P = P / P.sum(1, keepdims=True)
        p = pred_path(out, "V", split)
        if os.path.exists(p):
            os.remove(p)
        append_predictions(p, c.image_id, c.label, P)
    s = run("scripts/08_calibrate.py", "--out", out, "--sources", "V,C")
    cal = pd.read_csv(os.path.join(out, "results", "calibration.csv"))
    v = cal[(cal.source == "V") & (cal.split == "test")].set_index("map")
    assert v.loc["vector", "macro_f1"] >= v.loc["none", "macro_f1"] - 1e-9, s
    run("scripts/05_evaluate_e1.py", "--out", out, "--sources", "V,Vcal,C", "--tag", "cal", "--B", "50")

    # region explanations with the real SVM, template regions (no landmarks), then merge
    run("scripts/09_region_explain.py", "--out", out, "--source", "C", "--splits", "test", "--limit", "6",
        "--chunk", "4", "--shard", "0", "--nshards", "2")
    run("scripts/09_region_explain.py", "--out", out, "--source", "C", "--splits", "test", "--limit", "6",
        "--chunk", "4", "--shard", "1", "--nshards", "2")
    # resume: a second call skips the 6 done and explains the next 6
    s = run("scripts/09_region_explain.py", "--out", out, "--source", "C", "--splits", "test", "--limit", "6",
            "--shard", "0", "--nshards", "2")
    assert "(6 already done)" in s, s
    run("scripts/09_region_explain.py", "--out", out, "--source", "C", "--splits", "test", "--merge")
    e = pd.read_csv(os.path.join(out, "explanations", "C_test.csv"))
    assert len(e) == 18 and e.image_id.is_unique and e.template.all()
    z = np.load(os.path.join(out, "explanations", "C_test.npz"), allow_pickle=True)
    P, Z, phi = z["P"].astype(float), z["Z"], z["phi"]
    for i in range(len(e)):  # efficiency: SHAP values add up to f(all) - f(none) for the target class
        t = z["target"][i]
        f1 = P[i][Z[i].sum(1) == M][0, t]
        f0 = P[i][Z[i].sum(1) == 0][0, t]
        assert abs(phi[i].sum() - (f1 - f0)) < 1e-3

    # geometry helpers on a synthetic landmark set
    lm = np.random.default_rng(1).uniform(10, 90, (478, 2))
    g = geometry(lm)
    assert set(g) == set(GEOMETRY) and all(np.isfinite(list(g.values())))
    lab = region_map(None)
    assert lab.max() == M - 1 and (lab >= 0).mean() > 0.5
    assert region_masks(lm).shape == (M, 100, 100)

    # surrogate tree on fake geometry, with and without V's SHAP values
    mf = pd.read_csv(os.path.join(out, "manifest.csv"))
    geo = pd.DataFrame({"image_id": mf.image_id, "split": mf.split, "label": mf.label, "found": True})
    for k in GEOMETRY:
        geo[k] = rng.normal(size=len(mf)) + mf.label * (0.3 if k == "mouth_opening" else 0)
    geo.to_csv(os.path.join(out, "results", "D_geometry.csv"), index=False)
    s = run("scripts/10_surrogate_tree.py", "--out", out, "--depths", "2,3")
    f = pd.read_csv(os.path.join(out, "results", "surrogate_fidelity.csv"))
    assert {"fidelity_val", "fidelity_test"} <= set(f.columns)
    for split in ("train", "val", "test"):          # random stand-ins for V's explanations to exercise --with_shap
        ids = mf[mf.split == split].image_id
        fake = pd.DataFrame({"image_id": ids, **{f"shap_{r}": rng.normal(size=len(ids)) for r in REGIONS}})
        fake.to_csv(os.path.join(out, "explanations", f"V_{split}.csv"), index=False)
    run("scripts/10_surrogate_tree.py", "--out", out, "--depths", "2", "--choose", "2", "--with_shap")
    print("stage-2 tests passed")


if __name__ == "__main__":
    test_all()
