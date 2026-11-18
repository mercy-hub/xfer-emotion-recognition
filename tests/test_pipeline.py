"""CPU tests on a fake RAF-DB copy: data discovery (3 layouts), split rules, SVM source, metrics, E1 evaluation.

Run: python tests/test_pipeline.py
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

from xfer.metrics import mcnemar, bootstrap_ci, accuracy  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_all():
    tmp = tempfile.mkdtemp()
    counts = {}
    for layout in ("official", "folders", "csv"):
        raw, out = os.path.join(tmp, "raf_" + layout), os.path.join(tmp, "out_" + layout)
        run("tests/make_fake_raf.py", "--out", raw, "--layout", layout)
        run("scripts/01_prepare_data.py", "--root", raw, "--out", out)
        m = pd.read_csv(os.path.join(out, "manifest.csv"))
        counts[layout] = m.groupby("official_split").size().to_dict()
        # rules: test untouched, val only from official train, near-duplicate groups never straddle train/val
        assert set(m[m.official_split == "test"].split) == {"test"}
        assert set(m[m.official_split == "train"].split) <= {"train", "val"}
        g = m[m.split != "test"].groupby("group_id").split.nunique()
        assert (g == 1).all(), "a near-duplicate group was split across train and val"
        assert 0.1 < (m.split == "val").sum() / (m.official_split == "train").sum() < 0.3
    assert counts["official"] == counts["folders"] == counts["csv"], counts

    out = os.path.join(tmp, "out_official")
    run("scripts/02_run_svm.py", "--out", out, "--C_grid", "10")
    c = pd.read_csv(os.path.join(out, "predictions", "C_test.csv"))
    P = c.filter(like="p_").values
    assert np.allclose(P.sum(1), 1, atol=1e-6)
    assert (c.pred == c.label).mean() > 0.5, "SVM should learn the fake expressions"
    s = run("scripts/05_evaluate_e1.py", "--out", out, "--B", "100")
    assert "C: HOG/LBP + SVM" in s

    y = np.array([0, 1, 2, 3] * 10)
    assert mcnemar(y, y, y)["p_value"] == 1.0
    pt, lo, hi = bootstrap_ci(accuracy, y, y, B=50)
    assert pt == lo == hi == 1.0
    print("all tests passed")


if __name__ == "__main__":
    test_all()
