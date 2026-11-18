"""CPU smoke test for E4 on a fake RAF-DB copy: CPU arm (step 12), vision arm (step 13, tiny untrained backbone),
shared evaluation (step 15), and the small-VLM prompt builder. Run: python tests/test_stage4.py"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xfer.io import append_predictions, pred_path  # noqa: E402
from xfer.labels import prob_columns  # noqa: E402
from xfer.regions import GEOMETRY, REGIONS  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def test_all():
    tmp = tempfile.mkdtemp()
    raw, out = os.path.join(tmp, "raf"), os.path.join(tmp, "out")
    run("tests/make_fake_raf.py", "--out", raw, "--layout", "folders", "--n_train", "500", "--n_test", "120")
    run("scripts/01_prepare_data.py", "--root", raw, "--out", out)
    run("scripts/02_run_svm.py", "--out", out, "--C_grid", "10")
    rng = np.random.default_rng(0)
    m = pd.read_csv(os.path.join(out, "manifest.csv"))
    os.makedirs(os.path.join(out, "explanations"), exist_ok=True)
    for split in ("train", "val", "test"):
        c = pd.read_csv(pred_path(out, "C", split))
        P = c[prob_columns()].values + rng.random((len(c), 7)) * 0.8
        P[:, 6] *= 0.05
        P /= P.sum(1, keepdims=True)
        for tag in ("VT", "Vcal"):
            if os.path.exists(pred_path(out, tag, split)):
                os.remove(pred_path(out, tag, split))
            append_predictions(pred_path(out, tag, split), c.image_id, c.label, P)
        e = pd.DataFrame({"image_id": c.image_id, "target": P.argmax(1), "spearman": 0.8, "template": True,
                          **{f"shap_{r}": rng.normal(0, .1, len(c)) for r in REGIONS},
                          **{f"lime_{r}": rng.normal(0, .1, len(c)) for r in REGIONS}})
        e.to_csv(os.path.join(out, "explanations", f"V_{split}.csv"), index=False)
    geo = pd.DataFrame({"image_id": m.image_id, "split": m.split, "label": m.label, "found": True})
    for k in GEOMETRY + ["browInnerUp", "mouthSmileLeft", "mouthSmileRight", "jawOpen"]:
        geo[k] = rng.random(len(m))
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    geo.to_csv(os.path.join(out, "results", "D_geometry.csv"), index=False)

    s = run("scripts/12_compensator_tabular.py", "--out", out)
    assert "Compensator (pred+expl+face)" in s
    run("scripts/13_compensator_vision.py", "--out", out, "--backbone", "resnet18", "--no_pretrained",
        "--epochs", "1", "--batch", "8", "--limit", "16", "--floor", "0.0", "--workers", "0")
    s = run("scripts/15_evaluate_e4.py", "--out", out, "--B", "50")
    assert "cnn corrector, image+pred+attr" in s and "tab corrector, pred" in s, s

    sys.argv = ["x"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("vlm", os.path.join(ROOT, "scripts", "14_compensator_vlm.py"))
    vlm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vlm)
    from xfer.compensator import load_split
    d = load_split(out, "val")
    for abl in vlm.ABLATIONS:
        t = vlm.prompt_text(d, 0, abl)
        assert t.endswith(vlm.QUESTION) and (("first model" in t) == ("pred" in abl))
    print(vlm.prompt_text(d, 0, "image+pred+expl"))
    print("stage-4 tests passed")


if __name__ == "__main__":
    test_all()
