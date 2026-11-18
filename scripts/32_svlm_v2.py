"""Supervisor's matrix, cell 'strong VLM + SVLM compensator': Qwen2.5-VL-3B with LoRA corrects InternVL3-8B (V2).

Same recipe as step 14 (the Phase 1 small vision-language arm for LLaVA-1.5), with V2 as the base model:
  * trained on V2's failures among the 1,501 explained training images (prompt: image + V2's answer, 'image+pred');
  * runs only where the V2 error detector (the gradient-boosted detector of the interim compensator, scores shipped in
    bundle/det_V2_<split>.csv) is at least --floor; the gate threshold is then chosen on validation to maximise the
    net number of corrections, exactly as in step 15;
  * the test set is never used for any choice.
Writes predictions/SVLM2_<split>.csv (V2 after the compensator, all images) and results/svlm_v2.md.
Usage:  python scripts/32_svlm_v2.py --out OUT
"""
import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PIL import Image

from xfer.io import append_predictions, pred_path, read_predictions, set_seed
from xfer.labels import CLASSES, K, VLM_WORDS, prob_columns

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s14", os.path.join(HERE, "14_compensator_vlm.py"))
s14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s14)


def split_data(out, split, ids=None):
    m = pd.read_csv(os.path.join(out, "manifest.csv"))
    m = m[m.split == split].assign(image_id=lambda d: d.image_id.astype(str)).set_index("image_id")
    p = read_predictions(pred_path(out, "V2", split)); p["image_id"] = p.image_id.astype(str)
    p = p.drop_duplicates("image_id", keep="last").set_index("image_id")
    ids = np.asarray(sorted(ids) if ids is not None else sorted(p.index))
    P = np.clip(p.loc[ids, prob_columns()].values, 1e-6, 1)
    return dict(ids=ids, y=p.loc[ids].label.values.astype(int), P=P, pred=P.argmax(1), paths=m.loc[ids].path.values,
                shap=np.full((len(ids), 7), np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--abl", default="image+pred")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--floor", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    set_seed(42)
    bundle = os.path.join(os.path.dirname(HERE), "bundle")
    tr_ids = pd.read_csv(os.path.join(a.out, "descriptions", "V2_train_describe.csv")).image_id.astype(str).unique()
    D = {"train": split_data(a.out, "train", tr_ids), "val": split_data(a.out, "val"), "test": split_data(a.out, "test")}
    det = {}
    for s in ("val", "test"):
        t = pd.read_csv(os.path.join(bundle, f"det_V2_{s}.csv")); t["image_id"] = t.image_id.astype(str)
        det[s] = t.set_index("image_id").reindex(D[s]["ids"]).score.values
    fail = np.where(D["train"]["pred"] != D["train"]["y"])[0]
    vfail = np.where(D["val"]["pred"] != D["val"]["y"])[0]
    run_on = {s: np.where(det[s] >= a.floor)[0] for s in ("val", "test")}
    if a.limit:
        fail, vfail = fail[:a.limit], vfail[:a.limit]
        run_on = {s: v[:a.limit] for s, v in run_on.items()}
    img = lambda d, i: Image.open(d["paths"][i]).convert("RGB")
    print(f"training failures {len(fail)}; corrector runs on {len(run_on['val'])} validation and "
          f"{len(run_on['test'])} test images", flush=True)
    t0 = time.time()
    q = s14.Qwen(a.model)
    torch = q.torch
    params = [p for p in q.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    rng = np.random.default_rng(42)
    best, best_state = -1, None
    for ep in range(a.epochs):
        q.model.train()
        order = rng.permutation(fail); tot = 0.0
        for n, i in enumerate(order, 1):
            im = img(D["train"], i)
            if rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            loss = q.train_step(im, s14.prompt_text(D["train"], i, a.abl), VLM_WORDS[CLASSES[D["train"]["y"][i]]])
            if torch.isfinite(loss):
                (loss / a.accum).backward(); tot += loss.item()
            if n % a.accum == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        q.model.eval(); right = 0
        for j in range(0, len(vfail), a.batch):
            ii = vfail[j:j + a.batch]
            Pq = q.proba([img(D["val"], i) for i in ii], [s14.prompt_text(D["val"], i, a.abl) for i in ii])
            right += (Pq.argmax(1) == D["val"]["y"][ii]).sum()
        acc = right / max(len(vfail), 1)
        print(f"epoch {ep}: train loss {tot / max(len(order), 1):.4f}, validation-failure accuracy {acc:.4f}", flush=True)
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in q.model.state_dict().items() if "lora_" in k}
    q.model.load_state_dict(best_state, strict=False); q.model.eval()
    corr = {}
    for s in ("val", "test"):
        ii_all, probs = run_on[s], []
        for j in range(0, len(ii_all), a.batch):
            ii = ii_all[j:j + a.batch]
            probs.append(q.proba([img(D[s], i) for i in ii], [s14.prompt_text(D[s], i, a.abl) for i in ii]))
        corr[s] = np.concatenate(probs) if probs else np.zeros((0, K))
    # gate threshold on validation: maximise net corrections (fixed - broken), as in step 15
    v, y = D["val"], D["val"]["y"]
    cand = np.unique(np.r_[a.floor, det["val"][run_on["val"]]])
    def apply(s, thr):
        P = D[s]["P"].copy()
        on = det[s][run_on[s]] >= thr
        P[run_on[s][on]] = np.clip(corr[s][on], 1e-6, 1)
        return P / P.sum(1, keepdims=True)
    net = lambda P: int(((v["pred"] != y) & (P.argmax(1) == y)).sum() - ((v["pred"] == y) & (P.argmax(1) != y)).sum())
    thr = max(cand, key=lambda t: net(apply("val", t)))
    L = ["# Strong VLM (InternVL3-8B) + SVLM compensator (Qwen2.5-VL-3B, LoRA, image+pred)", "",
         f"Trained on {len(fail)} V2 failures; gate threshold {thr:.3f} chosen on validation; "
         f"{(time.time() - t0) / 60:.0f} min.", "", "| split | V2 alone | after SVLM | fired | fixed | broken |",
         "|---|---|---|---|---|---|"]
    for s in ("val", "test"):
        P = apply(s, thr); yy = D[s]["y"]; b = D[s]["pred"]; f = P.argmax(1)
        path = pred_path(a.out, "SVLM2", s)
        if os.path.exists(path):
            os.remove(path)
        append_predictions(path, D[s]["ids"], yy, P)
        L.append(f"| {s} | {100 * (b == yy).mean():.2f}% | {100 * (f == yy).mean():.2f}% | "
                 f"{100 * (b != f).mean():.1f}% | {int(((b != yy) & (f == yy)).sum())} | {int(((b == yy) & (f != yy)).sum())} |")
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    open(os.path.join(a.out, "results", "svlm_v2.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
