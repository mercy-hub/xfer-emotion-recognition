"""Step 2 of the new design: the explanation shared by every expert (region occlusion), with faithfulness curves.

For each image and each of the 10 fine facial regions (xfer/regions.py FINE), the region is replaced by a blurred
copy and the drop in the probability of EVERY candidate emotion is recorded (class-conditional by construction).
Then, for the predicted class, the regions are ranked by that drop and two faithfulness curves are measured:
  deletion   remove the top-1 .. top-k regions together: a faithful ranking makes the probability fall fast;
  insertion  start from the fully blurred face and restore the top-1 .. top-k regions: it should rise fast.
Calls per image: 1 (full) + 10 (one region removed) + 1 (all removed) + (k-1) deletion + k insertion = 17 for k=3.

Output (resumable chunks, merged with --merge):
  explanations/occ_<source>_<split>.npz  image_id, P_full (7), P_drop (10, 7), P_none (7), order (10),
                                         del_P (k, 7), ins_P (k, 7), template
  explanations/occ_<source>_<split>.csv  per image: pred, occ_<region> (drop for the predicted class), deletion and
                                         insertion areas, and the fine region with the largest drop
Usage (V2 is slow, so split it over both GPUs; A, B, C take minutes):
  CUDA_VISIBLE_DEVICES=0 python scripts/20_occlusion.py --out OUT --source V2 --splits val,test --shard 0 --nshards 2
  CUDA_VISIBLE_DEVICES=1 python scripts/20_occlusion.py --out OUT --source V2 --splits val,test --shard 1 --nshards 2
  python scripts/20_occlusion.py --out OUT --source V2 --splits val,test --merge
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PIL import Image

from xfer.explain import perturb
from xfer.io import load_manifest, set_seed
from xfer.labels import CLASSES
from xfer.predictors import make_predictor
from xfer.regions import FINE, fine_region_masks

R = len(FINE)


def occlusion_image(img, masks, predict, k=3):
    """All model calls for one image. Returns a dict of arrays (see the module docstring)."""
    full = np.ones(R, int)
    Z = [full, np.zeros(R, int)] + [full - np.eye(R, dtype=int)[r] for r in range(R)]
    P = predict(perturb(img, masks, np.stack(Z)))
    P_full, P_none, P_drop = P[0], P[1], P[2:]
    target = int(P_full.argmax())
    order = np.argsort(-(P_full[target] - P_drop[:, target]), kind="stable")    # largest drop first
    Zd = [full.copy() for _ in range(2, k + 1)]                                   # top-1 removal is already in P_drop
    for j, z in enumerate(Zd, start=2):
        z[order[:j]] = 0
    Zi = []
    for j in range(1, k + 1):
        z = np.zeros(R, int); z[order[:j]] = 1; Zi.append(z)
    P2 = predict(perturb(img, masks, np.stack(Zd + Zi)))
    del_P = np.vstack([P_drop[order[0]][None], P2[:k - 1]])
    ins_P = P2[k - 1:]
    return dict(P_full=P_full, P_none=P_none, P_drop=P_drop, order=order, del_P=del_P, ins_P=ins_P)


def curve_areas(P_full, P_none, del_P, ins_P, target):
    """Normalised areas over steps 0..k (trapezoid): deletion (lower = more faithful), insertion (higher = better)."""
    d = np.r_[P_full[target], del_P[:, target]]
    i = np.r_[P_none[target], ins_P[:, target]]
    trap = lambda y: float(np.trapezoid(y, dx=1.0 / (len(y) - 1))) if hasattr(np, "trapezoid") else \
        float(np.trapz(y, dx=1.0 / (len(y) - 1)))
    return trap(d), trap(i)


def shard_dir(out, source, split, shard, nshards):
    return os.path.join(out, "explanations", f"occ_{source}_{split}_s{shard}of{nshards}")


def done_in(d):
    ids = set()
    for f in glob.glob(os.path.join(d, "chunk_*.npz")):
        ids |= set(np.load(f, allow_pickle=True)["image_id"].astype(str))
    return ids


def save_chunk(d, recs):
    os.makedirs(d, exist_ok=True)
    k = len(glob.glob(os.path.join(d, "chunk_*.npz")))
    keys = ["P_full", "P_none", "P_drop", "order", "del_P", "ins_P"]
    np.savez_compressed(os.path.join(d, f"chunk_{k:05d}.npz"), image_id=np.array([r["image_id"] for r in recs]),
                        template=np.array([r["template"] for r in recs]),
                        **{key: np.stack([r[key] for r in recs]) for key in keys})


def merge(out, source, split):
    files = sorted(glob.glob(os.path.join(out, "explanations", f"occ_{source}_{split}_s*of*", "chunk_*.npz")))
    if not files:
        print(f"nothing to merge for {source} {split}"); return None
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    ids = cat["image_id"].astype(str)
    _, first = np.unique(ids, return_index=True)
    cat = {key: v[np.sort(first)] for key, v in cat.items()}
    np.savez_compressed(os.path.join(out, "explanations", f"occ_{source}_{split}.npz"), **cat)
    rows = []
    for j, iid in enumerate(cat["image_id"].astype(str)):
        t = int(cat["P_full"][j].argmax())
        drop = cat["P_full"][j, t] - cat["P_drop"][j, :, t]
        dA, iA = curve_areas(cat["P_full"][j], cat["P_none"][j], cat["del_P"][j], cat["ins_P"][j], t)
        rows.append(dict(image_id=iid, pred=t, p_pred=float(cat["P_full"][j, t]), template=bool(cat["template"][j]),
                         top_region=FINE[int(np.argmax(drop))], deletion_area=dA, insertion_area=iA,
                         **{f"occ_{r}": float(v) for r, v in zip(FINE, drop)}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "explanations", f"occ_{source}_{split}.csv"), index=False)
    print(f"merged {len(df)} images -> explanations/occ_{source}_{split}.csv | mean deletion area "
          f"{df.deletion_area.mean():.3f}, mean insertion area {df.insertion_area.mean():.3f} | most decisive "
          f"region: {df.top_region.value_counts().head(3).to_dict()}")
    return df


def load_landmarks(out):
    f = os.path.join(out, "features", "landmarks.npz")
    if not os.path.exists(f):
        print("[occlusion] no landmarks: template regions for every image")
        return {}
    z = np.load(f, allow_pickle=True)
    return {str(i): (lm if found else None) for i, lm, found in zip(z["image_id"], z["landmarks"], z["found"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["V2", "V", "A", "B", "C"], required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--k", type=int, default=3, help="steps of the deletion and insertion curves")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--work", default="/kaggle/working/poster", help="B only: POSTER++ folder")
    ap.add_argument("--ckpt", default=None, help="B only: trained POSTER++ weights")
    ap.add_argument("--sample", type=int, default=None,
                    help="explain a class-stratified random sample of N images per split (fixed seed), not all")
    ap.add_argument("--ids_from", default=None, help="CSV with image_id: only these images (e.g. a train subset)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        for split in a.splits.split(","):
            merge(a.out, a.source, split)
        return
    set_seed(42)
    m = load_manifest(a.out)
    if a.ids_from:
        m = m[m.image_id.astype(str).isin(set(pd.read_csv(a.ids_from, usecols=["image_id"]).image_id.astype(str)))]
    lms = load_landmarks(a.out)
    predict = make_predictor(a.source, a.out, a.work, a.ckpt, a.batch)
    for split in a.splits.split(","):
        sub = m[m.split == split].sort_values("image_id")
        if a.sample and a.sample < len(sub):
            frac = a.sample / len(sub)
            sub = sub.groupby("label", group_keys=False).apply(
                lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=0)).sort_values("image_id")
        sub = sub.iloc[a.shard::a.nshards]
        d = shard_dir(a.out, a.source, split, a.shard, a.nshards)
        sub = sub[~sub.image_id.astype(str).isin(done_in(d))]
        if a.limit:
            sub = sub.head(a.limit)
        print(f"[occ {a.source} shard {a.shard}/{a.nshards}] {split}: {len(sub)} images, "
              f"{R + 2 + 2 * a.k - 1} model calls each", flush=True)
        recs, t0 = [], time.time()
        for n, (iid, path) in enumerate(zip(sub.image_id.astype(str), sub.path), 1):
            img = Image.open(path).convert("RGB")
            lm = lms.get(iid)
            r = occlusion_image(img, fine_region_masks(lm, size=img.size[0]), predict, a.k)
            r["image_id"], r["template"] = iid, lm is None
            recs.append(r)
            if len(recs) == a.chunk or n == len(sub):
                save_chunk(d, recs); recs = []
                el = time.time() - t0
                print(f"  {n}/{len(sub)}  {el / n:.2f}s per image  ~{el / n * (len(sub) - n) / 3600:.1f} h left",
                      flush=True)


if __name__ == "__main__":
    main()
