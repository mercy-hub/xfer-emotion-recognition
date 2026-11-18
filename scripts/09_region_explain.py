"""Step 9: region explanations (Kernel SHAP and LIME over 7 facial regions) for any source.

Every image costs 32 model calls (xfer/explain.py). Measured on a T4: about 15 s per image for V (LLaVA-1.5), well
under 1 s for A, B and C. V is slow, so it is split into shards that run on the two GPUs at the same time:
  CUDA_VISIBLE_DEVICES=0 python scripts/09_region_explain.py --out OUT --source V --splits test --shard 0 --nshards 2 &
  CUDA_VISIBLE_DEVICES=1 python scripts/09_region_explain.py --out OUT --source V --splits test --shard 1 --nshards 2 &
Results are written every --chunk images, so a run that stops resumes where it left off. When all shards are done:
  python scripts/09_region_explain.py --out OUT --source V --splits test --merge
gives explanations/V_test.csv (shap_<region>, lime_<region>, spearman per image) and explanations/V_test.npz.
For the training split only a stratified subset is explained (--train_subset, default 2000 images).
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

from xfer.explain import N_SAMPLES, explain_image
from xfer.io import load_manifest, set_seed
from xfer.regions import REGIONS, region_masks


def select(m, split, train_subset):
    sub = m[m.split == split]
    if split == "train" and train_subset and train_subset < len(sub):
        frac = train_subset / len(sub)
        sub = sub.groupby("label", group_keys=False).apply(
            lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=0))
    return sub.sort_values("image_id").reset_index(drop=True)


def load_landmarks(out):
    f = os.path.join(out, "features", "landmarks.npz")
    if not os.path.exists(f):
        print("[explain] no landmarks yet: using the template regions for every image")
        return {}
    z = np.load(f, allow_pickle=True)
    return {str(i): (lm if found else None) for i, lm, found in zip(z["image_id"], z["landmarks"], z["found"])}


def make_predictor(source, a):
    if source == "V":
        from xfer.sources.llava_zs import LlavaZeroShot
        model = LlavaZeroShot()
    elif source == "A":
        from xfer.sources.clip_zs import ClipZeroShot
        model = ClipZeroShot()
    elif source == "B":
        from xfer.sources.poster import PosterPredictor
        ckpt = a.ckpt or (glob.glob(os.path.join(a.work, "checkpoints_seed42", "best.pth")) +
                          glob.glob("/kaggle/**/checkpoints_seed42/best.pth", recursive=True))[0]
        model = PosterPredictor(a.work, ckpt)
    elif source == "C":
        from xfer.sources.hog_svm import SvmPredictor
        model = SvmPredictor(os.path.join(a.out, "models", "svm_C.joblib"))
    else:
        raise ValueError(source)

    def predict(imgs):
        out = []
        for i in range(0, len(imgs), a.batch):
            out.append(model.predict_proba_images(imgs[i:i + a.batch]))
        return np.concatenate(out)
    return predict


def shard_dir(out, source, split, shard, nshards):
    return os.path.join(out, "explanations", f"{source}_{split}_s{shard}of{nshards}")


def done_in(d):
    ids = set()
    for f in glob.glob(os.path.join(d, "chunk_*.npz")):
        ids |= set(np.load(f, allow_pickle=True)["image_id"].astype(str))
    return ids


def save_chunk(d, recs):
    os.makedirs(d, exist_ok=True)
    k = len(glob.glob(os.path.join(d, "chunk_*.npz")))
    np.savez_compressed(os.path.join(d, f"chunk_{k:05d}.npz"),
                        image_id=np.array([r["image_id"] for r in recs]),
                        target=np.array([r["target"] for r in recs]),
                        phi=np.stack([r["phi"] for r in recs]), lime=np.stack([r["lime_coef"] for r in recs]),
                        lime_intercept=np.array([r["lime_intercept"] for r in recs]),
                        spearman=np.array([r["spearman"] for r in recs]),
                        Z=np.stack([r["Z"] for r in recs]).astype(np.int8),
                        P=np.stack([r["P"] for r in recs]).astype(np.float16),
                        template=np.array([r["template"] for r in recs]))


def merge(out, source, split):
    files = sorted(glob.glob(os.path.join(out, "explanations", f"{source}_{split}_s*of*", "chunk_*.npz")))
    if not files:
        print(f"nothing to merge for {source} {split}"); return
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ids = cat["image_id"].astype(str)
    _, first = np.unique(ids, return_index=True)
    cat = {k: v[np.sort(first)] for k, v in cat.items()}
    np.savez_compressed(os.path.join(out, "explanations", f"{source}_{split}.npz"), **cat)
    df = pd.DataFrame(dict(image_id=cat["image_id"].astype(str), target=cat["target"],
                           spearman=cat["spearman"], template=cat["template"]))
    for j, r in enumerate(REGIONS):
        df[f"shap_{r}"] = cat["phi"][:, j]
    for j, r in enumerate(REGIONS):
        df[f"lime_{r}"] = cat["lime"][:, j]
    df.to_csv(os.path.join(out, "explanations", f"{source}_{split}.csv"), index=False)
    print(f"merged {len(df)} images -> explanations/{source}_{split}.csv; "
          f"median SHAP-LIME rank agreement {np.nanmedian(df.spearman):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["V", "A", "B", "C"], required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--train_subset", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--work", default="/kaggle/working/poster", help="B only: POSTER++ folder")
    ap.add_argument("--ckpt", default=None, help="B only: trained POSTER++ weights")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        for split in a.splits.split(","):
            merge(a.out, a.source, split)
        return
    set_seed(42)
    m = load_manifest(a.out)
    lms = load_landmarks(a.out)
    predict = make_predictor(a.source, a)
    for split in a.splits.split(","):
        sub = select(m, split, a.train_subset)
        sub = sub.iloc[a.shard::a.nshards]
        d = shard_dir(a.out, a.source, split, a.shard, a.nshards)
        done = done_in(d)
        sub = sub[~sub.image_id.astype(str).isin(done)]
        if a.limit:
            sub = sub.head(a.limit)
        print(f"[{a.source} shard {a.shard}/{a.nshards}] {split}: {len(sub)} images to explain "
              f"({len(done)} already done), {N_SAMPLES} model calls each")
        recs, t0 = [], time.time()
        for n, (iid, path) in enumerate(zip(sub.image_id.astype(str), sub.path), 1):
            img = Image.open(path).convert("RGB")
            lm = lms.get(iid)
            masks = region_masks(lm, size=img.size[0])
            r = explain_image(img, masks, predict, iid)
            r["image_id"], r["template"] = iid, lm is None
            recs.append(r)
            if len(recs) == a.chunk or n == len(sub):
                save_chunk(d, recs); recs = []
                el = time.time() - t0
                print(f"  {n}/{len(sub)}  {el / n:.2f}s per image  ~{el / n * (len(sub) - n) / 3600:.1f} h left",
                      flush=True)


if __name__ == "__main__":
    main()
