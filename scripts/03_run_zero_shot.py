"""Step 3 (sources A and V): run CLIP or LLaVA zero-shot over the chosen splits. Needs a GPU.

Resumable: images already in predictions/<source>_<split>.csv are skipped, so if a Kaggle session ends, run the
same command again. Usage:
  python scripts/03_run_zero_shot.py --out /kaggle/working/outputs --source A --splits test,val,train
  python scripts/03_run_zero_shot.py --out /kaggle/working/outputs --source V --splits test,val --batch 8
Optional for V: --rationales writes predictions/V_<split>_rationales.csv (slow; run on val/test or a subset).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from xfer.io import append_predictions, batches, load_manifest, pred_path, set_seed, todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["A", "V"], required=True)
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--batch", type=int, default=0, help="default: 64 for A, 8 for V")
    ap.add_argument("--limit", type=int, default=None, help="process at most N images per split (smoke test)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no_4bit", action="store_true", help="V only: load in float16 instead of 4-bit")
    ap.add_argument("--rationales", action="store_true", help="V only: also generate short text reasons")
    a = ap.parse_args()
    set_seed(42)
    m = load_manifest(a.out)

    if a.source == "A":
        from xfer.sources.clip_zs import ClipZeroShot
        model = ClipZeroShot(a.model or "openai/clip-vit-large-patch14")
        bs = a.batch or 64
    else:
        from xfer.sources.llava_zs import LlavaZeroShot
        model = LlavaZeroShot(a.model or "llava-hf/llava-1.5-7b-hf", load_4bit=not a.no_4bit)
        bs = a.batch or 8

    for split in a.splits.split(","):
        path = pred_path(a.out, a.source, split)
        sub = todo(m, split, path, a.limit)
        print(f"[{a.source}] {split}: {len(sub)} images to do")
        t0, n = time.time(), 0
        for b in batches(sub, bs):
            probs = model.predict_proba(b.path.tolist())
            append_predictions(path, b.image_id, b.label, probs)
            n += len(b)
            if n % (bs * 20) == 0 or n == len(sub):
                el = time.time() - t0
                print(f"  {n}/{len(sub)}  {el / max(n, 1):.2f}s per image")
        if a.source == "V" and a.rationales:
            rpath = path.replace(".csv", "_rationales.csv")
            seen = set(pd.read_csv(rpath).image_id.astype(str)) if os.path.exists(rpath) else set()
            rsub = m[(m.split == split) & ~m.image_id.astype(str).isin(seen)]
            if a.limit:
                rsub = rsub.head(a.limit)
            for b in batches(rsub, bs):
                txt = model.generate_rationales(b.path.tolist())
                pd.DataFrame(dict(image_id=b.image_id.values, rationale=txt)).to_csv(
                    rpath, mode="a", header=not os.path.exists(rpath), index=False)
            print(f"  rationales written to {rpath}")


if __name__ == "__main__":
    main()
