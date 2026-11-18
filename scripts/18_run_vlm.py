"""Step 1 of the new design: run the frozen VLM (default InternVL3-8B) over the chosen splits. Needs a GPU.

Tasks (all resumable: images already written are skipped, so re-run the same command after a session ends):
  proba     predictions/<tag>_<split>.csv           class probabilities (same format as every other source)
  describe  descriptions/<tag>_<split>_describe.csv structured FACS-style description, emotion not given
  given     descriptions/<tag>_<split>_given.csv    description when the TRUE emotion is given in the prompt;
                                                    allowed on train and val only, refused on test (leakage)
Usage:
  python scripts/18_run_vlm.py --out OUT --task proba --splits val,test
  python scripts/18_run_vlm.py --out OUT --task describe --splits val,test
  python scripts/18_run_vlm.py --out OUT --task given --splits train --ids_from OUT/explanations/V_train.csv
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from xfer.io import append_predictions, batches, done_ids, load_manifest, pred_path, set_seed
from xfer.sources.vlm_zs import DEFAULT_MODEL, VLMZeroShot, parse_reply


def desc_path(out, tag, split, task):
    d = os.path.join(out, "descriptions")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{tag}_{split}_{task}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=["proba", "describe", "given"], required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--tag", default="V2", help="file prefix for this VLM (V is the old LLaVA run)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="at most N images per split (smoke test)")
    ap.add_argument("--ids_from", default=None, help="CSV with an image_id column: only these images")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--answer_prefix", default="", help='e.g. "<think>\\n</think>\\n<answer>" for UniFER-7B')
    ap.add_argument("--forms", action="store_true", help="score several surface forms per class (Happy, happy, ...)")
    a = ap.parse_args()
    set_seed(42)
    if a.task == "given" and "test" in a.splits.split(","):
        sys.exit("The true emotion must never be given to the VLM on test images.")

    m = load_manifest(a.out)
    if a.ids_from:
        keep = set(pd.read_csv(a.ids_from, usecols=["image_id"]).image_id.astype(str))
        m = m[m.image_id.astype(str).isin(keep)]
    model = VLMZeroShot(a.model, load_4bit=not a.no_4bit,
                        answer_prefix=a.answer_prefix.encode().decode("unicode_escape"), forms=a.forms)

    for split in a.splits.split(","):
        path = pred_path(a.out, a.tag, split) if a.task == "proba" else desc_path(a.out, a.tag, split, a.task)
        sub = m[m.split == split]
        sub = sub[~sub.image_id.astype(str).isin(done_ids(path))]
        if a.limit:
            sub = sub.head(a.limit)
        print(f"[{a.tag}:{a.task}] {split}: {len(sub)} images to do -> {path}", flush=True)
        t0, n = time.time(), 0
        for b in batches(sub, a.batch):
            paths = b.path.tolist()
            if a.task == "proba":
                probs = model.predict_proba(paths)
                if not np.isfinite(probs).all():
                    sys.exit("The VLM returned NaN probabilities (float16 overflow on the T4). Stop here and send "
                             "this log; the fallback is --model Qwen/Qwen2.5-VL-7B-Instruct.")
                append_predictions(path, b.image_id, b.label, probs)
            else:
                texts = model.describe(paths) if a.task == "describe" else model.describe_given(paths, b.label)
                rows = []
                for iid, lab, t in zip(b.image_id, b.label, texts):
                    r = parse_reply(t)
                    rows.append(dict(image_id=iid, label=lab, text=t.replace("\n", " | "),
                                     **{k: v for k, v in r.items()}))
                df = pd.DataFrame(rows)
                df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            n += len(b)
            if n % (a.batch * 25) == 0 or n == len(sub):
                print(f"  {n}/{len(sub)}  {(time.time() - t0) / max(n, 1):.2f}s per image", flush=True)


if __name__ == "__main__":
    main()
