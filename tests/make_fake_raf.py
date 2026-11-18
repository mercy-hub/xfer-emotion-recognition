"""Create a small fake RAF-DB copy for testing the pipeline without the real data.

Layouts: official (list_patition_label.txt), folders (train/1..7) and csv. Images are 100x100 synthetic faces whose
mouth and brow shapes depend on the class, so a classifier can learn something; a few near-duplicates are planted.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw


def face(label, rng):
    im = Image.new("RGB", (100, 100), tuple(int(v) for v in rng.integers(90, 200, 3)))
    d = ImageDraw.Draw(im)
    ox, oy = (int(v) for v in rng.integers(-8, 9, 2))
    d.ellipse((15 + ox, 10 + oy, 85 + ox, 95 + oy), fill=(int(rng.integers(150, 230)),) * 3, outline=(0, 0, 0))
    brow = [0, -6, 6, 4, -4, 6, 0][label]
    mouth_open = [14, 8, 2, 4, 2, 3, 1][label]
    smile = [0, -2, -4, 8, -6, -5, 0][label]
    for x in (32, 58):
        d.line((x, 35 + brow // 2, x + 12, 35 - brow // 2), fill=(0, 0, 0), width=2)
        d.ellipse((x + 3, 42, x + 9, 48), fill=(0, 0, 0))
    d.arc((35, 65 - smile, 65, 75 + smile + mouth_open), 0, 180, fill=(0, 0, 0), width=2)
    a = np.asarray(im).astype(np.int16) + rng.integers(-12, 12, (100, 100, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["official", "folders", "csv"], default="official")
    ap.add_argument("--n_train", type=int, default=280)
    ap.add_argument("--n_test", type=int, default=70)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    items = []
    for split, n in (("train", a.n_train), ("test", a.n_test)):
        for i in range(1, n + 1):
            lab = int(rng.integers(0, 7))
            items.append((split, f"{split}_{i:05d}", lab))
    # plant near-duplicates: a few train images copied with tiny noise, one leaking into test
    dup_src = items[:5]
    imgs = {name: face(lab, rng) for _, name, lab in items}
    for k, (split, name, lab) in enumerate(dup_src):
        nn = f"train_{a.n_train + k + 1:05d}" if k < 4 else f"test_{a.n_test + 1:05d}"
        sp = "train" if k < 4 else "test"
        arr = np.asarray(imgs[name]).astype(np.int16) + rng.integers(-2, 3, (100, 100, 3))
        imgs[nn] = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        items.append((sp, nn, lab))

    if a.layout == "official":
        lab_dir = os.path.join(a.out, "basic", "EmoLabel"); img_dir = os.path.join(a.out, "basic", "Image", "aligned")
        os.makedirs(lab_dir, exist_ok=True); os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(lab_dir, "list_patition_label.txt"), "w") as fh:
            for split, name, lab in items:
                fh.write(f"{name}.jpg {lab + 1}\n")
                imgs[name].save(os.path.join(img_dir, f"{name}_aligned.jpg"), quality=95)
    elif a.layout == "folders":
        for split, name, lab in items:
            d = os.path.join(a.out, "DATASET", split, str(lab + 1)); os.makedirs(d, exist_ok=True)
            imgs[name].save(os.path.join(d, f"{name}.jpg"), quality=95)
    else:
        import csv
        for split in ("train", "test"):
            d = os.path.join(a.out, "images", split); os.makedirs(d, exist_ok=True)
            with open(os.path.join(a.out, f"{split}_labels.csv"), "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["image", "label"])
                for sp, name, lab in items:
                    if sp == split:
                        imgs[name].save(os.path.join(d, f"{name}.jpg"), quality=95)
                        w.writerow([f"{name}.jpg", lab + 1])
    print("fake RAF-DB written to", a.out, "with", len(items), "images")


if __name__ == "__main__":
    main()
