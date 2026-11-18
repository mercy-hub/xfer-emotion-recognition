"""Step 1: build the manifest, find near-duplicates, split train/val, and write dataset statistics.

Usage (Kaggle):
  python scripts/01_prepare_data.py --root /kaggle/input/<raf-db-dataset> --out /kaggle/working/outputs
Outputs in --out:
  manifest.csv            one row per image: image_id, path, official_split, split, label, group_id, hashes, size
  data_report.json        layout found, duplicate counts, split fractions
  class_counts.csv/.md    images per class and split (for section 3.3)
  cross_split_pairs.csv   near-duplicate pairs between the official train and test sets (reported, not removed)
  figures/class_counts.png, figures/contact_sheet.png (check the label mapping by eye)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from xfer.data import class_counts, discover, make_split
from xfer.labels import CLASSES


def contact_sheet(df, path, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.1, len(CLASSES) * 1.2))
    for r, c in enumerate(CLASSES):
        sub = df[(df.label_name == c) & (df.split != "test")]
        pick = sub.sample(min(per_class, len(sub)), random_state=int(rng.integers(1e6))) if len(sub) else sub
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(pick):
                ax.imshow(Image.open(pick.path.iloc[k]).convert("RGB"))
        axes[r, 0].set_title(c, fontsize=7, loc="left")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_chart(t, path):
    t = t.drop(index="total")
    splits = [s for s in ["train", "val", "test"] if s in t.columns]
    x = np.arange(len(t))
    w = 0.8 / len(splits)
    shades = ["0.15", "0.5", "0.8"]
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, s in enumerate(splits):
        ax.bar(x + i * w, t[s].values, w, label=s, color=shades[i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + w * (len(splits) - 1) / 2)
    ax.set_xticklabels(t.index, fontsize=8)
    ax.set_ylabel("Images")
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing the RAF-DB copy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_hamming", type=int, default=16, help="near-duplicate threshold on the 256-bit dHash")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)

    df, layout = discover(a.root)
    print(f"Layout: {layout}; images found: {len(df)}")
    df, report, cross = make_split(df, a.val_fraction, a.seed, a.max_hamming)
    report["layout"] = layout
    report["root"] = a.root
    df.to_csv(os.path.join(a.out, "manifest.csv"), index=False)

    t = class_counts(df)
    t.to_csv(os.path.join(a.out, "class_counts.csv"))
    with open(os.path.join(a.out, "class_counts.md"), "w") as fh:
        cols = ["class"] + list(t.columns)
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for idx, row in t.iterrows():
            fh.write("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |\n")
    report["image_sizes"] = df.groupby(["width", "height"]).size().sort_values(ascending=False).head(5).to_dict()
    report["image_sizes"] = {f"{k[0]}x{k[1]}": int(v) for k, v in report["image_sizes"].items()}
    with open(os.path.join(a.out, "data_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    if cross:
        import pandas as pd
        pd.DataFrame([(df.image_id[i], df.image_id[j]) for i, j in cross], columns=["image_a", "image_b"]) \
            .to_csv(os.path.join(a.out, "cross_split_pairs.csv"), index=False)

    bar_chart(t, os.path.join(a.out, "figures", "class_counts.png"))
    contact_sheet(df, os.path.join(a.out, "figures", "contact_sheet.png"))
    print(t)
    print(json.dumps(report, indent=2))
    print("Check figures/contact_sheet.png: each row should show the expression named on its left.")


if __name__ == "__main__":
    main()
