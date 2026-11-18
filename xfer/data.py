"""Discover a RAF-DB copy, build a manifest, find near-duplicates, and make the train/validation split.

Supported layouts (detected automatically):
  1. Official release: .../EmoLabel/list_patition_label.txt and .../Image/aligned/*_aligned.jpg
  2. Folder per class: .../train/<class>/*.jpg and .../test/<class>/*.jpg, where <class> is 1..7 or a name
  3. CSV files: a train*.csv and a test*.csv with an image column and a label column

The official test set is never split or altered. The official training set is split ~80/20 into
train and val, stratified by class and grouped so that near-duplicate images stay on one side.
"""
import hashlib
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from .labels import CLASSES, K, NAME_ALIASES, to_index

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------------------- discovery
def _is_img(name):
    return name.lower().endswith(IMG_EXT)


def _official_split_from_name(name):
    n = os.path.basename(name).lower()
    if n.startswith("train"):
        return "train"
    if n.startswith("test"):
        return "test"
    return None


def _find_official(root):
    for dirpath, _, files in os.walk(root):
        if "list_patition_label.txt" in files or "list_partition_label.txt" in files:
            fn = "list_patition_label.txt" if "list_patition_label.txt" in files else "list_partition_label.txt"
            return os.path.join(dirpath, fn)
    return None


def _from_official(root, label_file):
    # index all images under root by basename (aligned images end in _aligned.jpg)
    by_name = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if _is_img(f):
                by_name.setdefault(f, os.path.join(dirpath, f))
    rows = []
    with open(label_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, lab = parts[0], parts[1]
            stem = os.path.splitext(name)[0]
            path = by_name.get(stem + "_aligned.jpg") or by_name.get(name)
            if path is None:
                continue
            rows.append(dict(path=path, official_split=_official_split_from_name(name), raw_label=lab))
    return pd.DataFrame(rows), "official"


def _class_dir_label(dirname):
    d = dirname.strip().lower()
    if (d.isdigit() and 0 <= int(d) <= K) or d in NAME_ALIASES:
        return d
    return None


def _from_folders(root):
    rows = []
    for dirpath, dirnames, files in os.walk(root):
        lab = _class_dir_label(os.path.basename(dirpath))
        if lab is None or not any(_is_img(f) for f in files):
            continue
        split = None
        for part in reversed(os.path.relpath(dirpath, root).lower().split(os.sep)):
            if part.startswith("train"):
                split = "train"; break
            if part.startswith("test") or part in ("valid", "val", "validation"):
                split = "test"; break
        if split is None:
            continue
        for f in files:
            if _is_img(f):
                rows.append(dict(path=os.path.join(dirpath, f), official_split=split, raw_label=lab))
    return pd.DataFrame(rows), "folders"


def _from_csv(root):
    csvs = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".csv") and re.search(r"(train|test)", f.lower()):
                csvs.append(os.path.join(dirpath, f))
    if not csvs:
        return pd.DataFrame(), "csv"
    by_name = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if _is_img(f):
                by_name.setdefault(f, []).append(os.path.join(dirpath, f))
    rows = []
    for c in csvs:
        split = "train" if "train" in os.path.basename(c).lower() else "test"
        df = pd.read_csv(c)
        cols = {x.lower(): x for x in df.columns}
        icol = next((cols[x] for x in ("image", "img", "filename", "file", "path", "image_name") if x in cols), df.columns[0])
        lcol = next((cols[x] for x in ("label", "emotion", "class", "expression") if x in cols), df.columns[1])
        for _, r in df.iterrows():
            name = os.path.basename(str(r[icol]))
            cands = by_name.get(name, [])
            # prefer a file whose path mentions the same split
            path = next((p for p in cands if split in p.lower()), cands[0] if cands else None)
            if path:
                rows.append(dict(path=path, official_split=split, raw_label=r[lcol]))
    return pd.DataFrame(rows), "csv"


def discover(root):
    """Return (dataframe with path, official_split, label, label_name), layout name)."""
    lf = _find_official(root)
    if lf:
        df, layout = _from_official(root, lf)
    else:
        df, layout = _from_folders(root)
        if df.empty:
            df, layout = _from_csv(root)
    if df.empty:
        raise RuntimeError(f"No RAF-DB layout recognised under {root}. See xfer/data.py for supported layouts.")
    df = df[df.official_split.isin(["train", "test"])].copy()
    # labels: names, RAF ids 1..7, or 0-based 0..6
    raw = df.raw_label.astype(str).str.strip()
    numeric = raw.str.fullmatch(r"\d+").all()
    if numeric:
        vals = raw.astype(int)
        if vals.min() == 0 and vals.max() <= K - 1:
            vals = vals + 1  # 0-based copy -> RAF ids
        df["label"] = [to_index(v) for v in vals]
    else:
        df["label"] = [to_index(v) for v in raw]
    df["label_name"] = [CLASSES[i] for i in df.label]
    df = df.drop(columns=["raw_label"]).drop_duplicates("path").sort_values("path").reset_index(drop=True)
    df["image_id"] = [os.path.splitext(os.path.basename(p))[0] for p in df.path]
    if df.image_id.duplicated().any():  # same file name in two folders: make ids unique
        df["image_id"] = [f"{s}_{os.path.splitext(os.path.basename(p))[0]}" for s, p in zip(df.official_split, df.path)]
    return df, layout


# ----------------------------------------------------------------------------- hashing
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size=16):
    """Difference hash with size*size bits (256 by default): robust to resizing and recompression, and
    fine-grained enough that different aligned faces do not collide."""
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate_pairs(hashes, max_dist=16, blocks=None, nbits=256):
    """Pairs (i, j) whose nbits-bit hashes differ in at most max_dist bits.

    Pigeonhole: if two hashes differ in <= max_dist < blocks bits, they agree exactly on at least one
    of `blocks` bit-blocks, so only images sharing a block value need comparing.
    """
    blocks = blocks or max_dist + 1
    assert max_dist < blocks
    hashes = [int(h) for h in hashes]
    edges = [(i * nbits) // blocks for i in range(blocks + 1)]
    pairs = set()
    for b in range(blocks):
        lo, hi = edges[b], edges[b + 1]
        mask = ((1 << (hi - lo)) - 1) << lo
        buckets = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[h & mask].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    if bin(hashes[i] ^ hashes[j]).count("1") <= max_dist:
                        pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def groups_from_pairs(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


# ----------------------------------------------------------------------------- split
def make_split(df, val_fraction=0.2, seed=42, max_dist=16, max_group_share=0.01):
    """Add md5, dhash, size, group_id and split columns. Returns (df, report dict)."""
    from sklearn.model_selection import StratifiedGroupKFold
    df = df.copy()
    df["md5"] = [md5_file(p) for p in df.path]
    df["dhash"] = [dhash(p) for p in df.path]
    sizes = [Image.open(p).size for p in df.path]
    df["width"] = [s[0] for s in sizes]
    df["height"] = [s[1] for s in sizes]

    # Tighten the threshold if it chains many different faces into one group (a sign it is too loose).
    while True:
        pairs = near_duplicate_pairs(df.dhash.tolist(), max_dist=max_dist)
        groups = groups_from_pairs(len(df), pairs)
        largest = np.bincount(groups).max()
        if largest <= max(3, max_group_share * len(df)) or max_dist == 0:
            break
        print(f"[split] largest near-duplicate group has {largest} images at distance {max_dist}; tightening")
        max_dist //= 2
    df["group_id"] = groups
    df["dhash"] = [format(h, "x") for h in df.dhash]

    split = np.where(df.official_split == "test", "test", "train").astype(object)
    tr = np.where(df.official_split == "train")[0]
    n_splits = max(2, int(round(1 / val_fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(tr, df.label.values[tr], df.group_id.values[tr]))
    split[tr[val_idx]] = "val"
    df["split"] = split

    off = df.official_split.values
    cross = [(i, j) for i, j in pairs if off[i] != off[j]]
    exact_cross = df.groupby("md5").official_split.nunique()
    report = dict(
        n_images=len(df),
        near_duplicate_pairs=len(pairs),
        near_duplicate_pairs_across_official_train_test=len(cross),
        exact_duplicate_files_across_train_test=int((exact_cross > 1).sum()),
        groups_with_more_than_one_image=int((df.group_id.value_counts() > 1).sum()),
        val_fraction_actual=float((df.split == "val").sum() / max(1, len(tr))),
        seed=seed, max_hamming_distance=max_dist, hash_bits=256,
        largest_group=int(largest),
    )
    return df, report, cross


def class_counts(df):
    t = pd.crosstab(df.label_name, df.split).reindex(CLASSES).fillna(0).astype(int)
    t = t[[c for c in ["train", "val", "test"] if c in t.columns]]
    t.loc["total"] = t.sum()
    t["all"] = t.sum(axis=1)
    return t
