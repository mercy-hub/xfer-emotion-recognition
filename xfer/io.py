"""Shared input/output: the manifest and the common prediction file format.

Every source writes predictions/<source>_<split>.csv with columns:
  image_id, label, p_surprise, ..., p_neutral, pred
Files are appended batch by batch, so a run that stops (Kaggle session limit) resumes where it left off.
"""
import os
import random

import numpy as np
import pandas as pd

from .labels import CLASSES, prob_columns


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_manifest(out_dir):
    return pd.read_csv(os.path.join(out_dir, "manifest.csv"))


def pred_path(out_dir, source, split):
    d = os.path.join(out_dir, "predictions")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{source}_{split}.csv")


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, usecols=["image_id"]).image_id.astype(str))


def append_predictions(path, image_ids, labels, probs, extra=None):
    probs = np.asarray(probs, dtype=float)
    df = pd.DataFrame(probs, columns=prob_columns())
    df.insert(0, "label", list(labels))
    df.insert(0, "image_id", list(image_ids))
    df["pred"] = probs.argmax(1)
    if extra:
        for k, v in extra.items():
            df[k] = list(v)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def read_predictions(path):
    df = pd.read_csv(path).drop_duplicates("image_id", keep="last")
    return df


def todo(manifest, split, path, limit=None):
    sub = manifest[manifest.split == split]
    seen = done_ids(path)
    sub = sub[~sub.image_id.astype(str).isin(seen)]
    if limit:
        sub = sub.head(limit)
    return sub


def batches(df, n):
    for i in range(0, len(df), n):
        yield df.iloc[i:i + n]
