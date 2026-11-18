"""Step 4 (source B): train POSTER++ on the train split, keep the epoch with the best validation macro-F1,
then predict train, val and test. Needs a GPU. Resumes from the last epoch if the session stopped.

Usage (Kaggle, GPU T4 x2):
  python scripts/04_train_poster.py --out /kaggle/working/outputs --work /kaggle/working/poster --epochs 60
  python scripts/04_train_poster.py --out /kaggle/working/outputs --work /kaggle/working/poster --predict_only
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xfer.io import append_predictions, load_manifest, pred_path, set_seed
from xfer.metrics import accuracy, macro_f1
from xfer.sources import poster


def run_eval(model, loader, torch, dev):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(dev)).float(), -1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True, help="folder for the repository, backbones and checkpoints")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=144)
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict_only", action="store_true")
    ap.add_argument("--tag", default="B", help="prediction file tag (B for the main model; e.g. B7 for seed 7)")
    ap.add_argument("--limit", type=int, default=None, help="use only N images per split (smoke test)")
    a = ap.parse_args()
    set_seed(a.seed)

    import torch
    from torch.utils.data import DataLoader
    poster.setup(os.path.join(a.work, "POSTER_V2"), os.path.join(a.work, "pretrain"))
    from data_preprocessing.sam import SAM

    m = load_manifest(a.out)
    if a.limit:
        m = m.groupby("split", group_keys=False).head(a.limit)
    ds = {s: poster.ManifestDataset(m[m.split == s], train=(s == "train")) for s in ("train", "val", "test")}
    eval_ds = {s: poster.ManifestDataset(m[m.split == s], train=False) for s in ("train", "val", "test")}
    mk = lambda d, sh: DataLoader(d, batch_size=a.batch, shuffle=sh, num_workers=a.workers, pin_memory=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = poster.build_model()
    model = torch.nn.DataParallel(model).to(dev)  # uses every visible GPU (2 on Kaggle T4 x2); plain wrapper on CPU
    ckpt_dir = os.path.join(a.work, f"checkpoints_seed{a.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    last, best = os.path.join(ckpt_dir, "last.pth"), os.path.join(ckpt_dir, "best.pth")
    log_path = os.path.join(ckpt_dir, "log.csv")

    if not a.predict_only:
        opt = SAM(model.parameters(), torch.optim.Adam, lr=a.lr, rho=0.05, adaptive=False, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
        crit = torch.nn.CrossEntropyLoss()
        start, best_f1 = 0, -1.0
        if os.path.exists(last):
            st = torch.load(last, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            start, best_f1 = st["epoch"] + 1, st["best_f1"]
            print(f"resumed at epoch {start}, best val macro-F1 so far {best_f1:.4f}")
        train_loader, val_loader = mk(ds["train"], True), mk(eval_ds["val"], False)
        for ep in range(start, a.epochs):
            model.train(); t0 = time.time(); tot = 0.0; n = 0
            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev)
                loss = crit(model(x), y)
                opt.zero_grad(); loss.backward(); opt.first_step(zero_grad=True)
                crit(model(x), y).backward(); opt.second_step(zero_grad=True)
                tot += loss.item() * len(y); n += len(y)
            sched.step()
            p, yv = run_eval(model, val_loader, torch, dev)
            acc, f1 = accuracy(yv, p.argmax(1)), macro_f1(yv, p.argmax(1))
            print(f"epoch {ep}: loss {tot / n:.4f}  val acc {acc:.4f}  val macro-F1 {f1:.4f}  {time.time() - t0:.0f}s")
            with open(log_path, "a", newline="") as fh:
                csv.writer(fh).writerow([ep, tot / n, acc, f1])
            if f1 > best_f1:
                best_f1 = f1
                torch.save(poster.strip_module(model.state_dict()), best)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep,
                            best_f1=best_f1), last)

    model.module.load_state_dict(torch.load(best, map_location="cpu", weights_only=False))
    for s in ("train", "val", "test"):
        p, y = run_eval(model, mk(eval_ds[s], False), torch, dev)
        path = pred_path(a.out, a.tag, s)
        if os.path.exists(path):
            os.remove(path)
        append_predictions(path, eval_ds[s].ids, y, p)
        print(f"{a.tag} {s}: acc {accuracy(y, p.argmax(1)):.4f}  macro-F1 {macro_f1(y, p.argmax(1)):.4f}")


if __name__ == "__main__":
    main()
