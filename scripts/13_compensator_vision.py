"""Step 13 (E4, vision-backbone arm C): a pretrained CNN corrects V on the images where it fails. One GPU.

Trained on V's failures among the explained training images (the same images as every other arm), validated on V's
failures in the validation split (epoch with the best accuracy is kept), then run on every validation and test
image whose shared detector score is at least --floor (the gate threshold chosen later is never lower than this).
Ablations of the corrector input:
  image             the face only
  image+pred        the face, plus V's probabilities as a vector joined to the image features
  image+pred+attr   the same, plus a fourth image channel holding V's region SHAP values painted on the regions
Writes predictions/corr-cnn-<ablation>_<split>.csv for step 15.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PIL import Image

from xfer.compensator import load_split
from xfer.io import append_predictions, pred_path, set_seed
from xfer.labels import K
from xfer.regions import region_masks

ABLATIONS = ["image", "image+pred", "image+pred+attr"]
SIZE = 224


def landmark_lookup(out):
    f = os.path.join(out, "features", "landmarks.npz")
    if not os.path.exists(f):
        return {}
    z = np.load(f, allow_pickle=True)
    return {str(i): (lm if ok else None) for i, lm, ok in zip(z["image_id"], z["landmarks"], z["found"])}


def attribution_map(iid, phi, lms):
    masks = region_masks(lms.get(iid), size=100)
    phi = np.nan_to_num(phi)
    m = np.abs(phi).max() or 1.0
    a = (masks * (phi / m)[:, None, None]).sum(0)
    return Image.fromarray(((a + 1) * 127.5).astype(np.uint8))


class Data:
    def __init__(self, d, idx, ablation, lms, train):
        import torch
        import torchvision.transforms.functional as TF
        self.d, self.idx, self.abl, self.lms, self.train = d, idx, ablation, lms, train
        self.torch, self.TF = torch, TF

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        torch, TF = self.torch, self.TF
        i = self.idx[k]
        img = Image.open(self.d["paths"][i]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        x = TF.normalize(TF.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.abl.endswith("attr"):
            a = attribution_map(self.d["ids"][i], self.d["shap"][i], self.lms).resize((SIZE, SIZE), Image.NEAREST)
            x = torch.cat([x, TF.to_tensor(a) * 2 - 1], 0)
        if self.train and np.random.rand() < 0.5:
            x = torch.flip(x, [2])                      # flip the face (and its map) left-right
        v = torch.tensor(np.log(np.clip(self.d["P"][i], 1e-6, 1)), dtype=torch.float32)
        return x, v, int(self.d["y"][i])


def build(ablation, backbone, pretrained=True):
    import timm
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                       in_chans=4 if ablation.endswith("attr") else 3)
            extra = K if "pred" in ablation else 0
            self.use_pred = extra > 0
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.f.num_features + extra, K))

        def forward(self, x, v):
            h = self.f(x)
            if self.use_pred:
                h = torch.cat([h, v], 1)
            return self.head(h)
    return Net()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablations", default=",".join(ABLATIONS))
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--floor", type=float, default=0.3, help="run only where the shared detector score >= floor")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use at most N images per split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true", help="smoke test without downloading weights")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lms = landmark_lookup(a.out)
    tr_ids = pd.read_csv(os.path.join(a.out, "explanations", "V_train.csv")).image_id.astype(str).sort_values().values
    D = {"train": load_split(a.out, "train", tr_ids), "val": load_split(a.out, "val"), "test": load_split(a.out, "test")}
    det = {}
    for s in ("val", "test"):
        t = pd.read_csv(pred_path(a.out, "det-pred+expl+face", s))
        t["image_id"] = t.image_id.astype(str)
        det[s] = t.set_index("image_id").reindex(D[s]["ids"]).score.values
    fail = {s: np.where(D[s]["pred"] != D[s]["y"])[0] for s in ("train", "val")}
    run_on = {s: np.where(det[s] >= a.floor)[0] for s in ("val", "test")}
    if a.limit:
        fail = {s: v[:a.limit] for s, v in fail.items()}
        run_on = {s: v[:a.limit] for s, v in run_on.items()}
    print(f"training failures {len(fail['train'])}, validation failures {len(fail['val'])}; corrector runs on "
          f"{len(run_on['val'])} validation and {len(run_on['test'])} test images (detector >= {a.floor})")

    for abl in a.ablations.split(","):
        t0 = time.time()
        net = build(abl, a.backbone, not a.no_pretrained).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        # class weights: failures are dominated by a few true classes (neutral above all)
        cnt = np.bincount(D["train"]["y"][fail["train"]], minlength=K).astype(float)
        w = torch.tensor(np.where(cnt > 0, cnt.sum() / (K * np.maximum(cnt, 1)), 0), dtype=torch.float32, device=dev)
        lossf = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
        tl = DataLoader(Data(D["train"], fail["train"], abl, lms, True), batch_size=a.batch, shuffle=True, num_workers=a.workers)
        vl = DataLoader(Data(D["val"], fail["val"], abl, lms, False), batch_size=64, num_workers=a.workers)
        best, best_state = -1, None
        for ep in range(a.epochs):
            net.train()
            for x, v, y in tl:
                x, v, y = x.to(dev), v.to(dev), y.to(dev)
                loss = lossf(net(x, v), y)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            net.eval(); right = n = 0
            with torch.no_grad():
                for x, v, y in vl:
                    right += (net(x.to(dev), v.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
            acc = right / max(n, 1)
            print(f"  [{abl}] epoch {ep}: validation-failure accuracy {acc:.4f}", flush=True)
            if acc > best:
                best, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state); net.eval()
        for s in ("val", "test"):
            dl = DataLoader(Data(D[s], run_on[s], abl, lms, False), batch_size=64, num_workers=a.workers)
            probs = []
            with torch.no_grad():
                for x, v, _ in dl:
                    probs.append(torch.softmax(net(x.to(dev), v.to(dev)).float(), 1).cpu().numpy())
            P = np.concatenate(probs) if probs else np.zeros((0, K))
            p = pred_path(a.out, f"corr-cnn-{abl}", s)
            if os.path.exists(p):
                os.remove(p)
            i = run_on[s]
            append_predictions(p, D[s]["ids"][i], D[s]["y"][i], P)
        print(f"[{abl}] best validation-failure accuracy {best:.4f}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
