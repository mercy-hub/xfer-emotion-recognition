"""Train a non-POSTER expert on the RAF-DB training split (best epoch chosen on validation), then predict
train, val and test. The test split is used once, after training.

--arch clip      CLIP ViT-L/14 image encoder fine-tuned end to end. The classifier starts from CLIP's own text
                 embeddings of the emotion prompts (the zero-shot classifier), so training begins from what CLIP
                 already knows. Cross-entropy with label smoothing. Tag A2 by default.
--arch convnext  Hard-case specialist: ImageNet ConvNeXt-Tiny trained with a class-balanced focal loss (more weight
                 on rare, hard classes) and lighting, pose and crop augmentation. Tag S by default.
Usage:  python scripts/28_finetune_expert.py --out OUT --arch clip|convnext [--epochs N] [--tag T]
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xfer.io import append_predictions, load_manifest, pred_path, set_seed
from xfer.labels import CLASSES, K
from xfer.metrics import accuracy, macro_f1

CLIP_MEAN, CLIP_STD = [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def tfm(train, mean, std):
    import torchvision.transforms as T
    if train:
        return T.Compose([T.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
                          T.RandomApply([T.RandomRotation(12)], p=0.5),
                          T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
                          T.RandomGrayscale(p=0.1), T.ToTensor(), T.Normalize(mean, std),
                          T.RandomErasing(p=0.25, scale=(0.02, 0.12))])
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])


class DS:
    def __init__(self, df, tf):
        self.paths, self.y, self.ids, self.tf = df.path.tolist(), df.label.astype(int).tolist(), \
            df.image_id.astype(str).tolist(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i]


def build_clip(torch, name):
    from xfer.sources.clip_zs import ClipZeroShot
    zs = ClipZeroShot(name, dtype=torch.float32)
    W = zs.text_emb.float().clone()                     # (K, d) zero-shot classifier
    clip = zs.model
    try:   # recompute activations in the backward pass (T4 memory)
        clip.gradient_checkpointing_enable()
    except Exception as e:
        print("[clip] no gradient checkpointing:", e)

    class ClipFT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision, self.proj = clip.vision_model, clip.visual_projection
            self.head = torch.nn.Parameter(W)
            self.scale = torch.nn.Parameter(torch.tensor(math.log(50.0)))

        def forward(self, x):
            e = self.proj(self.vision(pixel_values=x).pooler_output)
            e = e / e.norm(dim=-1, keepdim=True)
            h = self.head / self.head.norm(dim=-1, keepdim=True)
            return self.scale.exp() * e @ h.t()

    m = ClipFT()
    del zs
    return m


def focal_loss(torch, w, gamma=2.0):
    def f(logits, y):
        logp = torch.log_softmax(logits.float(), -1)
        lp = logp.gather(1, y[:, None])[:, 0]
        return (-(w[y]) * (1 - lp.exp()) ** gamma * lp).mean()
    return f


def run_eval(model, loader, torch, dev):
    model.eval(); P, Y = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for x, y in loader:
            P.append(torch.softmax(model(x.to(dev)).float(), -1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["clip", "convnext"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--work", default="/kaggle/working/experts")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    import torch
    from torch.utils.data import DataLoader
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = a.arch == "clip"
    tag = a.tag or ("A2" if clip else "S")
    epochs = a.epochs or (8 if clip else 25)
    batch = a.batch or (32 if clip else 64)
    lr = a.lr or (1e-5 if clip else 2e-4)
    mean, std = (CLIP_MEAN, CLIP_STD) if clip else (IMNET_MEAN, IMNET_STD)
    m = load_manifest(a.out)
    if a.limit:
        m = m.groupby("split", group_keys=False).head(a.limit)
    tr = m[m.split == "train"]
    mk = lambda s, train: DataLoader(DS(m[m.split == s], tfm(train, mean, std)), batch_size=batch, shuffle=train,
                                     num_workers=4, pin_memory=True, drop_last=train and len(m[m.split == s]) > batch)
    if clip:
        model = build_clip(torch, a.clip_model)
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith(("head", "scale"))], "lr": lr},
                  {"params": [model.head, model.scale], "lr": lr * 10}]
        wd = 0.1
    else:
        import timm
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=K, drop_path_rate=0.1)
        cnt = np.bincount(tr.label.astype(int), minlength=K).astype(float)
        w = 1 / np.sqrt(cnt); w = w / (w * cnt).sum() * cnt.sum()          # class-balanced (inverse square root)
        crit = focal_loss(torch, torch.tensor(w, dtype=torch.float32, device=dev))
        groups = [{"params": model.parameters(), "lr": lr}]
        wd = 0.05
    model = model.to(dev)
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    tl = mk("train", True)
    total = epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, len(tl))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    os.makedirs(a.work, exist_ok=True)
    best_path, best_f1 = os.path.join(a.work, f"{tag}_best.pth"), -1.0
    vl = mk("val", False)
    print(f"[{tag}] {a.arch}: {len(tr)} training images, {epochs} epochs, batch {batch}, lr {lr}", flush=True)
    for ep in range(epochs):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        p, yv = run_eval(model, vl, torch, dev)
        f1 = macro_f1(yv, p.argmax(1))
        print(f"[{tag}] epoch {ep}: loss {tot / n:.4f}  val acc {accuracy(yv, p.argmax(1)):.4f}  val macro-F1 {f1:.4f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=False))
    for s in ("train", "val", "test"):
        p, y = run_eval(model, mk(s, False), torch, dev)
        path = pred_path(a.out, tag, s)
        if os.path.exists(path):
            os.remove(path)
        append_predictions(path, m[m.split == s].image_id.astype(str).tolist(), y, p)
        print(f"{tag} {s}: acc {100 * accuracy(y, p.argmax(1)):.2f}%  macro-F1 {macro_f1(y, p.argmax(1)):.4f}", flush=True)


if __name__ == "__main__":
    main()
