"""Off-the-shelf expert trained on a DIFFERENT dataset: HSEmotion (Savchenko), EfficientNet-B2 pre-trained for face
identification on VGGFace2 and fine-tuned for facial expressions on AffectNet. No RAF-DB image was used in its
training, so its mistakes are not tied to RAF-DB's training split (unlike POSTER++ and UniFER).

  enet_b2_7  seven AffectNet classes                            -> tag H7
  enet_b2_8  eight classes (with contempt; contempt is dropped) -> tag H8
Weights: github.com/HSE-asavchenko/face-emotion-recognition (Apache-2.0). Needs timm 0.9.x to unpickle.
Usage:  python scripts/29_hsemotion.py --out OUT [--models enet_b2_7,enet_b2_8] [--splits train,val,test]
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xfer.io import append_predictions, load_manifest, pred_path
from xfer.labels import CLASSES
from xfer.metrics import accuracy, macro_f1

URL = "https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/models/affectnet_emotions/{}.pt"
HSE7 = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
HSE8 = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="enet_b2_7,enet_b2_8")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--cache", default="/kaggle/working/hsemotion")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import torch
    import torchvision.transforms as T
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = T.Compose([T.Resize((260, 260)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    m = load_manifest(a.out)
    os.makedirs(a.cache, exist_ok=True)
    for name in a.models.split(","):
        f = os.path.join(a.cache, name + ".pt")
        if not os.path.exists(f):
            urllib.request.urlretrieve(URL.format(name), f)
        net = torch.load(f, map_location="cpu", weights_only=False).to(dev).eval()
        order = HSE7 if name.endswith("_7") else HSE8
        cols = [order.index(c) for c in CLASSES]          # reorder to our class order (drops contempt for _8)
        tag = "H7" if name.endswith("_7") else "H8"
        for split in a.splits.split(","):
            df = m[m.split == split]
            if a.limit:
                df = df.head(a.limit)
            probs = []
            with torch.no_grad():
                for i in range(0, len(df), a.batch):
                    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in df.path.iloc[i:i + a.batch]]).to(dev)
                    z = net(x).float()[:, cols]
                    probs.append(torch.softmax(z, -1).cpu().numpy())
            P, y = np.concatenate(probs), df.label.astype(int).values
            path = pred_path(a.out, tag, split)
            if os.path.exists(path):
                os.remove(path)
            append_predictions(path, df.image_id.astype(str).tolist(), y, P)
            print(f"{tag} ({name}, AffectNet-trained) {split}: acc {100 * accuracy(y, P.argmax(1)):.2f}%  "
                  f"macro-F1 {macro_f1(y, P.argmax(1)):.4f}", flush=True)


if __name__ == "__main__":
    main()
