"""Source B: POSTER++ (Mao et al., 2023), using the authors' released model code.

The official repository (github.com/Talented-Q/POSTER_V2) hard-codes Windows paths to its two pretrained backbones
and selects the best epoch on the test set. Here the model class is imported from the repository after patching
those paths, and training is re-done with our own loop so that the best epoch is chosen on the validation split
and the test set is used once. Optimiser settings follow the repository: Adam inside SAM (rho 0.05), learning rate
3.5e-5, weight decay 1e-4, exponential decay 0.98 per epoch, batch 144 (reduce if memory is short).
"""
import os
import re
import subprocess
import sys

REPO_URL = "https://github.com/Talented-Q/POSTER_V2.git"
# Google Drive ids of the pretrained backbones, from the repository README
PRETRAIN = {"ir50.pth": "17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt",
            "mobilefacenet_model_best.pth.tar": "1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX"}


def setup(repo_dir, pretrain_dir):
    """Clone the repository if needed, patch the hard-coded paths, download the two backbones."""
    if not os.path.isdir(os.path.join(repo_dir, "models")):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, repo_dir], check=True)
    f = os.path.join(repo_dir, "models", "PosterV2_7cls.py")
    s = open(f).read()
    if "XFER_PATCHED" not in s:
        s = "import os  # XFER_PATCHED\n" + s
        s = re.sub(r"torch\.load\(r'[^']*mobilefacenet_model_best\.pth\.tar',",
                   "torch.load(os.path.join(os.environ['POSTER_PRETRAIN'], 'mobilefacenet_model_best.pth.tar'), weights_only=False,", s)
        s = re.sub(r"torch\.load\(r'[^']*ir50\.pth',",
                   "torch.load(os.path.join(os.environ['POSTER_PRETRAIN'], 'ir50.pth'), weights_only=False,", s)
        s = s.replace("from thop import profile", "try:\n    from thop import profile\nexcept ImportError:\n    profile = None")
        open(f, "w").write(s)
    os.makedirs(pretrain_dir, exist_ok=True)
    for name, gid in PRETRAIN.items():
        dst = os.path.join(pretrain_dir, name)
        if not os.path.exists(dst):
            import gdown
            gdown.download(id=gid, output=dst, quiet=False)
    os.environ["POSTER_PRETRAIN"] = os.path.abspath(pretrain_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(repo_dir))


def build_model(num_classes=7):
    from models.PosterV2_7cls import pyramid_trans_expr2
    return pyramid_trans_expr2(img_size=224, num_classes=num_classes)


def transforms(train):
    import torchvision.transforms as T
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ToTensor(), norm,
                          T.RandomErasing(scale=(0.02, 0.1))])
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])


class ManifestDataset:
    def __init__(self, df, train):
        self.paths, self.labels = df.path.tolist(), df.label.astype(int).tolist()
        self.ids = df.image_id.astype(str).tolist()
        self.tf = transforms(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def strip_module(state):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


class PosterPredictor:
    """Loads the trained POSTER++ checkpoint (best epoch on validation) for prediction on PIL images."""
    def __init__(self, work, ckpt):
        import torch
        self.torch = torch
        setup(os.path.join(work, "POSTER_V2"), os.path.join(work, "pretrain"))
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model()
        self.model.load_state_dict(strip_module(torch.load(ckpt, map_location="cpu", weights_only=False)))
        self.model = self.model.to(self.dev).eval()
        self.tf = transforms(train=False)

    def predict_proba_images(self, imgs, batch=64):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                x = torch.stack([self.tf(im.convert("RGB")) for im in imgs[i:i + batch]]).to(self.dev)
                out.append(torch.softmax(self.model(x).float(), -1).cpu().numpy())
        import numpy as np
        return np.concatenate(out)
