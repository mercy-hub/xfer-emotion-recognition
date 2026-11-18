"""Source A: CLIP used zero-shot.

Each class is described by several prompt templates; their text embeddings are averaged (prompt ensembling) and the
image is scored by cosine similarity, scaled by CLIP's learned temperature and turned into probabilities with a
softmax. No training on RAF-DB, so the errors of this source are independent of the training split.
Default model: openai/clip-vit-large-patch14 (the ViT-L/14 family that LLaVA-1.5's encoder also comes from, which is
why the reasoner tags A and V as correlated).
"""
import numpy as np
from PIL import Image

from ..labels import CLASSES, CLIP_WORDS

TEMPLATES = [
    "a photo of a {adj} face.",
    "a close-up photo of a person looking {adj}.",
    "a face showing {noun}.",
    "a photo of a person with a {adj} facial expression.",
    "an image of a {adj} person.",
]


def _tensor(out):
    """Newer transformers versions may return an output object instead of a tensor."""
    import torch
    if torch.is_tensor(out):
        return out
    for k in ("text_embeds", "image_embeds", "pooler_output"):
        if getattr(out, k, None) is not None:
            return getattr(out, k)
    return out[0]


class ClipZeroShot:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None, dtype=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.text_emb = self._class_embeddings()

    def _class_embeddings(self):
        torch = self.torch
        embs = []
        with torch.no_grad():
            for c in CLASSES:
                adj, noun = CLIP_WORDS[c]
                texts = [t.format(adj=adj, noun=noun) for t in TEMPLATES]
                tok = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
                e = _tensor(self.model.get_text_features(**tok))
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(0)
                embs.append(e / e.norm())
        return torch.stack(embs)  # (K, d)

    def predict_proba(self, paths):
        return self.predict_proba_images([Image.open(p).convert("RGB") for p in paths])

    def predict_proba_images(self, imgs):
        torch = self.torch
        imgs = [im.convert("RGB") for im in imgs]
        with torch.no_grad():
            px = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            e = _tensor(self.model.get_image_features(pixel_values=px))
            e = e / e.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * e @ self.text_emb.T
            return torch.softmax(logits.float(), dim=-1).cpu().numpy()
