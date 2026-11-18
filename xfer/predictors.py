"""One predict(list of PIL images) -> (n, 7) probabilities function per source, for the explanation scripts.

Sources: V2 (the new frozen VLM, InternVL3-8B by default), A (CLIP), B (POSTER++), C (HOG/LBP + SVM), and V (the
old LLaVA-1.5, kept for comparison).
"""
import glob
import os

import numpy as np


def make_predictor(source, out, work="/kaggle/working/poster", ckpt=None, batch=16, vlm_model=None):
    if source == "V2":
        from .sources.vlm_zs import DEFAULT_MODEL, VLMZeroShot
        model = VLMZeroShot(vlm_model or DEFAULT_MODEL)
        batch = min(batch, 8)
    elif source == "V":
        from .sources.llava_zs import LlavaZeroShot
        model = LlavaZeroShot()
        batch = min(batch, 8)
    elif source == "A":
        from .sources.clip_zs import ClipZeroShot
        model = ClipZeroShot()
    elif source == "B":
        from .sources.poster import PosterPredictor
        ckpt = ckpt or (glob.glob(os.path.join(work, "checkpoints_seed42", "best.pth")) +
                        glob.glob("/kaggle/**/checkpoints_seed42/best.pth", recursive=True))[0]
        model = PosterPredictor(work, ckpt)
    elif source == "C":
        from .sources.hog_svm import SvmPredictor
        model = SvmPredictor(os.path.join(out, "models", "svm_C.joblib"))
    else:
        raise ValueError(source)

    def predict(imgs):
        return np.concatenate([np.asarray(model.predict_proba_images(imgs[i:i + batch]), float)
                               for i in range(0, len(imgs), batch)])
    return predict
