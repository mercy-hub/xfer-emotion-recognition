"""Step 14 (E4, small vision-language arm B): Qwen2.5-VL-3B with LoRA corrects V on the images where it fails.

Same training images, validation images and detector floor as step 13. The model sees the face and a prompt, and is
taught to answer with the true expression word on V's failures. Its class probabilities are read from the first token
of the seven expression words (as for V, Equation 3.2). Ablations of the prompt:
  image             the question only
  image+pred        plus V's answer and confidence, and its second choice
  image+pred+expl   plus the regions that most supported and most opposed V's answer (region SHAP values)
Qwen2.5-VL has its own vision encoder, unlike LLaVA-1.5 and CLIP, which share CLIP's (section 3.7).
Writes predictions/corr-vlm-<ablation>_<split>.csv for step 15. One T4 GPU; 4-bit weights with LoRA adapters.
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
from xfer.labels import CLASSES, K, VLM_WORDS
from xfer.regions import REGIONS

ABLATIONS = ["image", "image+pred", "image+pred+expl"]
QUESTION = ("Which facial expression does the person show? Answer with one word from: Surprise, Fear, Disgust, "
            "Happiness, Sadness, Anger, Neutral.")


def prompt_text(d, i, abl):
    parts = []
    if "pred" in abl:
        o = np.argsort(-d["P"][i])
        parts.append(f"A first model answered {VLM_WORDS[CLASSES[o[0]]]} (confidence {d['P'][i, o[0]]:.2f}); its "
                     f"second choice was {VLM_WORDS[CLASSES[o[1]]]} ({d['P'][i, o[1]]:.2f}).")
    if "expl" in abl and not np.isnan(d["shap"][i]).any():
        phi = d["shap"][i]
        o = np.argsort(-phi)
        parts.append(f"The face regions that most supported that answer were the {REGIONS[o[0]]} ({phi[o[0]]:+.2f}) "
                     f"and the {REGIONS[o[1]]} ({phi[o[1]]:+.2f}); the region that most opposed it was the "
                     f"{REGIONS[o[-1]]} ({phi[o[-1]]:+.2f}).")
    parts.append(QUESTION)
    return " ".join(parts)


class Qwen:
    def __init__(self, model_id, lora_r=16):
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        except ImportError:
            from transformers import AutoModelForImageTextToText as M
        self.torch = torch
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        self.model = M.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16)
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=224 * 224, max_pixels=224 * 224)
        self.proc.tokenizer.padding_side = "left"
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        self.model = get_peft_model(self.model, cfg)
        self.model.config.use_cache = False
        self.word_ids = None

    def chat(self, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True)

    def first_ids(self):
        if self.word_ids is None:
            tok = self.proc.tokenizer
            base = tok("\n", add_special_tokens=False).input_ids
            ids = []
            for c in CLASSES:
                full = tok("\n" + VLM_WORDS[c], add_special_tokens=False).input_ids
                k = next(j for j, (p, q) in enumerate(zip(base + [None], full)) if p != q)
                ids.append(full[k])
            assert len(set(ids)) == K, f"class words share a first token: {ids}"
            self.word_ids = ids
        return self.word_ids

    def train_step(self, img, text, answer):
        torch = self.torch
        prompt = self.chat(text)
        full = prompt + answer + "<|im_end|>"
        enc = self.proc(text=[full], images=[img], return_tensors="pt").to(self.model.device)
        plen = self.proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :plen] = -100
        out = self.model(**enc, labels=labels)
        return out.loss

    def proba(self, imgs, texts):
        torch = self.torch
        enc = self.proc(text=[self.chat(t) for t in texts], images=imgs, return_tensors="pt", padding=True)
        enc = enc.to(self.model.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[:, -1, :].float()
        return torch.softmax(logits[:, self.first_ids()], -1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablations", default=",".join(ABLATIONS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--floor", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
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
    img = lambda d, i: Image.open(d["paths"][i]).convert("RGB")
    print(f"training failures {len(fail['train'])}; runs on {len(run_on['val'])} validation and "
          f"{len(run_on['test'])} test images")

    for abl in a.ablations.split(","):
        t0 = time.time()
        q = Qwen(a.model)
        torch = q.torch
        params = [p for p in q.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        rng = np.random.default_rng(a.seed)
        best, best_state = -1, None
        for ep in range(a.epochs):
            q.model.train()
            order = rng.permutation(fail["train"])
            tot = 0.0
            for n, i in enumerate(order, 1):
                im = img(D["train"], i)
                if rng.random() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                loss = q.train_step(im, prompt_text(D["train"], i, abl), VLM_WORDS[CLASSES[D["train"]["y"][i]]])
                if torch.isfinite(loss):
                    (loss / a.accum).backward(); tot += loss.item()
                if n % a.accum == 0 or n == len(order):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); opt.zero_grad()
            q.model.eval()
            vf = fail["val"]
            right = 0
            for j in range(0, len(vf), a.batch):
                ii = vf[j:j + a.batch]
                P = q.proba([img(D["val"], i) for i in ii], [prompt_text(D["val"], i, abl) for i in ii])
                right += (P.argmax(1) == D["val"]["y"][ii]).sum()
            acc = right / max(len(vf), 1)
            print(f"  [{abl}] epoch {ep}: train loss {tot / len(order):.4f}, validation-failure accuracy {acc:.4f}",
                  flush=True)
            if acc > best:
                best = acc
                best_state = {k: v.detach().cpu().clone() for k, v in q.model.state_dict().items() if "lora_" in k}
        q.model.load_state_dict(best_state, strict=False)
        q.model.eval()
        for s in ("val", "test"):
            ii_all, probs = run_on[s], []
            for j in range(0, len(ii_all), a.batch):
                ii = ii_all[j:j + a.batch]
                probs.append(q.proba([img(D[s], i) for i in ii], [prompt_text(D[s], i, abl) for i in ii]))
            P = np.concatenate(probs) if probs else np.zeros((0, K))
            p = pred_path(a.out, f"corr-vlm-{abl}", s)
            if os.path.exists(p):
                os.remove(p)
            append_predictions(p, D[s]["ids"][ii_all], D[s]["y"][ii_all], P)
        print(f"[{abl}] best validation-failure accuracy {best:.4f}; {(time.time() - t0) / 60:.0f} min", flush=True)
        del q
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
