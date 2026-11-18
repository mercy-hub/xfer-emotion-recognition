"""Source V (new design): a frozen, open vision-language model prompted zero-shot.

Default model: InternVL3-8B (OpenGVLab/InternVL3-8B-hf), the best 7-8B open model on RAF-DB in FERBench without any
emotion training. Any Hugging Face image-text-to-text model with a chat template works (for example
Qwen/Qwen2.5-VL-7B-Instruct), because the class only uses AutoProcessor and AutoModelForImageTextToText.

Three uses:
  predict_proba(paths)      class probabilities from the first token of each label word at the answer position,
                            renormalised over the seven classes (Equation 3.2), with a full-word fallback.
  describe(paths)           structured, FACS-style description: emotion plus brows, eyes, nose, cheeks, mouth and a
                            one-sentence reason (the claims checked against MediaPipe in stage 2).
  describe_given(paths, e)  the same description when the model is TOLD the emotion e. Used only on training and
                            validation images, to build reference explanations for the SLM. Never on test images.
"""
import re

import numpy as np
from PIL import Image

from ..labels import CLASSES, NAME_ALIASES, VLM_WORDS

DEFAULT_MODEL = "OpenGVLab/InternVL3-8B-hf"

QUESTION = ("Look at the person's face. Which one of these facial expressions does the person show: "
            "Surprise, Fear, Disgust, Happiness, Sadness, Anger or Neutral? Answer with one word.")

FIELDS = ["Emotion", "Brows", "Eyes", "Nose", "Cheeks", "Mouth", "Reason"]
_FORMAT = ("Brows: raised, lowered, drawn together or relaxed\n"
           "Eyes: wide open, narrowed, closed or relaxed\n"
           "Nose: wrinkled or relaxed\n"
           "Cheeks: raised or relaxed\n"
           "Mouth: smiling, open, corners down, lips pressed, upper lip raised or relaxed\n"
           "Reason: one sentence naming the visible features that support the emotion")
DESCRIBE_Q = ("Look at the person's face. Which one of these facial expressions does the person show: "
              "Surprise, Fear, Disgust, Happiness, Sadness, Anger or Neutral? Reply in exactly this format, "
              "one line each, choosing from the options given:\n"
              "Emotion: one word from the list\n" + _FORMAT)
GIVEN_Q = ("The facial expression of this person is {emotion}. Describe the visible facial features that show "
           "this expression. Reply in exactly this format, one line each, choosing from the options given:\n"
           + _FORMAT)


def parse_reply(text):
    """Structured reply -> dict with one key per field (lower case) and 'emotion_idx' (-1 if unreadable)."""
    out = {f.lower(): "" for f in FIELDS}
    for line in str(text).splitlines():
        m = re.match(r"\s*\**\s*(" + "|".join(FIELDS) + r")\s*\**\s*[:\-]\s*(.*)", line, flags=re.I)
        if m:
            out[m.group(1).lower()] = m.group(2).strip().strip("*").strip()
    emo = out["emotion"].lower()
    if not emo:  # free answer: take the first class word anywhere in the text
        hits = [(str(text).lower().find(w), w) for w in NAME_ALIASES if w in str(text).lower()]
        emo = min(hits)[1] if hits else ""
    word = re.sub(r"[^a-z]", "", emo.split()[0]) if emo else ""
    out["emotion_idx"] = CLASSES.index(NAME_ALIASES[word]) if word in NAME_ALIASES else -1
    out["parsed_fields"] = sum(bool(out[f.lower()]) for f in FIELDS[1:6])  # the five face fields
    return out


# extra surface forms per class, scored together (models fine-tuned for emotion may answer "Happy" or "sad")
FORMS = {"surprise": ["Surprise", "Surprised", "surprise", "surprised"], "fear": ["Fear", "Fearful", "fear", "fearful"],
         "disgust": ["Disgust", "Disgusted", "disgust", "disgusted"],
         "happiness": ["Happiness", "Happy", "happiness", "happy"], "sadness": ["Sadness", "Sad", "sadness", "sad"],
         "anger": ["Anger", "Angry", "anger", "angry"], "neutral": ["Neutral", "neutral"]}


class VLMZeroShot:
    def __init__(self, model_id=DEFAULT_MODEL, load_4bit=True, device_map="auto", answer_prefix="", forms=False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.dtype = torch.float16  # the T4 has no bfloat16
        kw = dict(device_map=device_map)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            # keep the vision tower and projector in float16: quantising them hurts small faces most
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype, bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["vision_tower", "vision_model", "visual", "multi_modal_projector",
                                       "mlp1", "lm_head"])
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=self.dtype, **kw).eval()
        except TypeError:  # older transformers use torch_dtype
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=self.dtype, **kw).eval()
        try:
            self.proc = AutoProcessor.from_pretrained(model_id)
        except Exception as e:   # some fine-tuned checkpoints ship without a processor: use the base model's
            print(f"[V] no processor in {model_id} ({type(e).__name__}); using Qwen/Qwen2.5-VL-7B-Instruct's")
            self.proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.tok = getattr(self.proc, "tokenizer", self.proc)
        self.tok.padding_side = "left"
        self.model_id = model_id
        self.answer_prefix = answer_prefix          # e.g. "<think>\n</think>\n<answer>" for reasoning-trained models
        self.prompt = self._prompt(QUESTION) + answer_prefix
        self.first_ids = [self._answer_ids(VLM_WORDS[c])[0] for c in CLASSES]
        self.mode = "first_token" if len(set(self.first_ids)) == len(CLASSES) else "full_word"
        self.form_ids = None
        if forms:   # union of the first tokens of every surface form; used only if no token is shared by two classes
            ids = {c: sorted({self._answer_ids(w)[0] for w in FORMS[c]} | {self._answer_ids_raw(w)[0] for w in FORMS[c]})
                   for c in CLASSES}
            flat = [i for c in CLASSES for i in ids[c]]
            if len(flat) == len(set(flat)):
                self.form_ids = ids
                self.mode = "first_token_forms"
        print(f"[V] {model_id}; scoring mode: {self.mode}; first tokens: "
              f"{dict(zip(CLASSES, self.tok.convert_ids_to_tokens(self.first_ids)))}"
              + (f"; word forms: { {c: self.tok.convert_ids_to_tokens(v) for c, v in self.form_ids.items()} }"
                 if self.form_ids else ""))

    # ------------------------------------------------------------------ prompts and inputs
    def _prompt(self, question):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.proc.apply_chat_template(conv, add_generation_prompt=True)

    def _join(self, prompt, word):
        return prompt + ("" if prompt.endswith(("\n", " ")) else " ") + word

    def _answer_ids(self, word):
        base = self.tok(self.prompt, add_special_tokens=False).input_ids
        full = self.tok(self._join(self.prompt, word), add_special_tokens=False).input_ids
        k = next(i for i, (a, b) in enumerate(zip(base + [None], full)) if a != b)
        return full[k:]

    def _answer_ids_raw(self, word):
        """Tokens of the word appended with no space (a model may answer '<answer>Happy')."""
        base = self.tok(self.prompt, add_special_tokens=False).input_ids
        full = self.tok(self.prompt + word, add_special_tokens=False).input_ids
        k = next(i for i, (a, b) in enumerate(zip(base + [None], full)) if a != b)
        return full[k:]

    def _inputs(self, imgs, prompts):
        kw = dict(images=imgs, text=prompts, return_tensors="pt", padding=True)
        try:  # InternVL: one 448 x 448 tile is enough for a 100 x 100 face (no extra crops)
            inp = self.proc(crop_to_patches=False, **kw)
        except (TypeError, ValueError):
            inp = self.proc(**kw)
        inp = inp.to(self.model.device)
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
        return inp

    @staticmethod
    def _load(paths):
        return [Image.open(p).convert("RGB") for p in paths]

    # ------------------------------------------------------------------ probabilities
    def predict_proba(self, paths):
        return self.predict_proba_images(self._load(paths))

    def predict_proba_images(self, imgs):
        torch = self.torch
        imgs = [im.convert("RGB") for im in imgs]
        if self.mode in ("first_token", "first_token_forms"):
            with torch.no_grad():
                inp = self._inputs(imgs, [self.prompt] * len(imgs))
                logits = self.model(**inp).logits[:, -1, :].float()
            if self.mode == "first_token":
                return torch.softmax(logits[:, self.first_ids], -1).cpu().numpy()
            allids = [i for c in CLASSES for i in self.form_ids[c]]
            p = torch.softmax(logits[:, allids], -1).cpu().numpy()
            out, j = np.zeros((len(imgs), len(CLASSES))), 0
            for k, c in enumerate(CLASSES):
                n = len(self.form_ids[c]); out[:, k] = p[:, j:j + n].sum(1); j += n
            return out
        return np.stack([self._full_word_proba(im) for im in imgs])

    def _full_word_proba(self, img):
        torch = self.torch
        scores = []
        with torch.no_grad():
            for c in CLASSES:
                ans = self._answer_ids(VLM_WORDS[c])
                inp = self._inputs([img], [self._join(self.prompt, VLM_WORDS[c])])
                logp = torch.log_softmax(self.model(**inp).logits[0].float(), -1)
                L = inp["input_ids"].shape[1]
                scores.append(sum(logp[p, t].item() for p, t in zip(range(L - len(ans) - 1, L - 1), ans)))
        s = np.array(scores)
        e = np.exp(s - s.max())
        return e / e.sum()

    # ------------------------------------------------------------------ text
    def answer_relevance(self, img, target=None):
        """Answer-token relevance (input x gradient) of one image for the class word's first token; returns
        (relevance map at the image size, target index). Needs gradients through the frozen model."""
        torch = self.torch
        inp = self._inputs([img.convert("RGB")], [self.prompt])
        pv = inp["pixel_values"].detach().clone().requires_grad_(True)
        inp["pixel_values"] = pv
        logits = self.model(**inp).logits[0, -1, :].float()
        sel = logits[self.first_ids]
        t = int(sel.argmax()) if target is None else int(target)
        self.model.zero_grad(set_to_none=True)
        sel[t].backward()
        g = (pv.grad * pv.detach()).float().abs().sum(1)[0]          # (H, W) of the first tile
        g = torch.nn.functional.interpolate(g[None, None], size=img.size[::-1], mode="bilinear")[0, 0]
        return g.cpu().numpy(), t

    def _generate(self, imgs, questions, max_new_tokens):
        torch = self.torch
        with torch.no_grad():
            inp = self._inputs(imgs, [self._prompt(q) for q in questions])
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return [t.strip() for t in self.tok.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]

    def describe(self, paths, max_new_tokens=110):
        return self._generate(self._load(paths), [DESCRIBE_Q] * len(paths), max_new_tokens)

    def describe_given(self, paths, emotions, max_new_tokens=100):
        qs = [GIVEN_Q.format(emotion=VLM_WORDS[CLASSES[int(e)]].lower()) for e in emotions]
        return self._generate(self._load(paths), qs, max_new_tokens)
