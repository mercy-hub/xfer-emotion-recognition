"""Source V: generative vision-language model (LLaVA-1.5-7B) prompted zero-shot.

Class probabilities (Equation 3.2): one forward pass on the prompt; the scores of the first token of each class word
at the answer position are renormalised over the seven classes. If two class words share a first token (which
would make that impossible), the code falls back to scoring the full word for each class (seven passes per image).
The model's short text reason is produced separately (generate_rationales) because it needs slow generation.
"""
import numpy as np
from PIL import Image

from ..labels import CLASSES, VLM_WORDS

QUESTION = ("Look at the person's face. Which one of these facial expressions does the person show: "
            "Surprise, Fear, Disgust, Happiness, Sadness, Anger or Neutral? Answer with one word.")
REASON_Q = ("Look at the person's face. Which one of these facial expressions does the person show: "
            "Surprise, Fear, Disgust, Happiness, Sadness, Anger or Neutral? Give the word, then in one sentence "
            "name the visible facial features (brows, eyes, mouth) that support it.")


class LlavaZeroShot:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", load_4bit=True, device_map="auto"):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.torch = torch
        kw = dict(device_map=device_map)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = torch.float16
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **kw).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"
        self.prompt = self._prompt(QUESTION)
        self.first_ids = self._first_token_ids()
        self.mode = "first_token" if len(set(self.first_ids)) == len(CLASSES) else "full_word"
        print(f"[V] scoring mode: {self.mode}; first tokens: "
              f"{dict(zip(CLASSES, self.proc.tokenizer.convert_ids_to_tokens(self.first_ids)))}")

    def _prompt(self, question):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        try:
            return self.proc.apply_chat_template(conv, add_generation_prompt=True)
        except Exception:  # older processors without a chat template
            return f"USER: <image>\n{question} ASSISTANT:"

    def _answer_ids(self, word):
        tok = self.proc.tokenizer
        base = tok(self.prompt, add_special_tokens=False).input_ids
        full = tok(self.prompt + " " + word, add_special_tokens=False).input_ids
        # first position where the two tokenisations differ
        k = next(i for i, (a, b) in enumerate(zip(base + [None], full)) if a != b)
        return full[k:]

    def _first_token_ids(self):
        return [self._answer_ids(VLM_WORDS[c])[0] for c in CLASSES]

    def _inputs(self, imgs, prompts):
        return self.proc(images=imgs, text=prompts, return_tensors="pt", padding=True).to(self.model.device,
                                                                                            self.torch.float16)

    def predict_proba(self, paths):
        return self.predict_proba_images([Image.open(p).convert("RGB") for p in paths])

    def predict_proba_images(self, imgs):
        torch = self.torch
        imgs = [im.convert("RGB") for im in imgs]
        if self.mode == "first_token":
            with torch.no_grad():
                inp = self._inputs(imgs, [self.prompt] * len(imgs))
                logits = self.model(**inp).logits[:, -1, :].float()   # left padding: last position is the answer slot
                sel = logits[:, self.first_ids]
                return torch.softmax(sel, -1).cpu().numpy()
        return np.stack([self._full_word_proba(im) for im in imgs])

    def _full_word_proba(self, img):
        torch = self.torch
        scores = []
        with torch.no_grad():
            for c in CLASSES:
                ans = self._answer_ids(VLM_WORDS[c])
                inp = self._inputs([img], [self.prompt + " " + VLM_WORDS[c]])
                logp = torch.log_softmax(self.model(**inp).logits[0].float(), -1)
                L = inp["input_ids"].shape[1]
                pos = range(L - len(ans) - 1, L - 1)
                scores.append(sum(logp[p, t].item() for p, t in zip(pos, ans)))
        s = np.array(scores)
        e = np.exp(s - s.max())
        return e / e.sum()

    def generate_rationales(self, paths, max_new_tokens=48, question=None):
        torch = self.torch
        imgs = [Image.open(p).convert("RGB") for p in paths]
        prompt = self._prompt(question or REASON_Q)
        with torch.no_grad():
            inp = self._inputs(imgs, [prompt] * len(imgs))
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        texts = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return [t.strip() for t in texts]
