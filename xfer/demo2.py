"""The final Phase 2 system applied to one face image, for the demonstration app (scripts/31_demo_app2.py).

Stage 1 (neuro): four experts, each with its own explanation, and the shared region occlusion.
  B   POSTER++                     native: gradient-weighted cross-attention relevance; Integrated Gradients (landmarks)
  A2  CLIP fine-tuned on RAF-DB    native: gradient x attention rollout for the predicted emotion
  A   CLIP zero-shot (same CLIP)   native: the same rollout with the zero-shot classifier
  V2  InternVL3-8B                 native: FACS-style claims checked against MediaPipe; answer-token relevance
  C   HOG/LBP + SVM                native: exact contributions of a linear SVM on the same features
  SLM Qwen2.5-1.5B + LoRA          keep / change / abstain and a short explanation of the VLM's answer
Stage 2 (XAI fusion): occlusion turned into region weights per expert; the final combiner (0.5 CatBoost + 0.5 weighted
  average, fitted on validation, xfer/final) gives the emotion probabilities; per-image CatBoost SHAP values say how
  much each expert drove the decision; a consensus map, agreement score and outlier flags.
Stage 3 (symbolic): the EMONTO + FACS knowledge (read from the OWL file) checks the action-unit pattern, allows a
  compound label only for a confusable pair, ranks geometric evidence above the VLM's claims, and decides: single
  label, compound label or referral to a person, with a justification trace.
Nothing is fitted on the uploaded image. --mock replaces every model with a deterministic fake, for tests without GPU.
"""
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

from . import ontology as ONT
from .claims import FIELDS as CLAIM_FIELDS, check_claims
from .demo import LANDMARKER_URL, FaceFinder, crop_face
from .explain import blurred, perturb
from .labels import CLASSES, K
from .reasoner import au_scores, aus
from .regions import FINE, GEOMETRY, fine_region_masks, geometry

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL = os.path.join(HERE, "final")
EXPERTS = ["B", "A2", "V2", "C", "A"]
NAMES = {"B": "POSTER++", "A2": "CLIP fine-tuned", "V2": "InternVL3-8B (VLM)", "C": "HOG/LBP + SVM",
         "A": "CLIP zero-shot"}
KINDS = {"B": "landmark and image transformer, trained on RAF-DB", "A2": "contrastive vision-language model, fine-tuned",
         "V2": "prompted vision-language model, not trained", "C": "classical features and classifier",
         "A": "the same CLIP before fine-tuning"}
NATIVE = {"B": "Integrated Gradients on the landmark stream", "A2": "attention relevance map",
          "V2": "answer-token relevance", "C": "exact weights of a linear SVM", "A": "attention relevance map"}
AU_NAMES = {"AU1": "inner brows raised", "AU2": "outer brows raised", "AU4": "brows lowered",
            "AU5": "upper eyelids raised", "AU6": "cheeks raised", "AU7": "eyelids tightened", "AU9": "nose wrinkled",
            "AU10": "upper lip raised", "AU12": "lip corners pulled up", "AU15": "lip corners pulled down",
            "AU17": "chin raised", "AU20": "lips stretched", "AU23": "lips pressed", "AU26": "jaw dropped"}
R = len(FINE)


def _script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(ROOT, "scripts", name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def to_regions(sal, masks):
    sal = np.abs(np.asarray(sal, float))
    v = np.array([sal[m].mean() if m.any() else 0.0 for m in masks])
    return v / (v.sum() or 1)


# ============================================================================ experts
class Fake:
    """Deterministic stand-in for an expert (tests without a GPU)."""
    def __init__(self, seed, sharp=4.0):
        self.W = np.random.default_rng(seed).normal(size=(3 * 16 * 16, K))
        self.sharp = sharp

    def predict_proba_images(self, imgs):
        x = np.stack([np.asarray(im.convert("RGB").resize((16, 16)), float).ravel() / 255 for im in imgs])
        return softmax(self.sharp * (x - 0.5) @ self.W / 8)

    def native(self, img, masks, c):
        return np.random.default_rng(c + int(self.W[0, 0] * 1000) % 97).dirichlet(np.ones(R))


class ClipExperts:
    """CLIP ViT-L/14 twice: zero-shot (A, the original weights) and fine-tuned on RAF-DB (A2, scripts/28)."""
    def __init__(self, a2_ckpt, name="openai/clip-vit-large-patch14"):
        import copy
        import torch
        import torchvision.transforms as T
        from transformers import CLIPModel, CLIPProcessor
        from .labels import CLIP_WORDS
        from .sources.clip_zs import TEMPLATES, _tensor
        self.torch, self.dev = torch, ("cuda" if torch.cuda.is_available() else "cpu")
        base = CLIPModel.from_pretrained(name, attn_implementation="eager", torch_dtype=torch.float32).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        with torch.no_grad():
            embs = []
            for c in CLASSES:
                adj, noun = CLIP_WORDS[c]
                tok = self.proc(text=[t.format(adj=adj, noun=noun) for t in TEMPLATES], return_tensors="pt",
                                padding=True).to(self.dev)
                e = _tensor(base.get_text_features(**tok))
                e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
                embs.append(e / e.norm())
        self.zs = dict(vision=base.vision_model, proj=base.visual_projection, head=torch.stack(embs),
                       scale=base.logit_scale.exp().detach())
        st = torch.load(a2_ckpt, map_location="cpu", weights_only=False)
        vis, proj = copy.deepcopy(base.vision_model), copy.deepcopy(base.visual_projection)
        vis.load_state_dict({k[7:]: v for k, v in st.items() if k.startswith("vision.")})
        proj.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("proj.")})
        head = st["head"].to(self.dev)
        self.ft = dict(vision=vis.eval(), proj=proj.eval(), head=head / head.norm(dim=-1, keepdim=True),
                       scale=st["scale"].to(self.dev).exp())
        del base.text_model
        self.tf_ft = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                T.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])])

    def _px(self, imgs, which):
        if which == "A2":
            return self.torch.stack([self.tf_ft(im.convert("RGB")) for im in imgs]).to(self.dev)
        return self.proc(images=[im.convert("RGB") for im in imgs], return_tensors="pt")["pixel_values"].to(self.dev)

    def _logits(self, px, m, attn=False):
        out = m["vision"](pixel_values=px, output_attentions=attn)
        e = m["proj"](out.pooler_output)
        e = e / e.norm(dim=-1, keepdim=True)
        return m["scale"] * e @ m["head"].T, out

    def proba(self, imgs, which):
        m = self.ft if which == "A2" else self.zs
        P = []
        with self.torch.no_grad():
            for i in range(0, len(imgs), 32):
                P.append(self.torch.softmax(self._logits(self._px(imgs[i:i + 32], which), m)[0].float(), -1).cpu().numpy())
        return np.concatenate(P)

    def rollout(self, img, masks, c, which):
        """Gradient x attention rollout (Chefer, Gur and Wolf, 2021) for emotion c, reduced to the regions."""
        torch = self.torch
        m = self.ft if which == "A2" else self.zs
        logits, out = self._logits(self._px([img], which), m, attn=True)
        for A in out.attentions:
            A.retain_grad()
        for p in m["vision"].parameters():
            p.grad = None
        logits[0, c].backward()
        Tn = out.attentions[0].shape[-1]
        Rm = torch.eye(Tn, device=self.dev)
        for A in out.attentions:
            Rm = Rm + (A.grad * A).clamp(min=0).mean(1)[0] @ Rm
        rel = Rm[0, 1:].detach()
        s = int(round(rel.numel() ** 0.5))
        rel = torch.nn.functional.interpolate(rel.reshape(1, 1, s, s), size=img.size[::-1], mode="bilinear")[0, 0]
        return to_regions(rel.cpu().numpy(), masks)


class Proxy:
    def __init__(self, f, native=None):
        self.f, self._native = f, native

    def predict_proba_images(self, imgs):
        return self.f(imgs)

    def native(self, img, masks, c):
        return self._native(img, masks, c) if self._native else np.full(R, np.nan)


class PosterExpert:
    def __init__(self, work, ckpt):
        from .sources.poster import PosterPredictor
        self.P = PosterPredictor(work, ckpt)
        self.nx = _script("23_native_xai.py")

    def predict_proba_images(self, imgs):
        return self.P.predict_proba_images(imgs)

    def native(self, img, masks, c, steps=16):
        """Integrated Gradients on the landmark stream (blurred-face baseline), as in section 3.4.2."""
        torch, model, tf, dev = self.P.torch, self.P.model, self.P.tf, self.P.dev
        x = tf(img.convert("RGB"))[None].to(dev)
        x0 = tf(Image.fromarray(np.clip(blurred(img), 0, 255).astype(np.uint8)))[None].to(dev)
        grads = 0
        for k in range(1, steps + 1):
            xl = (x0 + (k / steps) * (x - x0)).detach().requires_grad_(True)
            out = self.nx.poster_forward(model, x, xl)
            g, = torch.autograd.grad(out[0, c], xl)
            grads = grads + g
        ig = ((x - x0) * grads / steps).abs().sum(1)[0]
        ig = torch.nn.functional.interpolate(ig[None, None], size=img.size[::-1], mode="bilinear")[0, 0]
        return to_regions(ig.detach().cpu().numpy(), masks)


class SvmExpert:
    """The deployed RBF SVM for the answer; a linear SVM on the same features for exact native weights."""
    def __init__(self, svm_path, out, cache):
        import joblib
        from .sources.hog_svm import SvmPredictor
        self.P = SvmPredictor(svm_path)
        self.lin = None
        f = os.path.join(cache, "linsvm.joblib")
        try:
            if os.path.exists(f):
                self.lin = joblib.load(f)
            else:
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import LinearSVC
                from .sources.hog_svm import feature_matrix
                m = pd.read_csv(os.path.join(out, "manifest.csv"))
                tr = m[m.split == "train"]
                X = feature_matrix(tr.path.tolist())
                sc = StandardScaler().fit(X)
                clf = LinearSVC(C=0.003, class_weight="balanced", max_iter=5000, random_state=42).fit(sc.transform(X),
                                                                                                    tr.label.values)
                self.lin = dict(sc=sc, clf=clf)
                os.makedirs(cache, exist_ok=True)
                joblib.dump(self.lin, f)
        except Exception as e:
            print("linear SVM not available:", e)

    def predict_proba_images(self, imgs):
        return self.P.predict_proba_images(imgs)

    def native(self, img, masks, c):
        if self.lin is None:
            return np.full(R, np.nan)
        from .sources.hog_svm import GRID, SIZE, feature_regions, features_from_image
        z = self.lin["sc"].transform(features_from_image(img)[None])[0]
        contrib = self.lin["clf"].coef_[c] * z
        per_cell = np.bincount(feature_regions(), weights=contrib, minlength=GRID * GRID)
        cell = SIZE // GRID
        reg = np.zeros(R)
        for r in range(GRID):
            for q in range(GRID):
                y0, x0 = r * cell + cell // 2, q * cell + cell // 2
                hit = np.where(masks[:, y0, x0])[0]
                if len(hit):
                    reg[hit[0]] += per_cell[r * GRID + q]
        reg = np.clip(reg, 0, None)
        return reg / (reg.sum() or 1)


class VlmExpert:
    def __init__(self, model_id=None):
        from .sources.vlm_zs import DEFAULT_MODEL, VLMZeroShot
        self.V = VLMZeroShot(model_id or DEFAULT_MODEL, device_map={"": 0})

    def predict_proba_images(self, imgs):
        return self.V.predict_proba_images(imgs)

    def describe(self, img):
        from .sources.vlm_zs import DESCRIBE_Q
        return self.V._generate([img.convert("RGB")], [DESCRIBE_Q], 110)[0]

    def native(self, img, masks, c):
        sal, _ = self.V.answer_relevance(img.convert("RGB"), target=c)
        return to_regions(sal, masks)


class FakeVlm(Fake):
    def describe(self, img):
        P = self.predict_proba_images([img])[0]
        c = CLASSES[int(P.argmax())]
        return (f"Emotion: {c.capitalize()}\nBrows: raised\nEyes: wide open\nNose: relaxed\nCheeks: relaxed\n"
                f"Mouth: open\nReason: The raised brows and open mouth suggest {c}.")


class Slm:
    def __init__(self, lora_dir, base="Qwen/Qwen2.5-1.5B-Instruct", mock=False):
        self.mod = _script("24_slm.py")
        self.mock = mock
        self.off = (lora_dir is None) and not mock
        if mock or self.off:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base)
        self.tok.padding_side = "left"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
        self.model = PeftModel.from_pretrained(m.to(dev), lora_dir).eval()
        self.dev = dev

    def run(self, P, desc, active_aus):
        row = {c: float(P[k]) for k, c in enumerate(f"p_{x}" for x in CLASSES)}
        row.update({f: desc.get(f, "") for f in CLAIM_FIELDS})
        row["aus"] = active_aus
        fl = self.mod.fields(row)
        if self.off:
            return dict(decision="not run", emotion=int(np.argmax(P)),
                        explanation="The SLM adapter was not found in this session, so this check was skipped.",
                        raw="", fields=fl)
        if self.mock:
            c = int(np.argmax(P))
            text = (f"Decision: keep | Emotion: {CLASSES[c]} | Explanation: The measured "
                    f"{', '.join(AU_NAMES[a] for a in active_aus[:2]) or 'face'} fits {CLASSES[c]}.")
        else:
            prompt = self.mod.prompt_text(fl, self.tok)
            enc = self.tok([prompt], return_tensors="pt", add_special_tokens=False).to(self.dev)
            with self.torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=70, do_sample=False, pad_token_id=self.tok.eos_token_id)
            text = self.tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        dec, c, expl = self.mod.parse(text, np.asarray(P))
        expl = expl.split("|")[0].strip() or expl
        return dict(decision=dec, emotion=c, explanation=expl.strip(" |"), raw=text, fields=fl)


# ============================================================================ the pipeline
class Final:
    """The final combiner of section 4.7 (fitted by scripts/30_final_system.py)."""
    def __init__(self):
        from catboost import CatBoostClassifier
        self.cfg = json.load(open(os.path.join(FINAL, "final.json")))
        self.models = []
        for s in range(5):
            m = CatBoostClassifier()
            m.load_model(os.path.join(FINAL, f"catboost_s{s}.cbm"))
            self.models.append(m)
        self.w = np.array(self.cfg["wa_weights"])
        self.conf = {frozenset((CLASSES.index(a), CLASSES.index(b))) for a, b in self.cfg["confusable"]}

    def probs(self, Pexp):
        L = np.hstack([np.log(np.clip(Pexp[s], 1e-6, 1)) for s in EXPERTS])[None]
        pc = np.mean([m.predict_proba(L) for m in self.models], 0)
        pw = softmax(sum(self.w[j] * L[:, K * j:K * j + K] for j in range(len(EXPERTS))))
        return (0.5 * pc + 0.5 * pw)[0], L

    def shares(self, L, c):
        """Signed contribution of each expert to emotion c in the CatBoost half (mean SHAP over seeds)."""
        from catboost import Pool
        sv = np.mean([m.get_feature_importance(Pool(L), type="ShapValues")[0, c, :-1] for m in self.models], 0)
        return {s: float(sv[K * j:K * j + K].sum()) for j, s in enumerate(EXPERTS)}


class Pipeline2:
    def __init__(self, experts, vlm, slm, landmarker=None, owl=None, log=print):
        self.ex, self.vlm, self.slm, self.log = experts, vlm, slm, log
        owl = owl or os.path.join(ROOT, "ontology", "xfer_emonto.owl")
        try:
            ONT.use_owl(owl)
            self.knowledge = "OWL ontology " + os.path.basename(owl)
        except Exception as e:
            self.knowledge = f"xfer/ontology.py ({e.__class__.__name__} reading the OWL file)"
        log("knowledge read from: " + self.knowledge)
        self.final = Final()
        c = self.final.cfg
        self.au_names, self.thr = c["au_names"], np.array(c["au_thr"])
        self.meas = dict(zip(self.au_names, c["au_measurable"]))
        self.finder = FaceFinder(landmarker) if landmarker else None

    # ---------------------------------------------------------------- stage 0
    def prepare(self, img):
        img = img.convert("RGB")
        note = ""
        if self.finder is None:
            return img.resize((100, 100), Image.BICUBIC), None, {}, "no landmark model: template regions used"
        if img.size != (100, 100):
            lm, _ = self.finder.landmarks(img)
            if lm is None:
                note = "no face found in the photo: the whole image is used"
                face = img.resize((100, 100), Image.BICUBIC)
            else:
                face = crop_face(img, lm)
        else:
            face = img
        lm, bs = self.finder.landmarks(face, work_size=256)
        if lm is None:
            note = (note + "; " if note else "") + "no landmarks on the face crop: template regions used"
        return face, lm, bs, note

    def action_units(self, bs, found):
        if not found:
            return {}, []
        sc = au_scores(pd.DataFrame([bs])).iloc[0]
        act = [a for a, t in zip(self.au_names, self.thr) if t > 0 and sc[a] >= t]
        return {a: float(sc[a] / (t if t > 0 else 1)) for a, t in zip(self.au_names, self.thr)}, act

    # ---------------------------------------------------------------- stage 1: occlusion
    def occlusion(self, s, face, masks, c):
        full = np.ones(R, int)
        Z = np.stack([full, np.zeros(R, int)] + [full - np.eye(R, dtype=int)[r] for r in range(R)])
        P = self.ex[s].predict_proba_images(perturb(face, masks, Z))
        P_full, P_none, P_drop = P[0], P[1], P[2:]
        drop = np.clip(P_full[c] - P_drop[:, c], 0, None)
        top = int(np.argmax(drop))
        z = np.zeros(R, int); z[top] = 1
        ins = self.ex[s].predict_proba_images(perturb(face, masks, z[None]))[0]
        dn = np.clip((P_full[c] - P_drop[top, c]) / max(P_full[c], 1e-6), 0, 1)
        span = P_full[c] - P_none[c]
        inn = np.clip((ins[c] - P_none[c]) / span, 0, 1) if span > 0.02 else 0.5
        w = drop / (drop.sum() or 1)
        return dict(weights=w, faith=float(0.5 * (dn + inn)), top=FINE[top])

    # ---------------------------------------------------------------- the whole analysis
    def analyse(self, img, explain_vlm=True, progress=None):
        say = progress or (lambda *_: None)
        try:   # the web server runs each request in a new thread, whose current GPU is 0: POSTER++, CLIP,
            import torch   # the SVM twin and the SLM were loaded on GPU 1 (the VLM is pinned to GPU 0 by itself)
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
        except ImportError:
            pass
        t0 = time.time()
        res = dict(times={})
        say("Step 1 of 6: finding the face and its landmarks")
        face, lm, bs, note = self.prepare(img)
        found = lm is not None
        masks = fine_region_masks(lm if found else None, size=100)
        au_level, active = self.action_units(bs, found)
        res.update(face=face, landmarks=lm, found=found, note=note, masks=masks, au_level=au_level, active=active,
                   geometry=geometry(lm) if found else {})
        say("Step 2 of 6: the four experts")
        P = {s: self.ex[s].predict_proba_images([face])[0] for s in EXPERTS}
        res["P"] = P
        res["times"]["experts"] = time.time() - t0
        say("Step 3 of 6: the VLM's description and the SLM compensator")
        from .sources.vlm_zs import parse_reply
        text = self.vlm.describe(face)
        desc = parse_reply(text)
        n, v, items = check_claims(desc, dict.fromkeys(active, True), self.meas)
        res["vlm"] = dict(text=text, desc=desc, claims=items, n=n, verified=v, cons=(v / n if n else float("nan")))
        res["slm"] = self.slm.run(P["V2"], desc, active)
        say("Step 4 of 6: explanations (native and occlusion)")
        t1 = time.time()
        res["native"], res["occ"] = {}, {}
        for s in EXPERTS:
            c = int(P[s].argmax())
            if s == "V2" and not explain_vlm:
                res["native"][s] = np.full(R, np.nan); res["occ"][s] = None
                continue
            try:
                res["native"][s] = self.ex[s].native(face, masks, c)
            except Exception as e:
                self.log(f"native explanation of {s} failed: {e}")
                res["native"][s] = np.full(R, np.nan)
            res["occ"][s] = self.occlusion(s, face, masks, c)
        res["times"]["explanations"] = time.time() - t1
        say("Step 5 of 6: fusing the evidence")
        F, L = self.final.probs(P)
        o = np.argsort(-F)
        c1, c2 = int(o[0]), int(o[1])
        sh = self.final.shares(L, c1)
        res.update(F=F, c1=c1, c2=c2, shares=sh)
        tops = {s: int(P[s].argmax()) for s in EXPERTS}
        pos = {s: max(sh[s], 0.0) for s in EXPERTS}
        tot = sum(pos.values()) or 1
        res["support"] = {s: ("support" if tops[s] == c1 else "conflict") for s in EXPERTS}
        res["agreement"] = sum(pos[s] for s in EXPERTS if tops[s] == c1) / tot
        res["outliers"] = [s for s in EXPERTS if tops[s] != c1 and res["occ"].get(s) and res["occ"][s]["faith"] < 0.3]
        cons = np.zeros(R)
        for s in EXPERTS:
            if res["occ"].get(s) and tops[s] == c1:
                cons += pos[s] * res["occ"][s]["weights"]
        res["consensus"] = cons / (cons.sum() or 1)
        say("Step 6 of 6: ontology reasoning and decision")
        res.update(self.reason(res))
        res["times"]["total"] = time.time() - t0
        return res

    # ---------------------------------------------------------------- stage 3
    def reason(self, res):
        F, c1, c2 = res["F"], res["c1"], res["c2"]
        cfg = self.final.cfg
        tau, delta = cfg["tau"], cfg["delta"]
        e1, e2 = CLASSES[c1], CLASSES[c2]
        rules = []
        expected = [ONT.SUBSTITUTE.get(a, a) if not self.meas.get(a, True) else a for a in ONT.AU_CONFIG[e1]]
        expected = list(dict.fromkeys(expected))
        present = [a for a in expected if a in res["active"]]
        if not res["found"]:
            rules.append("AU pattern: no landmarks, so no action-unit evidence")
        elif e1 == "neutral":
            rules.append("AU pattern: neutral expects no action unit; " +
                         ("none is active" if not res["active"] else f"active: {', '.join(res['active'])}"))
        else:
            rules.append(f"AU pattern: {len(present)} of {len(expected)} action units of {e1} are active"
                         + (f" ({', '.join(present)})" if present else ""))
        cc = res["vlm"]["cons"]
        if not np.isnan(cc) and cc < 0.5:
            rules.append(f"precedence: geometry over semantics, the VLM's claims are only {cc:.0%} verified by the "
                         f"measurements")
        elif not np.isnan(cc):
            rules.append(f"precedence: the VLM's claims agree with the measurements ({cc:.0%} verified)")
        if res["outliers"]:
            rules.append("outlier: " + ", ".join(NAMES[s] for s in res["outliers"]) + " disagree with weak faithfulness")
        pair = frozenset((c1, c2)) in self.final.conf
        if F[c1] >= tau and F[c1] - F[c2] >= delta:
            kind = 0
            rules.append(f"decide: single label (top score {F[c1]:.2f} >= {tau:.2f}, margin {F[c1] - F[c2]:.2f} "
                         f">= {delta:.2f})")
        elif pair and F[c1] + F[c2] >= tau:
            kind = 1
            rules.append(f"confusability: {e1} and {e2} are a confusable pair in the ontology, so a compound label")
        else:
            kind = 2
            rules.append("decide: evidence too weak or split, refer the image to a person")
        return dict(kind=kind, rules=rules, expected=expected, present=present)


def load_experts(a, log=print):
    """Real experts on two GPUs (VLM on the first, the rest on the second) or fakes with --mock."""
    if a.mock:
        ex = {s: Fake(i, 4 + i) for i, s in enumerate(EXPERTS)}
        ex["V2"] = FakeVlm(7)
        return ex, ex["V2"], Slm(None, mock=True)
    import torch
    n = torch.cuda.device_count()
    log(f"{n} GPU(s) visible")
    ex = {}
    log("loading InternVL3-8B in 4-bit (V2)")
    vlm = VlmExpert(a.vlm_model)
    ex["V2"] = vlm
    if n > 1:
        torch.cuda.set_device(1)
    log("loading POSTER++ (B)")
    ex["B"] = PosterExpert(a.work, a.ckpt)
    log("loading CLIP zero-shot (A) and fine-tuned (A2)")
    clip = ClipExperts(a.a2)
    ex["A2"] = Proxy(lambda ims: clip.proba(ims, "A2"), lambda im, m, c: clip.rollout(im, m, c, "A2"))
    ex["A"] = Proxy(lambda ims: clip.proba(ims, "A"), lambda im, m, c: clip.rollout(im, m, c, "A"))
    log("loading the SVM (C) and fitting its linear twin for exact weights")
    ex["C"] = SvmExpert(a.svm, a.out, a.cache)
    if a.lora:
        log("loading the SLM compensator (Qwen2.5-1.5B + LoRA)")
    else:
        log("SLM adapter not given: the SLM check will be skipped (the final result does not use it)")
    slm = Slm(a.lora)
    return ex, vlm, slm
