"""The whole framework applied to one face image, for the demonstration app (scripts/17_demo_app.py).

Steps for an uploaded photograph:
  1. find the face with MediaPipe and crop it to a 100 x 100 face, like the aligned RAF-DB crops;
  2. landmarks, the seven facial regions, 8 geometric measures and 52 blendshapes on the crop (source D);
  3. predictions of V (LLaVA-1.5), A (CLIP), B (POSTER++) and C (SVM), temperature-scaled as in the experiments;
  4. region SHAP and LIME explanations (32 masked copies per source; V is optional because it takes ~16 s);
  5. the compensator: detector and corrector trained on the stored training-split evidence (structured-evidence
     arm, full input), exactly as in step 12;
  6. the ontology reasoner (rules read from the OWL file), with and without the compensated V: label, blend or abstain,
     consistency score and justification trace.
Every number comes from the trained models and the parameters fitted in the experiments; nothing is fitted on the
uploaded image.
"""
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

from . import ontology as ONT
from .compensator import features, gate_threshold, load_split
from .explain import explain_image, perturb
from .io import pred_path, read_predictions
from .labels import CLASSES, K, prob_columns
from .reasoner import (SOURCES, au_scores, au_thresholds, brackets, consistency, d_support, decide, region_support,
                       score, trace, translate)
from .regions import GEOMETRY, REGIONS, geometry, region_map, region_masks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAMES = {"V": "LLaVA-1.5 (V)", "A": "CLIP (A)", "B": "POSTER++ (B)", "C": "HOG/LBP + SVM (C)"}
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                  "face_landmarker.task")


# ----------------------------------------------------------------------------- calibration
def fit_temperatures(out, sources=SOURCES):
    """Temperature per source, fitted on the validation predictions exactly as in step 8."""
    from scipy.optimize import minimize
    from scipy.special import log_softmax
    T = {}
    for s in sources:
        df = read_predictions(pred_path(out, s, "val"))
        z = np.log(np.clip(df[prob_columns()].values, 1e-8, 1))
        y = df.label.values.astype(int)
        nll = lambda t: -log_softmax(z / np.exp(t[0]), axis=1)[np.arange(len(y)), y].mean()
        T[s] = float(np.exp(minimize(nll, [0.0], method="L-BFGS-B").x[0]))
    return T


def temper(p, T):
    z = np.log(np.clip(p, 1e-8, 1)) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------------- compensator
def train_compensator(out, block="pred+expl+face", seed=42):
    """Detector and corrector of the structured-evidence arm, trained as in step 12 (boosted trees were chosen on
    validation for every input block)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                l2_regularization=1.0, class_weight="balanced", random_state=seed)
    tr_ids = pd.read_csv(os.path.join(out, "explanations", "V_train.csv")).image_id.astype(str).sort_values().values
    D = {"train": load_split(out, "train", tr_ids), "val": load_split(out, "val")}
    X = {s: features(d, block) for s, d in D.items()}
    z = {s: (d["pred"] != d["y"]).astype(int) for s, d in D.items()}
    det = mk().fit(X["train"], z["train"])
    f = z["train"] == 1
    cor = mk().fit(X["train"][f], D["train"]["y"][f])

    def cprob(Xm):
        p = np.zeros((len(Xm), K))
        p[:, cor.classes_] = cor.predict_proba(Xm)
        return p
    thr = gate_threshold(det.predict_proba(X["val"])[:, 1], D["val"]["pred"], cprob(X["val"]).argmax(1), D["val"]["y"])
    return dict(det=det, cprob=cprob, thr=thr, block=block)


# ----------------------------------------------------------------------------- face finding
class FaceFinder:
    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                            output_face_blendshapes=True, num_faces=1,
                                            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
        self.det = vision.FaceLandmarker.create_from_options(opts)

    def landmarks(self, img, work_size=None):
        """(478, 2) pixel coordinates in img, and blendshape scores; (None, {}) if no face."""
        w, h = img.size
        x = img if work_size is None else img.resize((work_size, work_size), Image.BICUBIC)
        res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                            data=np.ascontiguousarray(np.asarray(x.convert("RGB")))))
        if not res.face_landmarks:
            return None, {}
        lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        return lm, bs


def crop_face(img, lm, scale=1.2, size=100):
    """Square crop around the landmarks, like the aligned RAF-DB faces."""
    x0, y0 = lm.min(0)
    x1, y1 = lm.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.04 * (y1 - y0)
    side = scale * max(x1 - x0, y1 - y0)
    a = np.asarray(img.convert("RGB"))
    pad = int(side)
    a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    l, t = int(cx - side / 2) + pad, int(cy - side / 2) + pad
    c = Image.fromarray(a[t:t + int(side), l:l + int(side)])
    return c.resize((size, size), Image.BICUBIC)


# ----------------------------------------------------------------------------- the pipeline
class Pipeline:
    def __init__(self, out, predictors, landmarker=None, owl=None, log=print):
        """out: the experiment outputs folder (predictions, explanations, results/D_geometry.csv);
        predictors: dict source -> object with predict_proba_images(list of PIL) -> (n, 7)."""
        self.out, self.models, self.log = out, predictors, log
        owl = owl or os.path.join(ROOT, "ontology", "xfer_emonto.owl")
        try:
            ONT.use_owl(owl)
            self.knowledge = "OWL ontology " + os.path.basename(owl)
        except Exception as e:                                   # owlready2 missing: same knowledge from Python
            self.knowledge = f"xfer/ontology.py ({e.__class__.__name__} reading the OWL file)"
        log("knowledge read from: " + self.knowledge)
        self.params = json.load(open(os.path.join(HERE, "demo_params.json")))
        self.T = fit_temperatures(out, [s for s in SOURCES if s in predictors])
        log("temperatures: " + ", ".join(f"{s} {t:.2f}" for s, t in self.T.items()))
        self.thr, self.meas = au_thresholds(out)
        self.comp = train_compensator(out)
        self.tree = train_surrogate(out)
        if self.tree:
            log(f"surrogate tree of LLaVA-1.5 trained (test fidelity {self.tree['fidelity']:.3f})")
        log(f"compensator trained; detector threshold {self.comp['thr']:.3f}")
        self.finder = None
        if landmarker:
            self.finder = FaceFinder(landmarker)

    # -- one image
    def prepare(self, img):
        """Returns the 100 x 100 face, its landmarks (100-pixel coordinates) and blendshapes, and a note."""
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

    def analyse(self, img, explain_v=True, progress=None):
        say = progress or (lambda *_: None)
        say("Step 1 of 7: finding the face")
        face, lm, bs, note = self.prepare(img)
        found = lm is not None
        masks = region_masks(lm if found else None)
        ev = dict(ids=np.array(["demo"]), y=np.zeros(1, int), found=np.array([found]))
        g = pd.DataFrame([bs]) if bs else pd.DataFrame([{}])
        for grp in ONT.AU_BLENDSHAPES.values():
            for names in grp:
                for c in names:
                    if c not in g.columns:
                        g[c] = np.nan
        ev["au"] = au_scores(g).values
        expl, raw = {}, {}
        for s in SOURCES:
            if s not in self.models:
                continue
            say(f"Step 2 of 7: asking {NAMES[s]}")
            p = np.asarray(self.models[s].predict_proba_images([face]), float)[0]
            raw[s] = p
            ev[f"P_{s}"] = temper(p, self.T[s])[None]
            if s == "V" and not explain_v:
                phi = np.full(len(REGIONS), np.nan)
                e = dict(phi=phi, lime_coef=phi, spearman=np.nan, target=int(p.argmax()))
            else:
                say(f"Step 3 of 7: explaining {NAMES[s]} (32 blurred copies)")
                bs_ = 8 if s == "V" else 32                       # LLaVA in batches of 8, as in step 9
                mdl = self.models[s]
                fn = lambda ims, mdl=mdl, b=bs_: np.concatenate([mdl.predict_proba_images(ims[i:i + b])
                                                                 for i in range(0, len(ims), b)])
                e = explain_image(face, masks, fn, "demo")
            expl[s] = e
            ev[f"phi_{s}"] = np.asarray(e["phi"], float)[None]
            ev[f"target_{s}"] = np.array([e["target"]])
        say("Steps 4 to 7: measuring the face, checking LLaVA, applying the ontology, deciding")
        res = dict(face=face, landmarks=lm, found=found, note=note, raw=raw, expl=expl, blendshapes=bs,
                   geometry=geometry(lm) if found else {k: np.nan for k in GEOMETRY})
        res["without"] = self._reason(ev, self.params["reasoner"])
        # compensator on V
        sup, _, pred = region_support(ev, "V")
        d = dict(P=ev["P_V"], pred=pred, shap=ev["phi_V"], lime=np.asarray(expl["V"]["lime_coef"], float)[None],
                 spearman=np.array([expl["V"]["spearman"]], float), support=sup,
                 geo=np.array([[res["geometry"][k] for k in GEOMETRY]], float), au=ev["au"],
                 found=np.array([float(found)]))
        X = features(d, self.comp["block"])
        det = float(self.comp["det"].predict_proba(X)[0, 1])
        corr = int(self.comp["cprob"](X)[0].argmax())
        fired = det >= self.comp["thr"]
        v_after = corr if fired else int(pred[0])
        res["compensator"] = dict(detector=det, threshold=self.comp["thr"], fired=fired, v_before=int(pred[0]),
                                  v_after=v_after, corrector=corr)
        ev2 = dict(ev)
        ev2["P_V"] = np.eye(K)[[v_after]].astype(float)
        res["with"] = self._reason(ev2, self.params["full_framework"])
        res["ev"] = ev
        res["why"] = self.explain_decision(res, face, masks)
        return res

    def explain_decision(self, res, face, masks):
        """Plain evidence for the final decision: where on the face, what the ontology expects and finds, a
        faithfulness test (blur the top region and re-ask the model), SHAP-LIME agreement, and LLaVA's surrogate rule."""
        w, ev = res["with"], res["ev"]
        final = w["c1"]
        alpha = dict(zip(SOURCES, self.params["full_framework"]["alpha"][:4]))
        backers = [s for s in SOURCES if int(ev[f"P_{s}"][0].argmax()) == final and
                   not np.isnan(np.asarray(res["expl"][s]["phi"], float)).all()]
        use = backers or [s for s in SOURCES if not np.isnan(np.asarray(res["expl"][s]["phi"], float)).all()]
        comb = np.zeros(len(REGIONS))
        for s in use:
            phi = np.asarray(res["expl"][s]["phi"], float)
            comb += alpha[s] * phi / (np.abs(phi).sum() or 1)
        top = [REGIONS[i] for i in np.argsort(-comb)[:2] if comb[i] > 0]
        # ontology: what this class should show, and what the face shows
        aus = list(ONT.AU_BLENDSHAPES)
        act = np.nan_to_num(ev["au"][0]) / np.where(self.thr > 0, self.thr, 1)
        expected = []
        for au in ONT.AU_CONFIG[CLASSES[final]]:
            read = au if self.meas[aus.index(au)] else ONT.SUBSTITUTE.get(au, au)
            i = aus.index(read)
            expected.append(dict(au=au, read=read, measurable=bool(self.meas[i]), present=bool(act[i] >= 1),
                                 level=float(act[i])))
        # faithfulness: blur each supporting model's top region and ask it again (LLaVA is skipped: too slow);
        # report the model whose answer moves most, so a saturated model does not hide the test
        test = None
        cands = [s for s in (backers or SOURCES) if s != "V"] or [s for s in SOURCES if s != "V"]
        for s in cands:
            phi = np.asarray(res["expl"][s]["phi"], float)
            if np.isnan(phi).all():
                continue
            k = int(np.nanargmax(phi))
            z = np.ones((1, len(REGIONS)), int); z[0, k] = 0
            p_after = np.asarray(self.models[s].predict_proba_images(perturb(face, masks, z)), float)[0]
            p_before = res["raw"][s]
            cand = dict(source=s, region=REGIONS[k], cls=final, before=float(p_before[final]),
                        after=float(p_after[final]), new_top=int(p_after.argmax()))
            if test is None or (cand["before"] - cand["after"]) > (test["before"] - test["after"]):
                test = cand
        lime_agree = {s: float(res["expl"][s]["spearman"]) for s in SOURCES
                      if res["expl"][s].get("spearman") is not None and not np.isnan(res["expl"][s]["spearman"])}
        rule = surrogate_path(self.tree, res, ev) if self.tree else None
        return dict(final=final, sources=use, combined=comb, top=top, expected=expected, test=test,
                    lime_agree=lime_agree, rule=rule)

    def _reason(self, ev, p):
        tr = translate(ev, self.thr, self.meas, p["mode"])
        br = brackets(ev, tr, p["lam"])
        sc = score(br, np.array(p["alpha"]))
        kind, c1, c2 = decide(sc, p["tau"], p["delta"])
        cons = consistency(br, c1)
        return dict(scores=sc[0], kind=int(kind[0]), c1=int(c1[0]), c2=int(c2[0]), consistency=float(cons[0]),
                    support={s: float(tr[f"sup_{s}"][0]) for s in SOURCES}, gate={s: float(tr[f"gate_{s}"][0]) for s in SOURCES},
                    au_support=tr["qD"][0], trace=trace(0, ev, tr, br, sc, kind, c1, c2, np.array(p["alpha"])))


def verdict(r):
    if r["kind"] == 0:
        return f"{CLASSES[r['c1']].upper()}", "single label"
    if r["kind"] == 1:
        return f"{CLASSES[r['c1']].upper()} or {CLASSES[r['c2']].upper()}", "blended label (a confusable pair)"
    return "NOT SURE: refer to a person", "abstention"


# ----------------------------------------------------------------------------- surrogate tree (global explanation)
NICE = {"inner_brow_raise": "inner brows raised", "outer_brow_raise": "outer brows raised", "brow_gap": "gap between brows",
        "eye_opening": "eye opening", "mouth_opening": "mouth opening", "mouth_width": "mouth width",
        "lip_corner_lift": "lip corners lifted", "nose_to_lip": "nose-to-lip distance"}


def train_surrogate(out, depth=4):
    """The depth-4 surrogate tree of step 10 (geometry and V's region SHAP), fitted to copy LLaVA-1.5's answers."""
    from sklearn.tree import DecisionTreeClassifier

    def table(split):
        g = pd.read_csv(os.path.join(out, "results", "D_geometry.csv"))
        g["image_id"] = g.image_id.astype(str)
        g = g[(g.split == split) & g.found]
        v = read_predictions(pred_path(out, "V", split))[["image_id", "pred"]]
        v["image_id"] = v.image_id.astype(str)
        e = pd.read_csv(os.path.join(out, "explanations", f"V_{split}.csv"))
        e["image_id"] = e.image_id.astype(str)
        return g.merge(v, on="image_id").merge(e[["image_id"] + [f"shap_{r}" for r in REGIONS]], on="image_id")
    try:
        cols = list(GEOMETRY) + [f"shap_{r}" for r in REGIONS]
        tr, te = table("train"), table("test")
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=0).fit(tr[cols].values, tr.pred.values)
        return dict(tree=t, cols=cols, fidelity=float((t.predict(te[cols].values) == te.pred).mean()))
    except Exception as ex:                                    # explanations missing: no surrogate in the app
        print("surrogate tree not available:", ex)
        return None


def surrogate_path(tr, res, ev):
    t, cols = tr["tree"], tr["cols"]
    if not res["found"] or np.isnan(np.asarray(res["expl"]["V"]["phi"], float)).all():
        return dict(available=False, fidelity=tr["fidelity"])
    x = np.array([[res["geometry"][c] if c in GEOMETRY else
                   float(res["expl"]["V"]["phi"][REGIONS.index(c[5:])]) for c in cols]])
    node, conds = 0, []
    T = t.tree_
    while T.children_left[node] != -1:
        f, thr = T.feature[node], T.threshold[node]
        name = cols[f]
        label = NICE.get(name, name) if not name.startswith("shap_") else f"weight of the {name[5:]} in LLaVA's answer"
        if x[0, f] <= thr:
            conds.append(f"{label} at most {thr:.2f}"); node = T.children_left[node]
        else:
            conds.append(f"{label} above {thr:.2f}"); node = T.children_right[node]
    counts = T.value[node][0]
    k = int(np.argmax(counts))
    return dict(available=True, conditions=conds, predicts=int(t.classes_[k]), purity=float(counts[k] / counts.sum()),
                llava=int(np.argmax(res["raw"]["V"])), fidelity=tr["fidelity"])
