"""Step 17: the demonstration app. Upload a face (or use the webcam) and see every stage of the framework.

    python scripts/17_demo_app.py --out OUT --ckpt .../best.pth --work .../poster --share

Pages
  Analyse a face    any photo: the verdict (label, blended label or referral), which evidence agreed, what each
                    model saw and where it looked, the facial action units, the compensator's decision, the trace
  RAF-DB examples   test images with their stored decision; the live pipeline can be run on them (private showing
                    only: the RAF-DB licence does not allow the images to be published)
  How it works      the framework and its headline results
--mock replaces the four models with fake ones, to test the app without a GPU; --selftest IMG writes the result page
to selftest.html and exits.
"""
import argparse
import base64
import html
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image

from xfer import ontology as ONT
from xfer.demo import LANDMARKER_URL, Pipeline
from xfer.labels import CLASSES
from xfer.reasoner import SOURCES
from xfer.regions import REGIONS, region_map, region_masks

MODEL = {"V": ("LLaVA-1.5", "generative vision-language model, zero-shot"),
         "A": ("CLIP", "contrastive vision-language model, zero-shot"),
         "B": ("POSTER++", "transformer trained on RAF-DB"),
         "C": ("HOG + LBP with SVM", "classical features and classifier"),
         "D": ("Face measurements", "MediaPipe landmarks read as action units")}
AU_NAMES = {"AU1": "inner brow raised", "AU2": "outer brow raised", "AU4": "brows lowered", "AU5": "upper lids raised",
            "AU6": "cheeks raised", "AU7": "lids tightened", "AU9": "nose wrinkled", "AU10": "upper lip raised",
            "AU12": "smile", "AU15": "lip corners down", "AU17": "chin raised", "AU20": "lips stretched",
            "AU23": "lips pressed", "AU26": "jaw dropped"}


# ----------------------------------------------------------------------------- models
class Fake:
    """Deterministic stand-in for a model, for testing the app without a GPU."""
    def __init__(self, seed):
        self.W = np.random.default_rng(seed).normal(size=(3 * 16 * 16, 7))

    def predict_proba_images(self, imgs):
        x = np.stack([np.asarray(im.convert("RGB").resize((16, 16)), float).ravel() / 255 for im in imgs])
        z = (x - 0.5) @ self.W / 4
        ex = np.exp(z - z.max(1, keepdims=True))
        return ex / ex.sum(1, keepdims=True)


def load_models(a, log):
    if a.mock:
        return {s: Fake(i) for i, s in enumerate(SOURCES)}
    m = {}
    log("loading POSTER++ (B)")
    from xfer.sources.poster import PosterPredictor
    m["B"] = PosterPredictor(a.work, a.ckpt)
    log("loading the SVM (C)")
    from xfer.sources.hog_svm import SvmPredictor
    m["C"] = SvmPredictor(a.svm)
    log("loading CLIP ViT-L/14 (A)")
    from xfer.sources.clip_zs import ClipZeroShot
    m["A"] = ClipZeroShot()
    log("loading LLaVA-1.5-7B in 4-bit (V); the first time this downloads about 14 GB")
    from xfer.sources.llava_zs import LlavaZeroShot
    m["V"] = LlavaZeroShot()
    return {s: m[s] for s in SOURCES}


# ----------------------------------------------------------------------------- images
def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heat_img(res, s, size=200):
    face = np.asarray(res["face"].convert("L"), float) / 255
    base = np.stack([face] * 3, -1) * 0.85 + 0.1
    phi = np.asarray(res["expl"][s]["phi"], float)
    if np.isnan(phi).all():
        return Image.fromarray((base * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), None
    masks = region_masks(res["landmarks"] if res["found"] else None)
    v = np.nanmax(np.abs(phi)) or 1
    out = base.copy()
    cmap = matplotlib.colormaps["RdBu_r"]
    for k in range(len(REGIONS)):
        c = np.array(cmap(0.5 + 0.5 * phi[k] / v)[:3])
        a = 0.62 * min(1, abs(phi[k]) / v) + 0.08
        out[masks[k]] = out[masks[k]] * (1 - a) + c * a
    img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    order = np.argsort(-phi)
    top = [REGIONS[i] for i in order[:2] if phi[i] > 0]
    return img, top


def combined_img(res, comb, size=220):
    face = np.asarray(res["face"].convert("L"), float) / 255
    out = np.stack([face] * 3, -1) * 0.85 + 0.1
    masks = region_masks(res["landmarks"] if res["found"] else None)
    v = np.abs(comb).max() or 1
    cmap = matplotlib.colormaps["RdBu_r"]
    for k in range(len(REGIONS)):
        c = np.array(cmap(0.5 + 0.5 * comb[k] / v)[:3])
        a = 0.65 * min(1, abs(comb[k]) / v) + 0.05
        out[masks[k]] = out[masks[k]] * (1 - a) + c * a
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)


PAL = ["#9fb7d9", "#e3c26b", "#b9a3d6", "#e39b8c", "#8cc7b5", "#e8a96a", "#a9c47f"]


def regions_img(res, size=240):
    face = np.asarray(res["face"].convert("RGB"), float) / 255
    lab = region_map(res["landmarks"] if res["found"] else None)
    out = face.copy()
    for k, c in enumerate(PAL):
        out[lab == k] = out[lab == k] * 0.55 + np.array(mcolors.to_rgb(c)) * 0.45
    return Image.fromarray((out * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)


# ----------------------------------------------------------------------------- page
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Public+Sans:wght@400;500;600&display=swap');
:root, .gradio-container {
  --paper:#eef1f4; --surface:#ffffff; --ink:#172231; --muted:#5d6a79; --rule:#dde3ea;
  --label:#1e6b52; --blend:#9a6a12; --abstain:#6a4a9c; --focus:#2f5fb3;
}
body, .gradio-container { background: var(--paper) !important; color: var(--ink); font-family:'Public Sans', system-ui, sans-serif; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display:none !important; }
#masthead { padding: 28px 4px 16px; border-bottom: 1px solid var(--rule); margin-bottom: 8px; display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap }
#masthead h1 { font-family:'Newsreader', Georgia, serif; font-weight:500; font-size: 30px; line-height:1.2; max-width: 46ch; margin:0; color: var(--ink) }
#masthead p { margin:10px 0 0; color:var(--muted); font-size:15px; max-width: 70ch; }
#masthead .status { font-size: 13px; color: var(--muted); text-align:left }
.panel { background: var(--surface); border:1px solid var(--rule); border-radius: 10px; }
#analyse { background: var(--ink) !important; color:#fff !important; border:none !important; font-weight:600; font-size:16px; border-radius: 8px !important; min-height: 46px }
#analyse:hover { background:#24344a !important }
#analyse:focus-visible, button:focus-visible, summary:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
.hint { color: var(--muted); font-size: 13.5px; line-height: 1.5; margin: 2px 4px; }
.empty { padding: 44px 40px; color: var(--muted); font-size: 15px; line-height:1.6 }
.empty h2 { font-family:'Newsreader', Georgia, serif; font-weight:500; color:var(--ink); font-size:27px; margin:0 0 8px; }

.result { display:grid; grid-template-columns: 1fr 210px; gap: 28px; padding: 26px 30px; border-left: 8px solid var(--c); }
.result .kind { font-size: 14px; color: var(--c); font-weight:600; }
.result .word { font-family:'Newsreader', Georgia, serif; font-weight:500; font-size: 52px; line-height:1.05; margin: 4px 0 10px; }
.result .word .or { font-style: italic; color: var(--muted); font-size: 0.55em; padding: 0 0.2em; }
.result .sent { font-size: 16px; line-height: 1.5; margin: 0 0 18px; max-width: 52ch }
.result figure { margin: 0 }
.result figure img { width: 210px; height: 210px; border-radius: 8px; display:block }
.result figcaption { font-size: 12.5px; color: var(--muted); margin-top: 6px; line-height:1.4 }
.checks { list-style:none; margin: 0; padding: 0; display:grid; gap: 10px }
.checks li { display:grid; grid-template-columns: 118px 1fr; gap: 12px; font-size: 15px; line-height: 1.45; align-items: baseline }
.checks .k { color: var(--muted); font-size: 13.5px }
.ok { color: var(--label); font-weight: 600 }
.miss { color: #9b3b2f; font-weight: 600 }
.dots { display:flex; gap: 6px; align-items:center; margin-top: 18px; font-size: 13px; color: var(--muted) }
.dot { width: 24px; height: 24px; border-radius: 50%; display:inline-grid; place-items:center; font-weight:600; font-size: 11.5px; border: 1.5px solid var(--c); color: var(--c); background: #fff }
.dot.on { background: var(--c); color: #fff }
.dots .txt { margin-left: 6px }

details.more { margin-top: 14px; }
details.more > summary { cursor: pointer; list-style: none; padding: 15px 22px; font-weight: 600; font-size: 15px; display:flex; align-items: baseline }
details.more > summary::-webkit-details-marker { display:none }
details.more > summary::after { content: '+'; color: var(--muted); font-weight: 400; font-size: 20px; line-height: 1; margin-left: auto }
details.more[open] > summary::after { content: '\\2212' }
details.more > summary span { font-weight: 400; color: var(--muted); font-size: 13.5px; margin-left: 10px }
.inner { padding: 0 22px 20px; }
.grid4 { display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px }
.grid4 figure { margin: 0 }
.grid4 img { width: 100%; aspect-ratio: 1; border-radius: 6px; display:block }
.grid4 figcaption { font-size: 13px; line-height: 1.45; margin-top: 8px }
.grid4 figcaption b { display:block; font-size: 14px }
.grid4 figcaption span { color: var(--muted) }
.legend { display:flex; gap: 10px; align-items:center; font-size: 12.5px; color: var(--muted); margin-top: 14px }
.legend .ramp { width: 110px; height: 8px; border-radius: 4px; background: linear-gradient(90deg,#2166ac,#f7f7f7,#b2182b) }
.plain { font-size: 15px; line-height: 1.6; margin: 0 0 10px; max-width: 75ch }
.small { font-size: 13.5px; color: var(--muted); line-height: 1.55 }
.chips { display:flex; flex-wrap: wrap; gap: 6px; margin-top: 8px }
.chip { font-size: 12.5px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--rule); color: var(--muted); background:#fff }
.chip.on { border-color: var(--ink); color: var(--ink); background: #e6ecf4; font-weight:600 }
table.scores { width:100%; border-collapse: collapse; font-size: 13.5px; font-variant-numeric: tabular-nums; margin-top: 4px }
table.scores th { text-align:right; font-weight:600; padding: 8px 10px; border-bottom: 1px solid var(--rule); color: var(--muted); font-size: 12.5px }
table.scores th:first-child, table.scores td:first-child { text-align:left }
table.scores td { text-align:right; padding: 6px 10px; border-bottom: 1px solid #eef1f4 }
table.scores td.final { font-weight:600 }
pre.trace { white-space: pre-wrap; font-size: 13.5px; line-height: 1.65; margin: 14px 0 0; font-family: 'Public Sans', sans-serif; background: #f6f8fa; border-radius: 6px; padding: 12px 14px }
.howto { padding: 28px 32px; font-size: 16px; line-height: 1.7; }
.howto h2 { font-family:'Newsreader', Georgia, serif; font-weight:500; font-size: 27px; margin: 0 0 10px }
.howto p { max-width: 72ch }
.howto table { border-collapse: collapse; margin: 16px 0; font-size: 15px }
.howto td { padding: 8px 18px 8px 0; border-bottom: 1px solid var(--rule) }
.howto td:last-child { font-variant-numeric: tabular-nums; font-weight: 600 }
@media (max-width: 760px) {
  .result { grid-template-columns: 1fr; padding: 20px }
  .result .word { font-size: 40px }
  .checks li { grid-template-columns: 1fr; gap: 2px }
  .grid4 { grid-template-columns: 1fr 1fr }
  #masthead h1 { font-size: 24px }
  #masthead .status { text-align:left }
}
"""

KIND = {0: ("Single label", "var(--label)"),
        1: ("Blended label: two expressions this face does not separate", "var(--blend)"),
        2: ("Not sure, so the face is referred to a person", "var(--abstain)")}

EMPTY = """
<div class="panel empty">
  <h2>Add a face to see the result and its explanation</h2>
  The system names the expression, or says it is not sure, and shows why: where on the face the evidence is, which
  facial movements it found, and a test that the explanation is faithful.
</div>"""


def e(s):
    return html.escape(str(s))


def cap(s):
    return s[:1].upper() + s[1:]


STEP_CSS = """
.steps-h { font-family:'Newsreader', Georgia, serif; font-weight:500; font-size: 23px; margin: 26px 0 4px 2px }
.steps-sub { color: var(--muted); font-size: 14px; margin: 0 0 12px 2px }
.steps { position: relative }
details.step { position: relative; margin: 0 0 10px 0; padding-left: 52px }
details.step::before { content:''; position:absolute; left: 17px; top: 40px; bottom: -12px; width: 2px; background: var(--rule) }
details.step:last-child::before { display:none }
details.step > summary { list-style:none; cursor:pointer; background: var(--surface); border:1px solid var(--rule); border-radius: 10px; padding: 13px 18px; display:flex; align-items: baseline; gap: 10px }
details.step > summary::-webkit-details-marker { display:none }
details.step > summary::after { content: '+'; color: var(--muted); font-size: 19px; line-height: 1; margin-left: auto }
details.step[open] > summary::after { content: '\\2212' }
details.step[open] > summary { border-bottom-left-radius: 0; border-bottom-right-radius: 0; border-bottom-color: transparent }
.num { position:absolute; left: 0; top: 7px; width: 36px; height: 36px; border-radius: 50%; background: var(--ink); color:#fff; display:grid; place-items:center; font-family:'Newsreader', Georgia, serif; font-size: 18px }
.step .t { font-weight: 600; font-size: 15px; white-space: nowrap }
.step .s { color: var(--muted); font-size: 14px }
.step .body { background: var(--surface); border:1px solid var(--rule); border-top: none; border-radius: 0 0 10px 10px; padding: 4px 18px 18px }
.row { display:flex; gap: 16px; align-items:flex-start; flex-wrap: wrap }
.row img.face { width: 120px; height: 120px; border-radius: 6px }
.reg { display:grid; grid-template-columns: 1fr 1fr; gap: 4px 14px; font-size: 12.5px; color: var(--muted) }
.reg span::before { content:''; display:inline-block; width: 10px; height: 10px; border-radius: 2px; background: var(--sw); margin-right: 6px; vertical-align: -1px }
table.mini { border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums }
table.mini td { padding: 6px 18px 6px 0; border-bottom: 1px solid #eef1f4 }
table.mini td:first-child { font-weight: 600 }
.meter { height: 8px; background:#edf0f3; border-radius: 4px; position: relative; margin: 12px 0 6px; max-width: 420px }
.meter .lvl { position:absolute; left:0; top:0; bottom:0; background: var(--ink); border-radius: 4px }
.meter .thr { position:absolute; top:-5px; bottom:-5px; width:3px; border-radius:2px; background: var(--abstain) }
@media (max-width: 760px) { .step .t { white-space: normal } .step .s { display:none } }
"""


def step(n, title, summary, body, open_=False):
    return (f'<details class="step"{" open" if open_ else ""}><summary><span class="num">{n}</span><span class="t">'
            f'{title}</span><span class="s">{summary}</span></summary><div class="body">{body}</div></details>')


def render(res, pipe, elapsed=None):
    w, y, ev = res["with"], res["why"], res["ev"]
    kind_text, col = KIND[w["kind"]]
    final, second = w["c1"], w["c2"]
    if w["kind"] == 0:
        word = e(cap(CLASSES[final]))
        sent = f'The models and the face measurements point to {e(CLASSES[final])}.'
    elif w["kind"] == 1:
        word = f'{e(cap(CLASSES[final]))}<span class="or">or</span>{e(CLASSES[second])}'
        sent = 'These two score close together, and faces often show them in the same way, so both are given.'
    else:
        word = "Refer to a person"
        sent = f'The evidence conflicts between {e(CLASSES[final])} and {e(CLASSES[second])}, so the system does not guess.'
    c = res["compensator"]
    if c["fired"]:
        sent += f' LLaVA-1.5\'s answer was corrected from {e(CLASSES[c["v_before"]])} to {e(CLASSES[c["v_after"]])}.'

    checks = []
    if y["top"]:
        checks.append(f'<li><span class="k">Where it looked</span><span>The {" and ".join(e(t) for t in y["top"])}</span></li>')
    if y["expected"] and res["found"]:
        parts = [(f'<span class="ok">{e(AU_NAMES.get(x["au"], x["au"]))}</span>' if x["present"] else
                  f'<span class="miss">no {e(AU_NAMES.get(x["au"], x["au"]))}</span>') for x in y["expected"] if x["measurable"]]
        n_ok = sum(x["present"] for x in y["expected"])
        checks.append(f'<li><span class="k">Facial movements</span><span>{n_ok} of {len(y["expected"])} expected for '
                      f'{e(CLASSES[final])}: {", ".join(parts)}</span></li>')
    elif not res["found"]:
        checks.append('<li><span class="k">Facial movements</span><span>Not measured: no face landmarks found</span></li>')
    t = y["test"]
    if t:
        drop = t["before"] - t["after"]
        verb = ("drops" if drop >= 0.03 else "barely changes")
        checks.append(f'<li><span class="k">Explanation test</span><span>Hiding the {e(t["region"])} {verb} '
                      f'{e(MODEL[t["source"]][0])}\'s confidence ({100 * t["before"]:.0f}% to {100 * t["after"]:.0f}%)'
                      + ('' if drop >= 0.03 else ': the answer rests on several regions together') + '</span></li>')
    own = {s: int(ev[f"P_{s}"][0].argmax()) for s in SOURCES}
    if c["fired"]:
        own["V"] = c["v_after"]
    own["D"] = int(np.argmax(w["au_support"])) if res["found"] else None
    agree = [s for s in "VABCD" if own[s] == final]
    dots = "".join(f'<span class="dot{" on" if s in agree else ""}" title="{e(MODEL[s][0])}">{s}</span>' for s in "VABCD")
    out = [f'<div class="panel result" style="--c:{col}"><div><div class="kind">{kind_text}</div><div class="word">{word}'
           f'</div><p class="sent">{sent}</p><ul class="checks">{"".join(checks)}</ul><div class="dots">{dots}'
           f'<span class="txt">{len(agree)} of 5 sources agree</span></div></div><figure><img alt="Explanation heatmap" '
           f'src="{b64(combined_img(res, y["combined"]))}"><figcaption>Red: regions behind the answer. Blue: regions '
           f'against it.</figcaption></figure></div>']

    # ---------------------------------------------------------------- step by step
    S = []
    legend = "".join(f'<span style="--sw:{c_}">{e(cap(r))}</span>' for r, c_ in zip(REGIONS, PAL))
    found_txt = ("Face found, cropped to 100 x 100 and given 478 landmarks" if res["found"] else
                 "No landmarks found: a fixed template of regions is used")
    S.append(step(1, "Find the face", e(found_txt),
                  f'<div class="row"><img class="face" alt="The face with its seven regions" src="{b64(regions_img(res))}">'
                  f'<div><p class="plain">The face is located with MediaPipe, cropped like the RAF-DB training faces, and '
                  f'divided into seven regions. Every explanation below is given in terms of these regions.</p>'
                  f'<div class="reg">{legend}</div></div></div>'))

    rows = "".join(f'<tr><td>{s}. {e(MODEL[s][0])}</td><td>{e(CLASSES[int(ev[f"P_{s}"][0].argmax())])}</td>'
                   f'<td>{100 * ev[f"P_{s}"][0].max():.0f}%</td><td class="small">{e(MODEL[s][1])}</td></tr>'
                   for s in SOURCES)
    summ2 = ", ".join(f'{s}: {e(CLASSES[int(ev[f"P_{s}"][0].argmax())])}' for s in SOURCES)
    S.append(step(2, "Ask four different models", summ2,
                  f'<p class="plain">Four recognisers built in different ways give their answer, with calibrated '
                  f'confidence.</p><table class="mini">{rows}</table>'))

    figs = []
    for s in SOURCES:
        img, top = heat_img(res, s)
        look = ("explanation skipped" if top is None else "looked at the " + " and ".join(top) if top else
                "no region stood out")
        figs.append(f'<figure><img alt="{e(MODEL[s][0])} explanation" src="{b64(img)}"><figcaption><b>{s}. '
                    f'{e(MODEL[s][0])}</b><span>{e(look)}</span></figcaption></figure>')
    agree_txt = ", ".join(f'{e(MODEL[s][0])} {v:.2f}' for s, v in y["lime_agree"].items())
    test_txt = ""
    if t:
        test_txt = (f' To test the explanation, the {e(t["region"])} region was hidden and {e(MODEL[t["source"]][0])} asked '
                    f'again: its confidence in {e(CLASSES[t["cls"]])} went from {100 * t["before"]:.0f}% to '
                    f'{100 * t["after"]:.0f}%.')
    S.append(step(3, "Explain each model", "where each model looked (SHAP and LIME)",
                  f'<p class="plain">Each region is blurred in turn and the change in the model\'s answer is measured '
                  f'(SHAP). LIME, computed from the same blurred copies, ranks the regions alike: {agree_txt} '
                  f'(1 = identical).{test_txt}</p><div class="grid4">{"".join(figs)}</div><div class="legend"><span>'
                  f'Against the answer</span><span class="ramp"></span><span>Towards it</span></div>'))

    act = np.nan_to_num(ev["au"][0]) / np.where(pipe.thr > 0, pipe.thr, 1)
    on = [AU_NAMES.get(au, au) for i, au in enumerate(ONT.AU_BLENDSHAPES) if pipe.meas[i] and act[i] >= 1]
    chips = "".join(f'<span class="chip{" on" if (pipe.meas[i] and act[i] >= 1) else ""}">'
                    f'{e(cap(AU_NAMES.get(au, au)))}</span>' for i, au in enumerate(ONT.AU_BLENDSHAPES) if pipe.meas[i])
    summ4 = ("active: " + ", ".join(on[:4]) + ("…" if len(on) > 4 else "")) if on else (
        "no movement measured" if not res["found"] else "no strong movement")
    S.append(step(4, "Measure the face", e(summ4),
                  f'<p class="plain">MediaPipe\'s 52 blendshape scores are read as facial action units, the muscle '
                  f'movements that make up an expression. Highlighted movements are active in this face.</p>'
                  f'<div class="chips">{chips}</div>'))

    r = y["rule"]
    rule = ""
    if r and r.get("available"):
        rule = (f'<p class="plain">How LLaVA-1.5 behaves in general: a decision tree that imitates it (matching it on '
                f'{100 * r["fidelity"]:.0f}% of test faces) says that when {"; ".join(e(x) for x in r["conditions"])}, '
                f'LLaVA usually answers <b>{e(CLASSES[r["predicts"]])}</b>.</p>')
    summ5 = (f'corrected {e(CLASSES[c["v_before"]])} to {e(CLASSES[c["v_after"]])}' if c["fired"] else
             f'kept {e(CLASSES[c["v_before"]])}')
    S.append(step(5, "Check LLaVA-1.5 with the compensator", summ5,
                  f'<p class="plain">A detector, trained on LLaVA\'s past answers, estimates how likely its answer is to '
                  f'be wrong: <b>{100 * c["detector"]:.0f}%</b> here. Above {100 * c["threshold"]:.0f}% (purple mark) a '
                  f'corrector trained only on LLaVA\'s mistakes replaces the answer.</p><div class="meter"><span '
                  f'class="lvl" style="width:{min(100, 100 * c["detector"]):.0f}%"></span><span class="thr" '
                  f'style="left:{100 * c["threshold"]:.0f}%"></span></div>{rule}'))

    if y["expected"] and res["found"]:
        lst = "; ".join(f'{e(AU_NAMES.get(x["au"], x["au"]))}: ' + (
            '<span class="ok">found</span>' if x["present"] else
            '<span class="miss">not found</span>' if x["measurable"] else 'cannot be measured') for x in y["expected"])
        n_ok = sum(x["present"] for x in y["expected"])
        summ6 = f'{n_ok} of {len(y["expected"])} expected movements for {e(CLASSES[final])} found'
        body6 = (f'<p class="plain">The OWL ontology records which action units show each emotion. For '
                 f'{e(CLASSES[final])} it expects: {lst}.</p>')
    else:
        summ6 = "checked against the ontology"
        body6 = ('<p class="plain">The OWL ontology records which action units show each emotion and which facial '
                 'regions they move; each model\'s explanation is scored by how much of it falls in those regions.</p>')
    S.append(step(6, "Check against the ontology", summ6, body6))

    th = "".join(f"<th>{s}</th>" for s in SOURCES) + "<th>Combined</th>"
    trs = "".join(f'<tr><td>{e(cap(cn))}</td>' + "".join(f'<td>{100 * ev[f"P_{s}"][0, k]:.0f}</td>' for s in SOURCES) +
                  f'<td class="final">{100 * w["scores"][k]:.0f}</td></tr>' for k, cn in enumerate(CLASSES))
    rule_txt = {0: 'the top score is high enough and leads the next clearly, so one label is given',
                1: 'neither leads clearly, but the two are a confusable pair in the ontology, so both are given',
                2: 'no class leads clearly and the top two are not a confusable pair, so the face is referred to a person'}
    summ7 = (f'{e(CLASSES[final])} {100 * w["scores"][final]:.0f}%, {e(CLASSES[second])} {100 * w["scores"][second]:.0f}%'
             f', so: {e(word if w["kind"] != 1 else CLASSES[final] + " or " + CLASSES[second]).lower()}')
    tr_txt = e(w["trace"].replace(" | ", "\n").replace("V: ", "V (after the compensator): ", 1))
    note = f'<p class="small">{e(cap(res["note"]))}.</p>' if res["note"] else ""
    took = f'<p class="small">Analysed in {elapsed:.1f} seconds.</p>' if elapsed else ""
    S.append(step(7, "Combine and decide", summ7,
                  f'<p class="plain">The calibrated answers are weighted and combined with the ontology evidence; '
                  f'{rule_txt[w["kind"]]}.</p><table class="scores"><tr><th>Expression (%)</th>{th}</tr>{trs}</table>'
                  f'<pre class="trace">{tr_txt}</pre>{note}{took}'))
    out.append('<div class="steps-h">Step by step</div><p class="steps-sub">How the system reached this result. Open a '
               'step to see its evidence.</p><div class="steps">' + "".join(S) + '</div>')
    return "\n".join(out)


HOW = """
<div class="panel howto">
<h2>How the system works</h2>
<p>Four recognisers of different kinds look at the same face: LLaVA-1.5, a generative vision-language model used without
any training on emotions; CLIP, also used without training; POSTER++, a transformer trained on the RAF-DB faces; and a
support vector machine on hand-crafted HOG and LBP features. Each is explained in the same way, by measuring how its
answer changes when each of seven facial regions is blurred out. MediaPipe measures the face and reads the movements of
its muscles as facial action units.</p>
<p>A compensator watches LLaVA-1.5: a detector estimates whether its answer is wrong and, only then, a corrector trained
on its past mistakes replaces it. A reasoner combines all the evidence with rules read from an OWL ontology of emotions,
action units and facial regions. It gives one label when the evidence is clear, both labels when the face shows one of
three easily confused pairs (fear and surprise, anger and disgust, sadness and neutral), and refers the face to a person
when the evidence conflicts.</p>
<table>
<tr><td>POSTER++ alone, RAF-DB test set of 3,068 faces</td><td>91.1% correct</td></tr>
<tr><td>Full framework</td><td>92.1% correct</td></tr>
<tr><td>LLaVA-1.5 alone, and with the compensator</td><td>59.8% and 81.5%</td></tr>
<tr><td>Errors among the faces the system answers (92% of them)</td><td>4.4%, against 8.9% for POSTER++</td></tr>
</table>
<h2 style="margin-top:22px">How the explanations are made</h2>
<p><b>Local explanations (region SHAP and LIME).</b> For each model, every one of the seven facial regions is blurred
in turn across 32 copies of the face, and the change in the model's answer is measured. SHAP turns these changes into
each region's share of the answer; LIME fits a simple linear model to the same copies. When both rank the regions alike,
the explanation is more trustworthy.</p>
<p><b>Global explanation (surrogate decision tree).</b> A small decision tree is trained to imitate LLaVA-1.5 from the
face measurements and its region explanations. Its rules describe how LLaVA behaves in general, and it matches LLaVA on
about 72% of test faces.</p>
<p><b>Knowledge-based explanation (ontology).</b> The OWL ontology records which facial action units show each emotion
and which regions they move. The system checks the face against these expectations, and the reasoner's justification
trace names every source, its evidence, and the rule that produced the result.</p>
<p><b>Testing the explanation.</b> The region an explanation points to is blurred and the model is asked again: if the
confidence falls, the explanation is faithful to what the model actually used.</p>
<p class="small">The system estimates the apparent expression in a photo, not what a person feels. Photos are analysed
in this session and are not stored.</p>
</div>"""


def build_examples(out, images_root, n=3):
    idx = {}
    for dp, _, files in os.walk(images_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                idx.setdefault(os.path.splitext(f)[0], os.path.join(dp, f))
    tr_path = os.path.join(out, "results", "traces_test.csv")
    if not os.path.exists(tr_path):
        return {}
    tr = pd.read_csv(tr_path)
    tr["image_id"] = tr.image_id.astype(str)
    ex = {}
    names = {0: "given a single label", 1: "given a blended label", 2: "referred to a person"}
    for k in (0, 1, 2):
        sub = tr[(tr.decision == k) & tr.image_id.isin(idx)]
        for r in sub.head(n).itertuples():
            ex[f"{cap(names[k])}: {r.image_id}"] = dict(path=idx[r.image_id], label=r.label, decision=names[k])
    return ex


# ----------------------------------------------------------------------------- app
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/kaggle/working/poster")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--svm", default=None)
    ap.add_argument("--landmarker", default=None)
    ap.add_argument("--images_root", default="/kaggle/input")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--selftest", default=None, help="analyse this image, write selftest.html and exit")
    a = ap.parse_args()
    a.svm = a.svm or os.path.join(a.out, "models", "svm_C.joblib")
    lmk = a.landmarker or os.path.join(a.out, "models", "face_landmarker.task")
    if not os.path.exists(lmk):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(lmk), exist_ok=True)
            urllib.request.urlretrieve(LANDMARKER_URL, lmk)
        except Exception as ex:
            print("landmark model not available:", ex); lmk = None
    t0 = time.time()
    pipe = Pipeline(a.out, load_models(a, print), lmk)
    print(f"ready in {time.time() - t0:.0f} s")

    def run(img, explain_v, progress=None):
        if img is None:
            return EMPTY
        t = time.time()
        say = None
        if progress is not None:
            say = lambda msg: progress(None, desc=msg)
        res = pipe.analyse(img, explain_v=explain_v, progress=say)
        return render(res, pipe, time.time() - t)

    if a.selftest:
        page = run(Image.open(a.selftest), True)
        open("selftest.html", "w").write(f"<html><head><meta charset='utf-8'><style>{CSS + STEP_CSS}</style></head><body>"
                                         f"<div class='gradio-container' style='padding:24px'>{page}</div></body></html>")
        print("wrote selftest.html")
        return

    import gradio as gr
    theme = gr.themes.Base(font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
                           primary_hue="slate", neutral_hue="slate", radius_size="md")
    examples = build_examples(a.out, a.images_root)
    head = ('<div id="masthead"><div><h1>Explainable Facial Emotion Recognition through Ontology-Grounded Fusion of '
            'Model Explanations and a Post-Hoc Vision-Language Compensator</h1><p>M.Sc. Information Technology research '
            'project, University of Lagos. Oluwasola Tolase Mercy (249074221); supervisor Dr. E. P. Fasina.</p></div>'
            f'<div class="status">Rules read from the {e(pipe.knowledge)}</div></div>')
    with gr.Blocks(title="Explainable Facial Emotion Recognition: ontology-grounded fusion and a post-hoc compensator", css=CSS + STEP_CSS, theme=theme) as app:
        gr.HTML(head)
        with gr.Tab("Analyse a face"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300):
                    inp = gr.Image(sources=["upload", "webcam"], type="pil", label="Face photo", height=330)
                    ev_ = gr.Checkbox(value=True, label="Explain LLaVA-1.5 too (about 15 seconds more)")
                    btn = gr.Button("Analyse this face", elem_id="analyse")
                    gr.HTML('<p class="hint">Use a clear, front-facing photo of one face. The photo is analysed in this '
                            'session only and is not stored.</p>')
                with gr.Column(scale=9):
                    page = gr.HTML(EMPTY)
            def run_ui(img, explain_v, progress=gr.Progress()):
                return run(img, explain_v, progress)
            btn.click(run_ui, [inp, ev_], page)
        with gr.Tab("RAF-DB examples"):
            if examples:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        dd = gr.Dropdown(list(examples), value=list(examples)[0], label="Test face")
                        eimg = gr.Image(type="pil", label="Face", height=260, interactive=False)
                        stored = gr.HTML()
                        ebtn = gr.Button("Analyse this face", elem_id="analyse")
                        gr.HTML('<p class="hint">Faces from the RAF-DB test set, for private demonstration only.</p>')
                    with gr.Column(scale=9):
                        epage = gr.HTML(EMPTY)

                def show(k):
                    x = examples[k]
                    s = (f'<p class="hint">Annotators\' label: <b>{e(x["label"])}</b>. In the experiments this face was '
                         f'{e(x["decision"])}.</p>')
                    return Image.open(x["path"]).convert("RGB"), s
                dd.change(show, dd, [eimg, stored])
                app.load(show, dd, [eimg, stored])
                def run_ex(k, progress=gr.Progress()):
                    return run(Image.open(examples[k]["path"]), True, progress)
                ebtn.click(run_ex, dd, epage)
            else:
                gr.HTML('<div class="panel empty"><h2>No test faces found</h2>Add the RAF-DB dataset as an input of the '
                        'notebook to see examples here.</div>')
        with gr.Tab("How it works"):
            gr.HTML(HOW)
    app.queue().launch(share=a.share, server_name="0.0.0.0", server_port=a.port)


if __name__ == "__main__":
    main()
