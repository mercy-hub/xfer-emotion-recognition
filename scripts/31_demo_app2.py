"""Step 31: the Phase 2 demonstration app. Upload a face (or use the webcam) and follow the three stages of the final
system: the experts and their explanations, the fusion, and the ontology decision with its justification trace.

    python scripts/31_demo_app2.py --out OUT --work POSTER_WORK --ckpt POSTER_BEST --a2 A2_BEST --lora SLM_LORA --share

--mock replaces every model with a fake (no GPU needed); --selftest IMG writes the result page to selftest2.html.
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
import numpy as np
import pandas as pd
from PIL import Image

from xfer.demo import LANDMARKER_URL
from xfer.demo2 import AU_NAMES, EXPERTS, KINDS, NAMES, NATIVE, Pipeline2, load_experts
from xfer.labels import CLASSES
from xfer.regions import FINE

REG_NAME = {"forehead": "forehead", "brow_l": "left brow", "brow_r": "right brow", "eye_l": "left eye",
            "eye_r": "right eye", "nose": "nose", "cheek_l": "left cheek", "cheek_r": "right cheek", "mouth": "mouth",
            "chin": "chin"}


def e(s):
    return html.escape(str(s))


def cap(s):
    return s[:1].upper() + s[1:]


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heat(res, w, size=180, cmap="YlOrRd"):
    face = np.asarray(res["face"].convert("L"), float) / 255
    out = np.stack([face] * 3, -1) * 0.8 + 0.15
    w = np.asarray(w, float)
    if np.isnan(w).all():
        return Image.fromarray((out * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    v = np.nanmax(w) or 1
    cm = matplotlib.colormaps[cmap]
    for k in range(len(FINE)):
        if np.isnan(w[k]) or w[k] <= 0:
            continue
        a = 0.7 * min(1, w[k] / v)
        out[res["masks"][k]] = out[res["masks"][k]] * (1 - a) + np.array(cm(0.25 + 0.75 * w[k] / v)[:3]) * a
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)


def regions_img(res, size=180):
    face = np.asarray(res["face"].convert("RGB"), float) / 255
    out = face.copy()
    cm = matplotlib.colormaps["tab10"]
    for k in range(len(FINE)):
        out[res["masks"][k]] = out[res["masks"][k]] * 0.55 + np.array(cm(k)[:3]) * 0.45
    return Image.fromarray((out * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)


def top_regions(w, n=2):
    w = np.asarray(w, float)
    if np.isnan(w).all():
        return "not computed"
    o = np.argsort(-np.nan_to_num(w))
    return ", ".join(f"{REG_NAME[FINE[i]]} {w[i]:.0%}" for i in o[:n] if w[i] > 0)


# ----------------------------------------------------------------------------- page
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Public+Sans:wght@400;500;600&display=swap');
:root, .gradio-container { --paper:#eef1f4; --surface:#ffffff; --ink:#172231; --muted:#5d6a79; --rule:#dde3ea;
  --label:#1e6b52; --blend:#9a6a12; --abstain:#6a4a9c; --focus:#2f5fb3; --bad:#9b3b2f; }
body, .gradio-container { background: var(--paper) !important; color: var(--ink); font-family:'Public Sans', system-ui, sans-serif; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
footer { display:none !important; }
#masthead { padding: 26px 4px 14px; border-bottom: 1px solid var(--rule); margin-bottom: 8px }
#masthead h1 { font-family:'Newsreader', Georgia, serif; font-weight:500; font-size: 29px; line-height:1.2; margin:0; max-width: 50ch }
#masthead p { margin:8px 0 0; color:var(--muted); font-size:14.5px; max-width: 80ch }
.panel { background: var(--surface); border:1px solid var(--rule); border-radius: 10px; }
#analyse { background: var(--ink) !important; color:#fff !important; border:none !important; font-weight:600; font-size:16px; border-radius: 8px !important; min-height: 46px }
.hint { color: var(--muted); font-size: 13.5px; line-height: 1.5; margin: 2px 4px; }
.empty { padding: 44px 40px; color: var(--muted); font-size: 15px; line-height:1.6 }
.empty h2 { font-family:'Newsreader', Georgia, serif; font-weight:500; color:var(--ink); font-size:26px; margin:0 0 8px; }
.result { display:grid; grid-template-columns: 1fr 200px; gap: 26px; padding: 24px 28px; border-left: 8px solid var(--c); }
.result .kind { font-size: 14px; color: var(--c); font-weight:600; }
.result .word { font-family:'Newsreader', Georgia, serif; font-weight:500; font-size: 50px; line-height:1.05; margin: 4px 0 8px; }
.result .word .or { font-style: italic; color: var(--muted); font-size: 0.55em; padding: 0 0.2em; }
.result .sent { font-size: 15.5px; line-height: 1.5; margin: 0 0 16px; max-width: 56ch }
.result figure { margin:0 } .result figure img { width:200px; height:200px; border-radius:8px; display:block }
.result figcaption, .card figcaption { font-size: 12.5px; color: var(--muted); margin-top: 6px; line-height:1.4 }
.checks { list-style:none; margin:0; padding:0; display:grid; gap: 9px }
.checks li { display:grid; grid-template-columns: 130px 1fr; gap: 12px; font-size: 14.5px; line-height:1.45 }
.checks .k { color: var(--muted); font-size: 13px }
.ok { color: var(--label); font-weight:600 } .miss { color: var(--bad); font-weight:600 }
.steps-h { font-family:'Newsreader', Georgia, serif; font-weight:500; font-size: 23px; margin: 24px 0 4px 2px }
.steps-sub { color: var(--muted); font-size: 14px; margin: 0 0 12px 2px }
details.step { position: relative; margin: 0 0 10px 0; padding-left: 52px }
details.step::before { content:''; position:absolute; left: 17px; top: 40px; bottom: -12px; width: 2px; background: var(--rule) }
details.step:last-child::before { display:none }
details.step > summary { list-style:none; cursor:pointer; background: var(--surface); border:1px solid var(--rule); border-radius: 10px; padding: 13px 18px; display:flex; align-items: baseline; gap: 10px }
details.step > summary::-webkit-details-marker { display:none }
details.step > summary::after { content: '+'; color: var(--muted); font-size: 19px; margin-left: auto }
details.step[open] > summary::after { content: '\\2212' }
details.step[open] > summary { border-bottom-left-radius: 0; border-bottom-right-radius: 0 }
.num { position:absolute; left:0; top:7px; width:36px; height:36px; border-radius:50%; background: var(--ink); color:#fff; display:grid; place-items:center; font-family:'Newsreader', Georgia, serif; font-size:18px }
.step .t { font-weight:600; font-size:15px; white-space: nowrap } .step .s { color: var(--muted); font-size: 14px }
.step .body { background: var(--surface); border:1px solid var(--rule); border-top:none; border-radius: 0 0 10px 10px; padding: 8px 18px 18px }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px } .card { min-width:0; overflow-wrap:anywhere } .card img { max-width:100%; height:auto }
.card { border:1px solid var(--rule); border-radius: 8px; padding: 10px; font-size: 13px; line-height:1.45 }
.card b { font-size: 13.5px } .card .pair { display:grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 8px 0 }
.card img { width:100%; aspect-ratio:1; border-radius:5px; display:block }
.tag { display:inline-block; font-size: 11.5px; padding: 2px 8px; border-radius: 999px; font-weight:600 }
.tag.support { background:#e3f1ea; color: var(--label) } .tag.conflict { background:#f6e6e3; color: var(--bad) }
table.t { width:100%; border-collapse: collapse; font-size: 13.5px; font-variant-numeric: tabular-nums; margin-top: 6px }
table.t th { text-align:right; font-weight:600; padding: 7px 8px; border-bottom: 1px solid var(--rule); color: var(--muted); font-size: 12.5px }
table.t td { text-align:right; padding: 6px 8px; border-bottom: 1px solid #eef1f4 }
table.t th:first-child, table.t td:first-child { text-align:left }
table.t td.best { font-weight:700 }
.bar { height: 9px; background: var(--ink); border-radius: 4px; display:inline-block; vertical-align: middle }
ul.rules { margin: 6px 0 0 18px; padding:0; font-size: 14px; line-height: 1.6 }
pre.trace { white-space: pre-wrap; font-size: 13px; line-height: 1.6; background:#f6f8fa; border-radius:6px; padding: 12px 14px; font-family: 'Public Sans', sans-serif }
.row { display:flex; gap: 18px; flex-wrap: wrap; align-items:flex-start }
.howto { padding: 26px 30px; font-size: 15.5px; line-height: 1.7 }
.howto h2 { font-family:'Newsreader', Georgia, serif; font-weight:500; font-size: 26px; margin: 0 0 10px }
.howto table { border-collapse: collapse; margin: 14px 0; font-size: 15px }
.howto td { padding: 7px 18px 7px 0; border-bottom: 1px solid var(--rule) }
@media (max-width: 860px) { .result { grid-template-columns: 1fr } .cards { grid-template-columns: 1fr 1fr } .checks li { grid-template-columns: 1fr } .step .s { display:none } }
"""
KIND = {0: ("Single label", "var(--label)"), 1: ("Compound label: two expressions this face does not separate", "var(--blend)"),
        2: ("Not sure, so the face is referred to a person", "var(--abstain)")}
EMPTY = ('<div class="panel empty"><h2>Add a face to see the result and its explanation</h2>Four expert models read '
         'the face, a small language model checks the vision-language model, the evidence is fused, and an emotion '
         'ontology decides: one emotion, two that this face does not separate, or a referral to a person. Every step '
         'can be opened to see its evidence.</div>')


def step(n, title, summary, body, open_=False):
    return (f'<details class="step"{" open" if open_ else ""}><summary><span class="num">{n}</span><span class="t">'
            f'{title}</span><span class="s">{summary}</span></summary><div class="body">{body}</div></details>')


def render(res, pipe):
    F, c1, c2, kind = res["F"], res["c1"], res["c2"], res["kind"]
    kt, col = KIND[kind]
    if kind == 0:
        word = e(cap(CLASSES[c1]))
        sent = f"The experts and the face measurements point to {e(CLASSES[c1])} (combined confidence {F[c1]:.0%})."
    elif kind == 1:
        word = f'{e(cap(CLASSES[c1]))}<span class="or">or</span>{e(CLASSES[c2])}'
        sent = (f"{cap(CLASSES[c1])} ({F[c1]:.0%}) and {CLASSES[c2]} ({F[c2]:.0%}) score close together, and the "
                f"ontology marks them as a pair faces often show in the same way, so both are given.")
    else:
        word = "Refer to a person"
        sent = (f"The evidence is split between {e(CLASSES[c1])} ({F[c1]:.0%}) and {e(CLASSES[c2])} ({F[c2]:.0%}), "
                f"so the system does not guess.")
    sup = [s for s in EXPERTS if res["support"][s] == "support"]
    v = res["vlm"]
    slm = res["slm"]
    au_line = ("no landmarks found" if not res["found"] else
               (f'{len(res["present"])} of {len(res["expected"])} expected for {CLASSES[c1]}: ' +
                ", ".join((f'<span class="ok">{e(AU_NAMES.get(a, a))}</span>' if a in res["present"] else
                           f'<span class="miss">no {e(AU_NAMES.get(a, a))}</span>') for a in res["expected"])
                if res["expected"] else ("none expected for neutral; active: " +
                                         (", ".join(e(AU_NAMES.get(a, a)) for a in res["active"]) or "none"))))
    checks = [("Experts agreeing", f'{len(sup)} of {len(EXPERTS)}: {", ".join(e(NAMES[s]) for s in sup) or "none"} '
                                   f'(agreement {res["agreement"]:.2f})'),
              ("Where it looked", e(top_regions(res["consensus"], 3))),
              ("Facial movements", au_line),
              ("VLM's claims", f'{v["verified"]} of {v["n"]} confirmed by the measurements' if v["n"] else "no claims"),
              ("SLM check", f'{e(slm["decision"])}: {e(slm["explanation"][:180])}')]
    card = (f'<div class="panel result" style="--c:{col}"><div><div class="kind">{e(kt)}</div><div class="word">{word}</div>'
            f'<p class="sent">{sent}</p><ul class="checks">' +
            "".join(f'<li><span class="k">{k}</span><span>{vv}</span></li>' for k, vv in checks) +
            f'</ul></div><figure><img src="{b64(heat(res, res["consensus"], 200))}" alt="consensus map"><figcaption>'
            f'Fused explanation: the regions whose occlusion moved the supporting experts most.</figcaption></figure></div>')
    # step 1: face
    aul = ", ".join(f"{e(AU_NAMES.get(a, a))} ({res['au_level'].get(a, 0):.1f}x)" for a in res["active"]) or "none"
    s1 = (f'<div class="row"><img src="{b64(regions_img(res))}" style="width:150px;border-radius:6px"><div style="max-width:60ch">'
          f'<p>{(e(cap(res["note"])) + ".") if res["note"] else "Face found, cropped to 100 x 100 like the RAF-DB faces, and given 478 landmarks."} '
          f'The ten coloured regions are the units of every explanation.</p><p><b>Active action units</b> (MediaPipe, '
          f'level relative to the training threshold): {aul}.</p></div></div>')
    # step 2: experts
    cards = []
    for s in EXPERTS:
        P = res["P"][s]
        c = int(P.argmax())
        occ = res["occ"].get(s)
        nat = res["native"][s]
        cards.append(f'<div class="card"><b>{e(NAMES[s])}</b><br><span style="color:var(--muted)">{e(KINDS[s])}</span>'
                     f'<div style="margin-top:6px">{e(CLASSES[c])} ({P[c]:.0%}) <span class="tag {res["support"][s]}">'
                     f'{res["support"][s]}</span></div><div class="pair"><figure style="margin:0"><img src="{b64(heat(res, nat, 120, "PuBu"))}">'
                     f'<figcaption>Own: {e(NATIVE[s])}</figcaption></figure><figure style="margin:0"><img src="'
                     f'{b64(heat(res, occ["weights"] if occ else np.full(len(FINE), np.nan), 120))}"><figcaption>Shared: '
                     f'occlusion</figcaption></figure></div>Own: {e(top_regions(nat))}<br>Occlusion: '
                     f'{e(top_regions(occ["weights"]) if occ else "not computed")}<br>Faithfulness: '
                     f'{(format(occ["faith"], ".2f") if occ else "-")}</div>')
    s2 = ('<p class="hint">Each expert gives its answer, its own explanation (left, blue) and the shared occlusion test '
          '(right, red): blur each region and see how much the expert\'s confidence drops. Faithfulness 1 means the region '
          'the explanation points to really carries the answer.</p><div class="cards">' + "".join(cards) + "</div>")
    # step 3: VLM + SLM
    rows = "".join(f"<tr><td>{e(f)}</td><td style='text-align:left'>{e(o)}</td><td style='text-align:left'>"
                   f"{'<span class=ok>confirmed</span>' if ok else ('<span class=miss>contradicted: the measurements show movement here</span>' if o == 'relaxed' else '<span class=miss>not found in the measurements</span>')}</td></tr>"
                   for f, o, ok in v["claims"])
    s3 = (f'<p><b>InternVL3-8B said:</b> {e(v["desc"].get("reason", ""))}</p><table class="t"><tr><th>Feature</th>'
          f'<th style="text-align:left">Claim</th><th style="text-align:left">Check against MediaPipe</th></tr>{rows}</table>'
          f'<p style="margin-top:14px"><b>SLM compensator</b> (Qwen2.5-1.5B + LoRA, reads the VLM\'s answer, claims and the '
          f'measured action units): <b>{e(slm["decision"])}</b>, emotion <b>{e(CLASSES[slm["emotion"]])}</b>.<br>'
          f'{e(slm["explanation"])}</p>')
    # step 4: fusion
    head = "".join(f"<th>{e(c[:4].capitalize())}</th>" for c in CLASSES)
    trs = ""
    for s in EXPERTS:
        P = res["P"][s]
        trs += f"<tr><td>{e(NAMES[s])}</td>" + "".join(
            f"<td class='{'best' if k == P.argmax() else ''}'>{P[k]:.2f}</td>" for k in range(len(CLASSES))) + "</tr>"
    trs += "<tr><td><b>Final system</b></td>" + "".join(
        f"<td class='{'best' if k == c1 else ''}'>{F[k]:.2f}</td>" for k in range(len(CLASSES))) + "</tr>"
    sh = res["shares"]
    m = max(abs(x) for x in sh.values()) or 1
    bars = "".join(f"<tr><td>{e(NAMES[s])}</td><td style='text-align:left'><span class='bar' style='width:{int(140 * max(sh[s], 0) / m)}px'>"
                   f"</span> {sh[s]:+.2f}</td></tr>" for s in EXPERTS)
    s4 = (f'<table class="t"><tr><th>Expert</th>{head}</tr>{trs}</table><p class="hint" style="margin-top:10px">The final '
          f'combiner averages a CatBoost model and a weighted average of the experts, both fitted on the validation '
          f'split. How much each expert pushed the decision towards {e(CLASSES[c1])} (CatBoost SHAP values):</p>'
          f'<table class="t" style="max-width:480px">{bars}</table><p>Agreement score {res["agreement"]:.2f}; outliers: '
          f'{", ".join(e(NAMES[s]) for s in res["outliers"]) or "none"}.</p>')
    # step 5: ontology
    tr_lines = [f"{NAMES[s]}: {CLASSES[int(res['P'][s].argmax())]} (p = {res['P'][s].max():.2f}, "
                f"{res['support'][s]}, faithfulness {res['occ'][s]['faith']:.2f})" if res["occ"].get(s) else
                f"{NAMES[s]}: {CLASSES[int(res['P'][s].argmax())]} (p = {res['P'][s].max():.2f}, {res['support'][s]})"
                for s in EXPERTS]
    tr_lines.append(f"VLM says: {v['desc'].get('reason', '')} [claims verified {v['verified']} of {v['n']}]")
    tr_lines.append(f"SLM: {slm['decision']} - {slm['explanation']}")
    tr_lines.append("Action units: " + (", ".join(res["active"]) or "none"))
    tr_lines.append(f"Fused: {CLASSES[c1]} {F[c1]:.2f}, {CLASSES[c2]} {F[c2]:.2f}; agreement {res['agreement']:.2f} -> "
                    + ["LABEL " + CLASSES[c1], f"COMPOUND {CLASSES[c1]}-{CLASSES[c2]}", "ABSTAIN (refer to a person)"][kind])
    s5 = (f'<p>Knowledge read from the {e(pipe.knowledge)} (EMONTO extended with FACS action units and confusable '
          f'pairs). Rules that fired:</p><ul class="rules">' + "".join(f"<li>{e(r)}</li>" for r in res["rules"]) +
          f'</ul><pre class="trace">{e(chr(10).join(tr_lines))}</pre>')
    t = res["times"]
    steps = ('<div class="steps-h">How the decision was made</div><div class="steps-sub">Open any step to see its '
             f'evidence. Analysed in {t["total"]:.0f} s (explanations {t.get("explanations", 0):.0f} s).</div>' +
             step(1, "The face", f'{len(res["active"])} action units active', s1) +
             step(2, "Stage 1: four experts and their explanations",
                  f'{len([s for s in EXPERTS if res["support"][s] == "support"])} of 5 outputs support the result', s2) +
             step(3, "The VLM's rationale and the SLM compensator", f'SLM: {slm["decision"]}', s3) +
             step(4, "Stage 2: fusing the evidence", f'{CLASSES[c1]} {F[c1]:.0%}', s4) +
             step(5, "Stage 3: ontology reasoning and decision", ["single label", "compound label", "referral"][kind],
                  s5, open_=True))
    return card + steps


HOW = """<div class="panel howto"><h2>How the system works</h2>
<p><b>Stage 1, neuro.</b> Four experts of different kinds read the face: POSTER++, CLIP (fine-tuned on RAF-DB, and as it
was before fine-tuning), the vision-language model InternVL3-8B, and HOG/LBP features with an SVM. Each explains itself in
its own way, and all four are also explained by one shared test: blur each facial region and measure the drop in
confidence. A small language model checks the vision-language model's answer and its written claims against the action
units measured by MediaPipe.</p>
<p><b>Stage 2, fusion.</b> A combiner fitted on validation images (CatBoost averaged with a weighted average) turns the
experts' answers into one probability per emotion; its SHAP values show which experts drove the decision.</p>
<p><b>Stage 3, symbolic.</b> An extension of the EMONTO ontology with FACS action units checks the facial movements,
allows a compound label only for emotions faces often confuse, ranks measurements above the language model's claims, and
decides: one emotion, two, or referral to a person.</p>
<table><tr><td>Final system on the RAF-DB test set (3,068 faces)</td><td>94.07 per cent</td></tr>
<tr><td>Best single expert (CLIP fine-tuned)</td><td>92.54 per cent</td></tr>
<tr><td>POSTER++ alone</td><td>91.10 per cent</td></tr>
<tr><td>Error among accepted faces (96 per cent accepted)</td><td>3.8 per cent</td></tr></table>
<p style="color:var(--muted);font-size:13.5px">The system estimates the apparent expression in a photo, not what a person
feels. Photos are analysed in this session and are not stored.</p></div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/kaggle/working/poster")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--a2", default=None, help="fine-tuned CLIP checkpoint (A2_best.pth)")
    ap.add_argument("--lora", default=None, help="SLM LoRA folder (slm/lora)")
    ap.add_argument("--svm", default=None)
    ap.add_argument("--vlm_model", default=None)
    ap.add_argument("--cache", default="/kaggle/working/demo_cache")
    ap.add_argument("--landmarker", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--selftest", default=None)
    a = ap.parse_args()
    a.svm = a.svm or os.path.join(a.out, "models", "svm_C.joblib")
    lmk = a.landmarker or os.path.join(a.cache, "face_landmarker.task")
    if not os.path.exists(lmk):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(lmk), exist_ok=True)
            urllib.request.urlretrieve(LANDMARKER_URL, lmk)
        except Exception as ex:
            print("landmark model not available:", ex); lmk = None
    t0 = time.time()
    ex, vlm, slm = load_experts(a)
    pipe = Pipeline2(ex, vlm, slm, lmk)
    print(f"ready in {time.time() - t0:.0f} s", flush=True)

    def run(img, explain_vlm=True, progress=None):
        if img is None:
            return EMPTY
        say = (lambda msg: progress(None, desc=msg)) if progress is not None else None
        return render(pipe.analyse(img, explain_vlm=explain_vlm, progress=say), pipe)

    if a.selftest:
        page = run(Image.open(a.selftest))
        open("selftest2.html", "w").write(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"
                                          f"<div class='gradio-container' style='padding:24px'>{page}</div></body></html>")
        print("wrote selftest2.html")
        return

    import gradio as gr
    theme = gr.themes.Base(font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
                           primary_hue="slate", neutral_hue="slate", radius_size="md")
    m = pd.read_csv(os.path.join(a.out, "manifest.csv"))
    m = m[(m.split == "test") & m.path.map(os.path.exists)]
    examples = {}
    for k, c in enumerate(CLASSES):
        for r in m[m.label == k].head(3).itertuples():
            examples[f"{cap(c)}: {r.image_id}"] = r.path
    head = ('<div id="masthead"><h1>Explainable Facial Emotion Recognition through Ontology-Grounded Fusion of Model '
            'Explanations and a Post-Hoc Vision-Language Compensator</h1><p>M.Sc. Information Technology research project, '
            'University of Lagos. Oluwasola Tolase Mercy (249074221); supervisor Dr. E. P. Fasina. Final system: four '
            'experts, XAI fusion, SLM compensator and EMONTO + FACS reasoning.</p></div>')
    with gr.Blocks(title="Explainable facial emotion recognition: the final system", css=CSS, theme=theme) as app:
        gr.HTML(head)
        with gr.Tab("Analyse a face"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300):
                    inp = gr.Image(sources=["upload", "webcam"], type="pil", label="Face photo", height=330)
                    xv = gr.Checkbox(value=True, label="Explain the VLM too (about 10 seconds more)")
                    btn = gr.Button("Analyse this face", elem_id="analyse")
                    gr.HTML('<p class="hint">Use a clear photo of one face. It is analysed in this session only and '
                            'is not stored.</p>')
                with gr.Column(scale=9):
                    page = gr.HTML(EMPTY)
            btn.click(lambda img, x, progress=gr.Progress(): run(img, x, progress), [inp, xv], page)
        with gr.Tab("RAF-DB examples"):
            if examples:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        dd = gr.Dropdown(list(examples), value=list(examples)[0], label="Test face (annotated emotion)")
                        eimg = gr.Image(type="pil", label="Face", height=260, interactive=False)
                        ebtn = gr.Button("Analyse this face", elem_id="analyse")
                        gr.HTML('<p class="hint">RAF-DB test faces, for private demonstration only.</p>')
                    with gr.Column(scale=9):
                        epage = gr.HTML(EMPTY)
                dd.change(lambda k: Image.open(examples[k]).convert("RGB"), dd, eimg)
                app.load(lambda k: Image.open(examples[k]).convert("RGB"), dd, eimg)
                ebtn.click(lambda k, progress=gr.Progress(): run(Image.open(examples[k]), True, progress), dd, epage)
            else:
                gr.HTML('<div class="panel empty"><h2>No test faces found</h2>Add the RAF-DB dataset as an input.</div>')
        with gr.Tab("How it works"):
            gr.HTML(HOW)
    app.queue().launch(share=a.share, server_name="0.0.0.0", server_port=a.port)


if __name__ == "__main__":
    main()
