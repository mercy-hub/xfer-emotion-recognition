"""Stage 1 of the new design: each expert's own ("native") explanation, next to the shared occlusion explanation.

  --method linsvm    HOG/LBP + linear SVM: exact per-region contributions (weight x standardised feature) and
                     calibrated probabilities (predictions/CL_<split>.csv). CPU.
  --method clip_au   CLIP AU-phrase concept probing: similarity of each face to phrases such as "raised inner
                     eyebrows", tested against MediaPipe's measured action units (AUC per action unit). GPU.
  --method clip_rel  CLIP relevance map per emotion (gradient x attention rollout, Chefer et al. 2021). GPU.
  --method poster    POSTER++: gradient-weighted relevance of its three cross-attention outputs, and Integrated
                     Gradients through the landmark stream only (the image stream is held at the real face). GPU.
  --method vlm_rel   the VLM's answer-token relevance (input x gradient of the answer's first token). GPU.
Every map is summed over the 10 fine face regions and compared with the same expert's occlusion explanation
(Spearman over regions): the faithfulness check of the native explanation that the diagram asks for.
Output: explanations/native_<method>_test.csv and a summary line in results/native_xai.md.
Usage:  python scripts/23_native_xai.py --out OUT --method clip_rel --n 300
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from xfer.explain import blurred
from xfer.io import append_predictions, load_manifest, pred_path, set_seed
from xfer.labels import CLASSES, CLIP_WORDS, K
from xfer.metrics import accuracy, macro_f1
from xfer.regions import FINE, REGIONS, fine_region_masks, region_map

R = len(FINE)
AU_PHRASES = {"AU1": "raised inner eyebrows", "AU2": "raised outer eyebrows", "AU4": "lowered, furrowed eyebrows",
              "AU5": "wide open eyes", "AU7": "narrowed, squinting eyes", "AU10": "a raised upper lip",
              "AU12": "a smile with the lip corners pulled up", "AU15": "lip corners pulled down",
              "AU17": "a raised chin", "AU20": "lips stretched sideways", "AU23": "tightly pressed lips",
              "AU26": "an open mouth with a dropped jaw"}


def landmarks(out):
    z = np.load(os.path.join(out, "features", "landmarks.npz"), allow_pickle=True)
    return {str(i): (lm if f else None) for i, lm, f in zip(z["image_id"], z["landmarks"], z["found"])}


def sample(out, n, split="test"):
    """Test images that also have the VLM's occlusion explanation, so every method is compared on the same faces."""
    m = load_manifest(out)
    m = m[m.split == split].copy()
    m["image_id"] = m.image_id.astype(str)
    f = os.path.join(out, "explanations", f"occ_V2_{split}.csv")
    if os.path.exists(f):
        keep = set(pd.read_csv(f, usecols=["image_id"]).image_id.astype(str))
        m = m[m.image_id.isin(keep)]
    return m.sort_values("image_id").head(n)


def to_regions(sal, masks):
    sal = np.asarray(sal, float)
    return np.array([sal[mk].sum() for mk in masks]) / max(sal.sum(), 1e-12)


def occ_drops(out, source, split="test"):
    z = np.load(os.path.join(out, "explanations", f"occ_{source}_{split}.npz"), allow_pickle=True)
    t = z["P_full"].argmax(1)
    drop = z["P_full"][np.arange(len(t)), t][:, None] - z["P_drop"][np.arange(len(t)), :, t]
    return dict(zip(z["image_id"].astype(str), drop))


def agreement(df, out, source):
    occ = occ_drops(out, source)
    rho = []
    for r in df.itertuples(index=False):
        if r.image_id in occ:
            a = np.array([getattr(r, f"rel_{g}") for g in FINE])
            b = occ[r.image_id]
            if np.std(a) > 0 and np.std(b) > 0:
                rho.append(spearmanr(a, b).correlation)
    return float(np.nanmean(rho)) if rho else float("nan"), len(rho)


def write(out, method, df, summary):
    d = os.path.join(out, "explanations")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, f"native_{method}_test.csv"), index=False)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    with open(os.path.join(out, "results", "native_xai.md"), "a") as f:
        f.write(f"- {method}: {summary}\n")
    print(summary)


# ----------------------------------------------------------------------------- linear SVM (exact weights)
def run_linsvm(a):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from xfer.sources.hog_svm import GRID, SIZE, feature_matrix, feature_regions
    m = load_manifest(a.out)
    X = {s: feature_matrix(m[m.split == s].path.tolist()) for s in ("train", "val", "test")}
    Y = {s: m[m.split == s].label.values.astype(int) for s in ("train", "val", "test")}
    sc = StandardScaler().fit(X["train"])
    Z = {s: sc.transform(v) for s, v in X.items()}
    best = None
    for C in (0.003, 0.01, 0.03, 0.1):
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=42).fit(Z["train"], Y["train"])
        f1 = macro_f1(Y["val"], clf.predict(Z["val"]))
        print(f"linear SVM C={C}: val macro-F1 {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    cal = CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=42),
                                 method="sigmoid", cv=5).fit(Z["train"], Y["train"])
    for s in ("train", "val", "test"):
        path = pred_path(a.out, "CL", s)
        if os.path.exists(path):
            os.remove(path)
        append_predictions(path, m[m.split == s].image_id, Y[s], cal.predict_proba(Z[s]))
    # exact contributions: w_c,j * z_j summed over the features of each grid cell, then over each fine region
    cells = feature_regions()
    cell = SIZE // GRID
    lab = region_map(None, SIZE)                        # template regions (coarse) for the cell centres
    fine = fine_region_masks(None, SIZE)
    cell_fine = np.full(GRID * GRID, -1)
    for r in range(GRID):
        for c in range(GRID):
            y0, x0 = r * cell + cell // 2, c * cell + cell // 2
            hit = np.where(fine[:, y0, x0])[0]
            cell_fine[r * GRID + c] = hit[0] if len(hit) else -1
    test = m[m.split == "test"]
    pred = clf.predict(Z["test"])
    rows = []
    for i, iid in enumerate(test.image_id.astype(str)):
        contrib = clf.coef_[pred[i]] * Z["test"][i]
        per_cell = np.bincount(cells, weights=contrib, minlength=GRID * GRID)
        reg = np.array([per_cell[cell_fine == k].sum() for k in range(R)])
        rows.append(dict(image_id=iid, pred=int(pred[i]), **{f"rel_{g}": float(v) for g, v in zip(FINE, reg)}))
    df = pd.DataFrame(rows)
    acc, f1 = accuracy(Y["test"], cal.predict_proba(Z["test"]).argmax(1)), macro_f1(Y["test"], cal.predict_proba(Z["test"]).argmax(1))
    rho, n = agreement(df, a.out, "C")
    write(a.out, "linsvm", df, f"linear SVM (C={C}) test accuracy {100 * acc:.1f}%, macro-F1 {f1:.3f} (RBF SVM: 72.6%, "
          f"0.574); exact region contributions for all {len(df)} test images; Spearman with the SVM's occlusion "
          f"ranking {rho:.2f} on {n} images (note: occlusion is for the RBF SVM, the deployed expert)")


# ----------------------------------------------------------------------------- CLIP
def clip_eager():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from xfer.sources.clip_zs import TEMPLATES, _tensor
    name = "openai/clip-vit-large-patch14"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(name, attn_implementation="eager", torch_dtype=torch.float32).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(name)

    def text_emb(texts):
        with torch.no_grad():
            tok = proc(text=texts, return_tensors="pt", padding=True).to(dev)
            e = _tensor(model.get_text_features(**tok))
            return e / e.norm(dim=-1, keepdim=True)
    cls = []
    for c in CLASSES:
        adj, noun = CLIP_WORDS[c]
        e = text_emb([t.format(adj=adj, noun=noun) for t in TEMPLATES]).mean(0)
        cls.append(e / e.norm())
    return torch, model, proc, dev, torch.stack(cls), text_emb


def run_clip_au(a):
    from sklearn.metrics import roc_auc_score
    from xfer.reasoner import au_scores, au_thresholds
    torch, model, proc, dev, _, text_emb = clip_eager()
    aus = list(AU_PHRASES)
    T = text_emb([f"a photo of a face with {AU_PHRASES[x]}." for x in aus])
    m = load_manifest(a.out)
    m = m[m.split == "test"]
    S = []
    for i in range(0, len(m), 64):
        imgs = [Image.open(p).convert("RGB") for p in m.path.iloc[i:i + 64]]
        with torch.no_grad():
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
            from xfer.sources.clip_zs import _tensor
            e = _tensor(model.get_image_features(pixel_values=px))
            e = e / e.norm(dim=-1, keepdim=True)
            S.append((e @ T.T).cpu().numpy())
    S = np.concatenate(S)
    S = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)     # relative to the other phrases
    thr, _ = au_thresholds(a.out)
    g = pd.read_csv(os.path.join(a.out, "results", "D_geometry.csv"))
    g["image_id"] = g.image_id.astype(str)
    g = g.set_index("image_id").reindex(m.image_id.astype(str))
    sc = au_scores(g.reset_index())
    found = g.found.fillna(False).values.astype(bool)
    rows, aucs = [], {}
    for j, x in enumerate(aus):
        col = list(sc.columns).index(x)
        act = sc.values[found, col] >= thr[col]
        if 0 < act.mean() < 1:
            aucs[x] = roc_auc_score(act, S[found, j])
    df = pd.DataFrame(S, columns=[f"concept_{x}" for x in aus])
    df.insert(0, "image_id", m.image_id.astype(str).values)
    good = [f"{x} {v:.2f}" for x, v in sorted(aucs.items(), key=lambda t: -t[1])]
    write(a.out, "clip_au", df, f"CLIP AU-phrase probing on {len(df)} test faces; AUC against MediaPipe's measured "
          f"action units (0.5 = chance): " + ", ".join(good) + f"; mean {np.mean(list(aucs.values())):.2f}")


def run_clip_rel(a):
    torch, model, proc, dev, cls, _ = clip_eager()
    lms = landmarks(a.out)
    rows, t0 = [], time.time()
    for n, r in enumerate(sample(a.out, a.n).itertuples(index=False), 1):
        img = Image.open(r.path).convert("RGB")
        px = proc(images=[img], return_tensors="pt")["pixel_values"].to(dev)
        out = model.vision_model(pixel_values=px, output_attentions=True)
        atts = out.attentions
        for A in atts:
            A.retain_grad()
        e = model.visual_projection(out.pooler_output)
        e = e / e.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * e @ cls.T
        c = int(logits.argmax())
        model.zero_grad(set_to_none=True)
        logits[0, c].backward()
        Tn = atts[0].shape[-1]
        Rm = torch.eye(Tn, device=dev)
        for A in atts:
            Ab = (A.grad * A).clamp(min=0).mean(1)[0]
            Rm = Rm + Ab @ Rm
        rel = Rm[0, 1:].detach()
        s = int(round(rel.numel() ** 0.5))
        rel = torch.nn.functional.interpolate(rel.reshape(1, 1, s, s), size=img.size[::-1], mode="bilinear")[0, 0]
        masks = fine_region_masks(lms.get(str(r.image_id)), size=img.size[0])
        reg = to_regions(rel.cpu().numpy(), masks)
        rows.append(dict(image_id=str(r.image_id), pred=c, **{f"rel_{g}": float(v) for g, v in zip(FINE, reg)}))
        if n % 50 == 0:
            print(f"  {n}  {(time.time() - t0) / n:.2f}s per image", flush=True)
    df = pd.DataFrame(rows)
    rho, k = agreement(df, a.out, "A")
    top = df[[f"rel_{g}" for g in FINE]].idxmax(1).str[4:].value_counts(normalize=True).head(3).round(2).to_dict()
    write(a.out, "clip_rel", df, f"CLIP relevance maps (gradient x attention rollout) for {len(df)} test faces; most "
          f"relevant region {top}; Spearman with CLIP's occlusion ranking {rho:.2f} on {k} images")


# ----------------------------------------------------------------------------- POSTER++
def poster_forward(model, x_img, x_land, keep=None):
    """POSTER++'s forward with the landmark stream fed x_land and the image stream x_img (models/PosterV2_7cls.py).
    keep: list that receives the three cross-attention outputs (B, C, H, W) for gradient-weighted relevance."""
    import torch
    import torch.nn.functional as F
    mod = sys.modules[type(model).__module__]
    x_face = F.interpolate(x_land, size=112)
    f1, f2, f3 = model.face_landback(x_face)
    f3 = model.last_face_conv(f3)
    f1, f2, f3 = mod._to_channel_last(f1), mod._to_channel_last(f2), mod._to_channel_last(f3)
    q1, q2, q3 = (mod._to_query(f1, model.N[0], model.num_heads[0], model.dim_head[0]),
                  mod._to_query(f2, model.N[1], model.num_heads[1], model.dim_head[1]),
                  mod._to_query(f3, model.N[2], model.num_heads[2], model.dim_head[2]))
    i1, i2, i3 = model.ir_back(x_img)
    i1, i2, i3 = model.conv1(i1), model.conv2(i2), model.conv3(i3)
    w1, s1 = model.window1(i1)
    w2, s2 = model.window2(i2)
    w3, s3 = model.window3(i3)
    o1, o2, o3 = model.attn1(w1, q1), model.attn2(w2, q2), model.attn3(w3, q3)
    o1, o2, o3 = model.ffn1(o1, s1), model.ffn2(o2, s2), model.ffn3(o3, s3)
    o1, o2, o3 = mod._to_channel_first(o1), mod._to_channel_first(o2), mod._to_channel_first(o3)
    if keep is not None:
        for o in (o1, o2, o3):
            o.retain_grad()
        keep += [o1, o2, o3]
    o1 = model.embed_q(o1).flatten(2).transpose(1, 2)
    o2 = model.embed_k(o2).flatten(2).transpose(1, 2)
    o3 = model.embed_v(o3)
    return model.VIT(torch.cat([o1, o2, o3], dim=1))


def run_poster(a):
    import torch
    from xfer.sources.poster import PosterPredictor
    ckpt = a.ckpt or (glob.glob(os.path.join(a.work, "checkpoints_seed42", "best.pth")) +
                      glob.glob("/kaggle/**/checkpoints_seed42/best.pth", recursive=True))[0]
    P = PosterPredictor(a.work, ckpt)
    model, tf, dev = P.model.float().eval(), P.tf, P.dev
    lms = landmarks(a.out)
    rows_r, rows_g, t0 = [], [], time.time()
    for n, r in enumerate(sample(a.out, a.n).itertuples(index=False), 1):
        img = Image.open(r.path).convert("RGB")
        x = tf(img)[None].to(dev)
        x0 = tf(Image.fromarray(np.clip(blurred(img), 0, 255).astype(np.uint8)))[None].to(dev)
        masks = fine_region_masks(lms.get(str(r.image_id)), size=img.size[0])
        # gradient-weighted cross-attention relevance
        keep = []
        logits = poster_forward(model, x, x, keep)
        c = int(logits.argmax())
        model.zero_grad(set_to_none=True)
        logits[0, c].backward()
        cam = 0
        for o in keep:
            w = o.grad.mean((2, 3), keepdim=True)
            m_ = torch.relu((w * o).sum(1, keepdim=True))
            m_ = torch.nn.functional.interpolate(m_, size=img.size[::-1], mode="bilinear")[0, 0]
            cam = cam + m_ / (m_.max() + 1e-9)
        reg = to_regions(cam.detach().cpu().numpy(), masks)
        rows_r.append(dict(image_id=str(r.image_id), pred=c, **{f"rel_{g}": float(v) for g, v in zip(FINE, reg)}))
        # Integrated Gradients through the landmark stream only (image stream held at the real face)
        steps, grads = a.ig_steps, 0
        for k in range(1, steps + 1):
            xl = (x0 + (k / steps) * (x - x0)).detach().requires_grad_(True)
            out = poster_forward(model, x, xl)
            g, = torch.autograd.grad(out[0, c], xl)
            grads = grads + g
        ig = ((x - x0) * grads / steps).abs().sum(1)[0]
        ig = torch.nn.functional.interpolate(ig[None, None], size=img.size[::-1], mode="bilinear")[0, 0]
        reg = to_regions(ig.detach().cpu().numpy(), masks)
        rows_g.append(dict(image_id=str(r.image_id), pred=c, **{f"rel_{g}": float(v) for g, v in zip(FINE, reg)}))
        if n % 25 == 0:
            print(f"  {n}  {(time.time() - t0) / n:.2f}s per image", flush=True)
    for name, rows in (("poster_rel", rows_r), ("poster_ig", rows_g)):
        df = pd.DataFrame(rows)
        rho, k = agreement(df, a.out, "B")
        top = df[[f"rel_{g}" for g in FINE]].idxmax(1).str[4:].value_counts(normalize=True).head(3).round(2).to_dict()
        label = ("POSTER++ gradient-weighted cross-attention relevance" if name == "poster_rel" else
                 f"POSTER++ Integrated Gradients on the landmark stream ({a.ig_steps} steps, blurred baseline)")
        write(a.out, name, df, f"{label} for {len(df)} test faces; most relevant region {top}; Spearman with "
              f"POSTER++'s occlusion ranking {rho:.2f} on {k} images")


# ----------------------------------------------------------------------------- VLM answer-token relevance
def run_vlm_rel(a):
    from xfer.sources.vlm_zs import DEFAULT_MODEL, VLMZeroShot
    vlm = VLMZeroShot(a.vlm_model or DEFAULT_MODEL)
    lms = landmarks(a.out)
    rows, t0 = [], time.time()
    for n, r in enumerate(sample(a.out, a.n).itertuples(index=False), 1):
        img = Image.open(r.path).convert("RGB")
        try:
            sal, c = vlm.answer_relevance(img)
        except Exception as e:     # report and continue (e.g. out of memory on one image)
            print("skip", r.image_id, type(e).__name__, str(e)[:120]); continue
        reg = to_regions(sal, fine_region_masks(lms.get(str(r.image_id)), size=img.size[0]))
        rows.append(dict(image_id=str(r.image_id), pred=c, **{f"rel_{g}": float(v) for g, v in zip(FINE, reg)}))
        if n % 25 == 0:
            print(f"  {n}  {(time.time() - t0) / n:.2f}s per image", flush=True)
    df = pd.DataFrame(rows)
    rho, k = agreement(df, a.out, "V2")
    top = df[[f"rel_{g}" for g in FINE]].idxmax(1).str[4:].value_counts(normalize=True).head(3).round(2).to_dict()
    write(a.out, "vlm_rel", df, f"VLM answer-token relevance (input x gradient) for {len(df)} test faces; most "
          f"relevant region {top}; Spearman with the VLM's occlusion ranking {rho:.2f} on {k} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True, choices=["linsvm", "clip_au", "clip_rel", "poster", "vlm_rel"])
    ap.add_argument("--n", type=int, default=300, help="test faces for the gradient methods")
    ap.add_argument("--ig_steps", type=int, default=24)
    ap.add_argument("--work", default="/kaggle/working/poster")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--vlm_model", default=None)
    a = ap.parse_args()
    set_seed(42)
    {"linsvm": run_linsvm, "clip_au": run_clip_au, "clip_rel": run_clip_rel, "poster": run_poster,
     "vlm_rel": run_vlm_rel}[a.method](a)


if __name__ == "__main__":
    main()
