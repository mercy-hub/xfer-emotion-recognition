"""Stage 1 of the new design: the SLM compensator for the frozen VLM (keep / change / abstain + explanation), with
field-level Shapley values as its explanation. Needs a GPU.

Input to the SLM (text, seven fields): the VLM's answer and confidence, its five described face fields (brows, eyes,
nose, cheeks, mouth) and the action units MediaPipe measured. Target (training subset only):
  keep              when the VLM is right;
  change to <c>     when it is wrong (c = the annotated emotion);
  abstain           when it is wrong, gives the annotated emotion under 2% and fewer than half its claims are
                    confirmed by MediaPipe (an ambiguous case for this design);
followed by an explanation: the VLM's own description of the annotated emotion when the emotion was given in the
prompt (descriptions/V2_train_given.csv) and at least 80% of its claims are confirmed; otherwise a sentence built
from the confirmed action units of that emotion.
Model: Qwen/Qwen2.5-1.5B-Instruct with LoRA (rank 16), trained on the 1,501 explained training images (failures
up-weighted by repetition). The test set is used once.
Outputs: predictions/V2s_<split>.csv (probabilities after the SLM), slm/slm_<split>.csv (decision, emotion,
explanation), slm/shapley_test.csv (exact Shapley value of each field for the SLM's decision on --shap_n images),
results/slm.md.
Usage:  python scripts/24_slm.py --out OUT
"""
import argparse
import itertools
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from xfer import ontology as ONT
from xfer.claims import FIELDS, claim_table
from xfer.io import append_predictions, pred_path, read_predictions, set_seed
from xfer.labels import CLASSES, K, prob_columns
from xfer.metrics import accuracy, macro_f1
from xfer.reasoner import au_scores, au_thresholds, aus

AU_NAMES = {"AU1": "inner brows raised", "AU2": "outer brows raised", "AU4": "brows lowered",
            "AU5": "upper eyelids raised", "AU6": "cheeks raised", "AU7": "eyelids tightened", "AU9": "nose wrinkled",
            "AU10": "upper lip raised", "AU12": "lip corners pulled up", "AU15": "lip corners pulled down",
            "AU17": "chin raised", "AU20": "lips stretched", "AU23": "lips pressed", "AU26": "jaw dropped"}
SHAP_FIELDS = ["answer", "brows", "eyes", "nose", "cheeks", "mouth", "action_units"]
MASK = "(not given)"


VLM = "V2"   # which VLM the SLM checks (V2 = InternVL3-8B; V = LLaVA-1.5 for the supervisor's matrix)


def load(out, split, thr, measurable, ids=None):
    p = read_predictions(pred_path(out, VLM, split))
    p["image_id"] = p.image_id.astype(str)
    d = pd.read_csv(os.path.join(out, "descriptions", f"{VLM}_{split}_describe.csv")).drop_duplicates("image_id", keep="last")
    d["image_id"] = d.image_id.astype(str)
    df = p.merge(d.drop(columns=["label"]), on="image_id", how="inner")
    if ids is not None:
        df = df[df.image_id.isin(ids)]
    df = df.sort_values("image_id").reset_index(drop=True)
    g = pd.read_csv(os.path.join(out, "results", "D_geometry.csv"))
    g["image_id"] = g.image_id.astype(str)
    g = g.set_index("image_id").reindex(df.image_id)
    act = au_scores(g.reset_index()).values >= np.where(thr > 0, thr, np.inf)
    act[~g.found.fillna(False).values.astype(bool)] = False
    df["aus"] = [[a for a, on in zip(aus(), row) if on] for row in act]
    cons, *_ = claim_table(df, act, aus(), measurable)
    df["claim_cons"] = cons
    return df, act


def fields(r):
    P = np.array([r[c] for c in prob_columns()])
    o = np.argsort(-P)
    return {"answer": f"{CLASSES[o[0]]} (confidence {P[o[0]]:.2f}); second choice {CLASSES[o[1]]} ({P[o[1]]:.2f})",
            **{f: (str(r.get(f, "")) if str(r.get(f, "")) not in ("", "nan") else "not described") for f in FIELDS},
            "action_units": ", ".join(f"{a} {AU_NAMES[a]}" for a in r["aus"]) or "none above threshold"}


def prompt_text(fl, tok):
    body = (f"VLM answer: {fl['answer']}\n"
            f"VLM description: brows {fl['brows']}; eyes {fl['eyes']}; nose {fl['nose']}; cheeks {fl['cheeks']}; "
            f"mouth {fl['mouth']}\n"
            f"Measured action units (MediaPipe): {fl['action_units']}\n"
            "Should the VLM's answer be kept, changed or should the case be referred to a person (abstain)? "
            "Reply as: Decision, Emotion, Explanation.")
    msgs = [{"role": "system", "content": "You check a vision-language model's facial emotion answer against the "
             "evidence. Emotions: surprise, fear, disgust, happiness, sadness, anger, neutral."},
            {"role": "user", "content": body}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def explanation(r, given, thr_idx):
    c = CLASSES[int(r["label"])]
    g = given.get(r["image_id"])
    if g is not None and g["cons"] >= 0.8 and str(g["reason"]) not in ("", "nan"):
        return str(g["reason"]).strip()
    cfg = [ONT.SUBSTITUTE.get(a, a) for a in ONT.AU_CONFIG[c]]
    seen = [AU_NAMES[a] for a in cfg if a in r["aus"]]
    if c == "neutral":
        return "No facial action is clearly active, which fits a neutral expression."
    if seen:
        return f"The {', '.join(seen)} {'is' if len(seen) == 1 else 'are'} visible, which fits {c}."
    return f"The overall configuration of the face fits {c} better than the VLM's answer."


def target(r, given):
    P = np.array([r[c] for c in prob_columns()])
    y, pred = int(r["label"]), int(P.argmax())
    expl = explanation(r, given, None)
    if pred == y:
        return f"Decision: keep\nEmotion: {CLASSES[y]}\nExplanation: {expl}", "keep"
    if P[y] < 0.02 and not np.isnan(r["claim_cons"]) and r["claim_cons"] < 0.5:
        return ("Decision: abstain\nEmotion: uncertain\nExplanation: The VLM's description is contradicted by the "
                "measured face, so the case should be checked by a person."), "abstain"
    return f"Decision: change\nEmotion: {CLASSES[y]}\nExplanation: {expl}", "change"


def parse(text, P):
    t = text.lower()
    m = re.search(r"decision:\s*(keep|change|abstain)", t)
    dec = m.group(1) if m else "keep"
    e = re.search(r"emotion:\s*([a-z]+)", t)
    emo = e.group(1) if e else ""
    from xfer.labels import NAME_ALIASES
    c = CLASSES.index(NAME_ALIASES[emo]) if emo in NAME_ALIASES else int(np.argmax(P))
    ex = re.search(r"explanation:\s*(.*)", text, flags=re.I | re.S)
    return dec, c, (ex.group(1).strip() if ex else "")


def to_probs(dec, c, P):
    if dec == "keep":
        return P
    if dec == "change":
        q = 0.3 * P
        q[c] += 0.7
        return q
    return 0.5 * P + 0.5 / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8, help="effective batch (micro-batches of --micro, accumulated)")
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--repeat_fail", type=int, default=3, help="repeat failure examples in training")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: at most N images per split")
    ap.add_argument("--shap_n", type=int, default=100)
    ap.add_argument("--vlm", default="V2", help="V2 (InternVL3-8B) or V (LLaVA-1.5)")
    a = ap.parse_args()
    global VLM
    VLM = a.vlm
    SD = "slm" if VLM == "V2" else f"slm_{VLM}"
    set_seed(42)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    thr, measurable = au_thresholds(a.out)
    subset = set(pd.read_csv(os.path.join(a.out, "descriptions", f"{VLM}_train_describe.csv"), usecols=["image_id"]).image_id.astype(str))
    tr, _ = load(a.out, "train", thr, measurable, subset)
    va, _ = load(a.out, "val", thr, measurable)
    te, _ = load(a.out, "test", thr, measurable)
    if a.limit:
        tr, va, te = tr.head(a.limit * 4), va.head(a.limit), te.head(a.limit)
    given = {}
    gpath = os.path.join(a.out, "descriptions", "V2_train_given.csv")
    g = pd.read_csv(gpath).drop_duplicates("image_id", keep="last")
    g["image_id"] = g.image_id.astype(str)
    gg = g.set_index("image_id").reindex(tr.image_id)
    act = np.array([[x in row for x in aus()] for row in tr.aus])
    gcons, *_ = claim_table(gg.reset_index(), act, aus(), measurable)
    for iid, c, rs in zip(tr.image_id, gcons, gg.reason.values):
        given[iid] = dict(cons=0 if np.isnan(c) else c, reason=rs)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)
    model.gradient_checkpointing_enable()          # T4 memory: recompute activations instead of storing them
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))

    # ------------------------------------------------------------ training data
    ex, kinds = [], []
    for r in tr.to_dict("records"):
        tgt, kind = target(r, given)
        p = prompt_text(fields(r), tok)
        reps = 1 if kind == "keep" else a.repeat_fail
        ex += [(p, tgt + tok.eos_token)] * reps
        kinds.append(kind)
    counts = pd.Series(kinds).value_counts().to_dict()
    print("training targets (images):", counts, "| examples after repetition:", len(ex), flush=True)
    rng = np.random.default_rng(42)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(ex) / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    model.train()
    tok.padding_side = "right"
    t0, step = time.time(), 0
    for ep in range(a.epochs):
        order = rng.permutation(len(ex))
        for i in range(0, len(order), a.batch):
            bb = [ex[j] for j in order[i:i + a.batch]]
            for mi in range(0, len(bb), a.micro):     # micro-batches with gradient accumulation (same effective batch)
                b = bb[mi:mi + a.micro]
                full = tok([p + t for p, t in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                labels = full.input_ids.clone()
                for k, (p, _) in enumerate(b):
                    n_p = len(tok(p, add_special_tokens=False).input_ids)
                    labels[k, :n_p] = -100
                labels[full.attention_mask == 0] = -100
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    loss = model(**full, labels=labels).loss
                scaler.scale(loss * len(b) / len(bb)).backward()
                del full, labels
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {ep + 1} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    model.config.use_cache = True
    torch.cuda.empty_cache() if dev == "cuda" else None
    tok.padding_side = "left"
    os.makedirs(os.path.join(a.out, SD), exist_ok=True)
    model.save_pretrained(os.path.join(a.out, SD, "lora"))

    # ------------------------------------------------------------ inference on val and test
    L = ["# SLM compensator (Qwen2.5-1.5B-Instruct + LoRA)", "",
         f"Training targets on {len(tr)} explained training images: {counts}.", ""]
    rows_sum = []
    for split, df in (("val", va), ("test", te)):
        outs = []
        for i in range(0, len(df), 32):
            chunk = df.iloc[i:i + 32]
            ps = [prompt_text(fields(r), tok) for r in chunk.to_dict("records")]
            enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                gen = model.generate(**enc, max_new_tokens=70, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        P = df[prob_columns()].values
        res = [parse(t, P[k]) for k, t in enumerate(outs)]
        dec = np.array([r[0] for r in res]); emo = np.array([r[1] for r in res])
        Q = np.stack([to_probs(d_, c_, P[k].copy()) for k, (d_, c_) in enumerate(zip(dec, emo))])
        path = pred_path(a.out, f"{VLM}s", split)
        if os.path.exists(path):
            os.remove(path)
        append_predictions(path, df.image_id, df.label, Q)
        pd.DataFrame(dict(image_id=df.image_id, label=df.label, vlm=P.argmax(1), decision=dec, emotion=emo,
                          explanation=[r[2] for r in res], raw=outs)).to_csv(
            os.path.join(a.out, SD, f"slm_{split}.csv"), index=False)
        y, v = df.label.values.astype(int), P.argmax(1)
        final = np.where(dec == "change", emo, v)
        acc_ans = (final == y)[dec != "abstain"].mean()
        fixed = int(((v != y) & (final == y) & (dec == "change")).sum())
        broken = int(((v == y) & (final != y) & (dec == "change")).sum())
        rows_sum.append(dict(split=split, images=len(df), vlm_accuracy=f"{100 * accuracy(y, v):.1f}%",
                             after_slm=f"{100 * accuracy(y, Q.argmax(1)):.1f}%", macro_F1=round(macro_f1(y, Q.argmax(1)), 3),
                             keep=f"{100 * (dec == 'keep').mean():.1f}%", change=f"{100 * (dec == 'change').mean():.1f}%",
                             abstain=f"{100 * (dec == 'abstain').mean():.1f}%", fixed=fixed, broken=broken,
                             accuracy_when_answering=f"{100 * acc_ans:.1f}%"))
    L += [pd.DataFrame(rows_sum).to_markdown(index=False), ""]

    # ------------------------------------------------------------ field-level Shapley (exact, 7 fields)
    words = {d: tok(" " + d, add_special_tokens=False).input_ids for d in ("keep", "change", "abstain")}
    sl = pd.read_csv(os.path.join(a.out, SD, "slm_test.csv"))
    sub = te.head(a.shap_n).reset_index(drop=True)
    decs = sl.set_index(sl.image_id.astype(str)).loc[sub.image_id, "decision"].values
    M = len(SHAP_FIELDS)
    coal = list(itertools.product([0, 1], repeat=M))
    shap_rows = []
    for k, r in enumerate(sub.to_dict("records")):
        fl = fields(r)
        texts = []
        for z in coal:
            f2 = {f: (fl[f] if z[j] else MASK) for j, f in enumerate(SHAP_FIELDS)}
            texts.append(prompt_text(f2, tok) + "Decision:")
        ids = words[decs[k]]
        v = np.zeros(len(coal))
        for i in range(0, len(texts), 16):
            batch = [t for t in texts[i:i + 16]]
            enc = tok([t + tok.decode(ids) for t in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                try:   # only the last positions are needed: avoids a (batch x length x 152k) logits tensor
                    lg = model(**enc, logits_to_keep=len(ids) + 1).logits
                except TypeError:
                    lg = model(**enc).logits[:, -(len(ids) + 1):]
                logp = torch.log_softmax(lg.float(), -1)
            n = len(ids)
            for j in range(len(batch)):
                lp = sum(logp[j, -n - 1 + t, ids[t]].item() for t in range(n))
                v[i + j] = lp
        val = dict(zip(coal, v))
        phi = np.zeros(M)
        for j in range(M):
            for z in coal:
                if z[j]:
                    continue
                s = sum(z)
                w = math.factorial(s) * math.factorial(M - s - 1) / math.factorial(M)
                z1 = tuple(1 if t == j else z[t] for t in range(M))
                phi[j] += w * (val[z1] - val[z])
        shap_rows.append(dict(image_id=r["image_id"], decision=decs[k], **{f"phi_{f}": phi[j] for j, f in enumerate(SHAP_FIELDS)}))
    sh = pd.DataFrame(shap_rows)
    sh.to_csv(os.path.join(a.out, SD, "shapley_test.csv"), index=False)
    mean_abs = sh[[f"phi_{f}" for f in SHAP_FIELDS]].abs().mean()
    share = (mean_abs / mean_abs.sum()).round(3).to_dict()
    L += [f"Field-level Shapley values (exact, {M} fields, {len(sh)} test images): mean share of the decision's "
          f"log-probability attributable to each field: " + ", ".join(f"{k[4:]} {v:.0%}" for k, v in share.items()), ""]
    for dname in ("keep", "change", "abstain"):
        s_ = sh[sh.decision == dname]
        if len(s_):
            m_ = s_[[f"phi_{f}" for f in SHAP_FIELDS]].mean().round(2).to_dict()
            L.append(f"- {dname} ({len(s_)} images): " + ", ".join(f"{k[4:]} {v:+.2f}" for k, v in m_.items()))
    ex_rows = sl.head(3)
    L += ["", "Examples:"] + [f"- [{CLASSES[int(r.label)]}] VLM {CLASSES[int(r.vlm)]} -> {r.raw!s}".replace("\n", " | ")
                              for r in ex_rows.itertuples()]
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    open(os.path.join(a.out, "results", "slm.md" if VLM == "V2" else f"slm_{VLM}.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
