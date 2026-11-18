"""Stage 3, step 1 of the new design: the fused evidence of each test face becomes individuals of the EMONTO
extension (Face, Evidence from each Source, ActionUnit, FacialRegion, Decision), and the ontology is queried and
checked over them.

Reads explanations/fused_test.csv (from 21_new_framework.py), descriptions/V2_test_describe.csv and
results/D_geometry.csv; imports ontology/xfer_emonto.owl and writes ontology/instances_test.owl (all test faces).
Then:
  * SPARQL over the instances answers four checks of the reasoning (results/instantiation.md);
  * HermiT checks consistency and classifies a defined class (ConflictedFace) on a subset of --hermit faces.
Usage:  python scripts/22_instantiate.py --out OUT [--limit N] [--hermit 150]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from xfer import ontology as ONT
from xfer.claims import CLAIM_AUS, FIELDS, FIELD_REGION, options
from xfer.labels import CLASSES
from xfer.reasoner import au_scores, au_thresholds, aus

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERTS = ["V2", "A", "B", "C"]
NEW = "http://example.org/xfer/instances#"


def build(base_path, fused, desc, active, out_path, limit=None):
    from owlready2 import (DataProperty, FunctionalProperty, ObjectProperty, Thing, get_ontology)
    base = get_ontology("file://" + os.path.abspath(base_path)).load()
    inst = get_ontology(NEW)
    inst.imported_ontologies.append(base)
    with inst:
        class Face(Thing): comment = ["A static face image analysed by the framework."]
        class hasEvidence(ObjectProperty): domain = [Face]; range = [base.Evidence]
        class hasDecision(ObjectProperty, FunctionalProperty): domain = [Face]; range = [base.Decision]
        class supportsEmotion(ObjectProperty): domain = [base.Evidence]; range = [base.Emotion]
        class aboutRegion(ObjectProperty): domain = [base.Evidence]; range = [base.FacialRegion]
        class decidedEmotion(ObjectProperty): domain = [base.Decision]; range = [base.Emotion]
        class strength(DataProperty, FunctionalProperty): domain = [base.Evidence]; range = [float]
        class stance(DataProperty, FunctionalProperty): domain = [base.Evidence]; range = [str]
        class verified(DataProperty, FunctionalProperty): domain = [base.Evidence]; range = [bool]
        class firedRule(DataProperty): domain = [base.Decision]; range = [str]
        class agreementScore(DataProperty, FunctionalProperty): domain = [base.Decision]; range = [float]
        class annotatedEmotion(ObjectProperty, FunctionalProperty): domain = [Face]; range = [base.Emotion]
        class ConflictedFace(Face):
            equivalent_to = [Face & hasEvidence.some(stance.value("conflict"))]
        V2 = base.Source("V2")
        V2.comment = ["InternVL3-8B, the new frozen VLM (own vision encoder, not correlated with CLIP)."]
        emo = {c: base[c] for c in CLASSES}
        reg = {r: base[r] for r in ["forehead", "brows", "eyes", "nose", "cheeks", "mouth", "chin"]}
        src = {s: (V2 if s == "V2" else base[s]) for s in EXPERTS + ["D"]}
        au = {a: base[a] for a in aus()}
        rows = fused.head(limit) if limit else fused
        dmap = desc.set_index("image_id") if desc is not None else None
        for i, r in enumerate(rows.itertuples(index=False)):
            fid = str(r.image_id)
            face = Face(fid)
            face.annotatedEmotion = emo[r.label]
            regions = {g: getattr(r, f"region_{g}") for g in reg}
            top_region = max(regions, key=regions.get)
            for s in EXPERTS:
                e = base.Evidence(f"{fid}_{s}")
                e.fromSource = src[s]
                e.supportsEmotion = [emo[r.emotion]] if getattr(r, f"{s}_stance") == "support" else []
                e.stance = getattr(r, f"{s}_stance")
                e.strength = float(getattr(r, f"{s}_faith"))
                if regions[top_region] > 0:
                    e.aboutRegion = [reg[top_region]]
                face.hasEvidence.append(e)
            d = base.Evidence(f"{fid}_D")                               # measured action units (source D)
            d.fromSource = src["D"]
            d.claimsAU = [au[a] for a, on in zip(aus(), active[fid]) if on]
            d.stance = "measurement"
            face.hasEvidence.append(d)
            if dmap is not None and fid in dmap.index:                  # the VLM's claims, each verified or not
                row = dmap.loc[fid]
                act = dict(zip(aus(), active[fid]))
                for f in FIELDS:
                    for o in options(f, row.get(f, "")):
                        if o == "relaxed":
                            continue
                        c = base.Evidence(f"{fid}_V2claim_{f}_{o.replace(' ', '_')}")
                        c.fromSource = V2
                        aus_ = [ONT.SUBSTITUTE.get(a, a) for a in CLAIM_AUS[(f, o)]]
                        c.claimsAU = [au[a] for a in aus_ if a in au]
                        c.aboutRegion = [reg[FIELD_REGION[f]]]
                        c.verified = any(act.get(a, False) for a in aus_)
                        c.stance = "claim"
                        face.hasEvidence.append(c)
            kind = {"label": base.LabelDecision, "compound": base.BlendDecision, "abstain": base.Abstention}[r.decision]
            dec = kind(f"{fid}_decision")
            dec.decidedEmotion = [emo[r.emotion]] + ([emo[r.second]] if r.decision == "compound" else [])
            dec.agreementScore = float(r.agreement)
            dec.firedRule = [x.strip() for x in str(r.rules).split("|")]
            face.hasDecision = dec
    inst.save(file=out_path, format="rdfxml")
    return base, inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--owl", default=os.path.join(HERE, "ontology", "xfer_emonto.owl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hermit", type=int, default=150, help="faces in the subset checked by HermiT")
    ap.add_argument("--hermit_only", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    ONT.use_owl(a.owl)
    fused = pd.read_csv(os.path.join(a.out, "explanations", "fused_test.csv"))
    dpath = os.path.join(a.out, "descriptions", "V2_test_describe.csv")
    desc = pd.read_csv(dpath).drop_duplicates("image_id", keep="last") if os.path.exists(dpath) else None
    if desc is not None:
        desc["image_id"] = desc.image_id.astype(str)
    thr, _ = au_thresholds(a.out)
    g = pd.read_csv(os.path.join(a.out, "results", "D_geometry.csv"))
    g["image_id"] = g.image_id.astype(str)
    g = g[g.split == "test"].set_index("image_id")
    act = au_scores(g.reset_index()).values >= np.where(thr > 0, thr, np.inf)
    act[~g.found.fillna(False).values.astype(bool)] = False
    active = dict(zip(g.index, act))

    if a.hermit_only:
        from owlready2 import sync_reasoner
        os.makedirs(os.path.join(a.out, "ontology"), exist_ok=True)
        path = os.path.join(a.out, "ontology", f"instances_test_subset{a.hermit_only}.owl")
        sub = fused.head(a.hermit_only)
        _, si = build(a.owl, sub, desc, active, path)
        with si:
            sync_reasoner(infer_property_values=False, debug=0)      # HermiT; raises if inconsistent
        n_conf = len(list(si.ConflictedFace.instances()))
        expect = int((sub[[f"{s}_stance" for s in EXPERTS]] == "conflict").any(axis=1).sum())
        print(f"HermiT on {a.hermit_only} faces: consistent; it classifies {n_conf} of them as ConflictedFace (a face "
              f"with at least one expert in conflict with the decision, a class defined in the ontology, not computed "
              f"in Python); the fused table has {expect} such faces.")
        return

    os.makedirs(os.path.join(a.out, "ontology"), exist_ok=True)
    out_owl = os.path.join(a.out, "ontology", "instances_test.owl")
    base, inst = build(a.owl, fused, desc, active, out_owl, a.limit)
    from owlready2 import default_world
    X, I = "http://example.org/xfer/emonto#", NEW
    ns = base.base_iri
    q = lambda s: list(default_world.sparql(f"PREFIX x: <{ns}> PREFIX i: <{I}> " + s))
    L = ["# Stage 3, step 1: the evidence of every test face as EMONTO individuals", ""]
    n_faces = len(q("SELECT ?f WHERE { ?f a i:Face }"))
    n_ev = len(q("SELECT ?e WHERE { ?f i:hasEvidence ?e }"))
    L.append(f"Instantiated {n_faces} faces with {n_ev} evidence individuals (each expert's stance, the measured "
             f"action units of source D, and every VLM claim with its verification), {n_faces} decisions with the "
             f"rules that fired. File: <OUT>/ontology/instances_test.owl (imports xfer_emonto.owl).")
    L.append("")
    # 1. every compound decision names a confusable pair of the ontology
    comp = q("SELECT ?d ?a ?b WHERE { ?d a x:BlendDecision . ?d i:decidedEmotion ?a . ?d i:decidedEmotion ?b . "
             "FILTER(STR(?a) < STR(?b)) }")
    pairs = {frozenset(p) for p in ONT.CONFUSABLE}
    ok = sum(frozenset((a_.name, b_.name)) in pairs for _, a_, b_ in comp)
    L.append(f"Check 1 (confusability): {ok} of {len(comp)} compound decisions name a ConfusablePair of the ontology.")
    # 2. AU-pattern rule: decisions whose emotion's configuration includes at least one measured active AU
    sup = q("SELECT DISTINCT ?f WHERE { ?f i:hasDecision ?d . ?d i:decidedEmotion ?e . ?e x:hasConfiguration ?c . "
            "?c x:includesAU ?u . ?f i:hasEvidence ?m . ?m x:fromSource x:D . ?m x:claimsAU ?u }")
    L.append(f"Check 2 (AU pattern): for {len(sup)} of {n_faces} faces ({100 * len(sup) / max(n_faces, 1):.1f}%) "
             f"at least one measured action unit belongs to the configuration of the decided emotion.")
    # 3. precedence: VLM claims contradicted by the measurements
    bad = q("SELECT ?c WHERE { ?c x:fromSource i:V2 . ?c i:verified false }")
    good = q("SELECT ?c WHERE { ?c x:fromSource i:V2 . ?c i:verified true }")
    L.append(f"Check 3 (precedence, geometric over semantic): {len(good)} VLM claims confirmed by MediaPipe and "
             f"{len(bad)} contradicted; contradicted claims lower the VLM's weight in the decision.")
    # 4. abstentions and their agreement
    ab = q("SELECT ?f ?s WHERE { ?f i:hasDecision ?d . ?d a x:Abstention . ?d i:agreementScore ?s }")
    lab = q("SELECT ?f ?s WHERE { ?f i:hasDecision ?d . ?d a x:LabelDecision . ?d i:agreementScore ?s }")
    L.append(f"Check 4 (abstention): {len(ab)} abstentions, mean agreement "
             f"{np.mean([s for _, s in ab]) if ab else float('nan'):.2f}, against "
             f"{np.mean([s for _, s in lab]) if lab else float('nan'):.2f} for single labels.")
    # 5. correct decisions, queried from the instances
    corr = q("SELECT ?f WHERE { ?f i:hasDecision ?d . ?d a x:LabelDecision . ?d i:decidedEmotion ?e . "
             "?f i:annotatedEmotion ?e }")
    L.append(f"Check 5: {len(corr)} of {len(lab)} single-label decisions match the annotated emotion "
             f"({100 * len(corr) / max(len(lab), 1):.1f}%).")
    L.append("")

    # HermiT on a subset, in a fresh process (an OWL 2 DL reasoner on all 3,068 faces would take too long)
    if a.hermit:
        import subprocess
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--out", a.out, "--owl", a.owl,
                            "--hermit_only", str(a.hermit)], capture_output=True, text=True)
        L.append(r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip()
                 else f"HermiT on {a.hermit} faces: FAILED ({r.stderr.strip()[-400:]})")
    text = "\n".join(L)
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    open(os.path.join(a.out, "results", "instantiation.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
