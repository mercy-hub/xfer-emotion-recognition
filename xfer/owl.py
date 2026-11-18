"""The extended EMONTO ontology as an OWL 2 file (section 3.5), its consistency check, competency questions, and the
loader that gives the reasoner its knowledge.

The published EMONTO OWL file could not be located, so the EMONTO classes this project reuses (Emotion, Source,
Evidence) are re-declared here from their published description and annotated as such. The extension adds
ActionUnit, FacialRegion, AUConfiguration, Measurement (a MediaPipe blendshape), ConfusablePair and Decision, and
the properties that link them. HermiT (through Owlready2) checks consistency and classifies two defined classes.

    python -m xfer.owl build  ontology/xfer_emonto.owl     # write the file from xfer/ontology.py
    python -m xfer.owl check  ontology/xfer_emonto.owl     # HermiT consistency + competency questions
"""
import os
import sys

IRI = "http://example.org/xfer-emonto#"


def build(path):
    from owlready2 import (AllDifferent, AllDisjoint, DataProperty, FunctionalProperty, ObjectProperty,
                           SymmetricProperty, Thing, get_ontology)
    from . import ontology as K
    from .labels import CLASSES
    from .regions import REGIONS

    onto = get_ontology(IRI)
    with onto:
        class Emotion(Thing): comment = ["Re-declared from EMONTO (Graterol et al., 2021): an emotion category."]
        class Source(Thing): comment = ["Re-declared from EMONTO: a recognition method that produces evidence."]
        class Evidence(Thing): comment = ["Re-declared from EMONTO, extended: a symbolic claim made by a source."]
        class ActionUnit(Thing): comment = ["Extension: a FACS action unit (Ekman and Friesen, 1978)."]
        class FacialRegion(Thing): comment = ["Extension: one of the seven regions used by the region explanations."]
        class AUConfiguration(Thing): comment = ["Extension: the prototype action units of an emotion (Table 11)."]
        class Measurement(Thing): comment = ["Extension: a MediaPipe blendshape score read as an action-unit proxy."]
        class ConfusablePair(Thing): comment = ["Extension: two emotions that are easily mistaken for each other."]
        class Decision(Thing): comment = ["Extension: the output of the reasoner."]
        class LabelDecision(Decision): pass
        class BlendDecision(Decision): pass
        class Abstention(Decision): pass
        AllDisjoint([Emotion, Source, Evidence, ActionUnit, FacialRegion, AUConfiguration, Measurement,
                     ConfusablePair, Decision])
        AllDisjoint([LabelDecision, BlendDecision, Abstention])

        class hasConfiguration(ObjectProperty, FunctionalProperty): domain = [Emotion]; range = [AUConfiguration]
        class includesAU(ObjectProperty): domain = [AUConfiguration]; range = [ActionUnit]
        class movesRegion(ObjectProperty): domain = [ActionUnit]; range = [FacialRegion]
        class measuredBy(ObjectProperty): domain = [ActionUnit]; range = [Measurement]
        class substitutedBy(ObjectProperty, FunctionalProperty): domain = [ActionUnit]; range = [ActionUnit]
        class pairMember(ObjectProperty): domain = [ConfusablePair]; range = [Emotion]
        class correlatedWith(ObjectProperty, SymmetricProperty): domain = [Source]; range = [Source]
        class claimsAU(ObjectProperty): domain = [Evidence]; range = [ActionUnit]
        class fromSource(ObjectProperty, FunctionalProperty): domain = [Evidence]; range = [Source]
        class blendshapeName(DataProperty): domain = [Measurement]; range = [str]
        class measurable(DataProperty, FunctionalProperty): domain = [ActionUnit]; range = [bool]
        class auWeight(DataProperty, FunctionalProperty): domain = [AUConfiguration]; range = [float]
        class classIndex(DataProperty, FunctionalProperty): domain = [Emotion]; range = [int]

        regions = {r: FacialRegion(r) for r in REGIONS}
        aus = {}
        for au in K.AU_BLENDSHAPES:
            a = ActionUnit(au)
            a.movesRegion = [regions[r] for r in K.AU_REGIONS[au]]
            for g, names in enumerate(K.AU_BLENDSHAPES[au]):
                m = Measurement(f"{au}_measure{g}")
                m.blendshapeName = list(names)
                a.measuredBy.append(m)
            a.measurable = au not in K.SUBSTITUTE
            aus[au] = a
        for au, sub in K.SUBSTITUTE.items():
            aus[au].substitutedBy = aus[sub]
        emos = {}
        for i, c in enumerate(CLASSES):
            e = Emotion(c)
            e.classIndex = i
            cfg = AUConfiguration(f"{c}_configuration")
            cfg.includesAU = [aus[a] for a in K.AU_CONFIG[c]]
            cfg.auWeight = 1.0
            e.hasConfiguration = cfg
            emos[c] = e
        for a, b in K.CONFUSABLE:
            p = ConfusablePair(f"{a}_{b}")
            p.pairMember = [emos[a], emos[b]]
        src = {s: Source(s) for s in ["V", "A", "B", "C", "D"]}
        src["V"].correlatedWith = [src["A"]]                 # LLaVA-1.5 uses a CLIP image encoder
        AllDifferent(list(regions.values()) + list(aus.values()) + list(emos.values()) + list(src.values()))

        # defined classes for the reasoner to classify (competency questions 5 and 6)
        class BrowEmotion(Emotion):
            equivalent_to = [Emotion & hasConfiguration.some(includesAU.some(movesRegion.value(regions["brows"])))]
        class MouthOnlyAU(ActionUnit):
            equivalent_to = [ActionUnit & movesRegion.only(FacialRegion) & movesRegion.value(regions["mouth"])]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    onto.save(file=path, format="rdfxml")
    return onto


def load(path):
    from owlready2 import get_ontology
    return get_ontology("file://" + os.path.abspath(path)).load()


def knowledge(path):
    """Dictionaries the reasoner uses, read from the OWL file (same shapes as xfer/ontology.py)."""
    onto = load(path)
    emos = sorted(onto.Emotion.instances(), key=lambda e: e.classIndex)
    cfg = {e.name: sorted(a.name for a in e.hasConfiguration.includesAU) for e in emos}
    regions = {a.name: sorted(r.name for r in a.movesRegion) for a in onto.ActionUnit.instances()}
    sub = {a.name: a.substitutedBy.name for a in onto.ActionUnit.instances() if a.substitutedBy}
    blend = {a.name: [list(m.blendshapeName) for m in sorted(a.measuredBy, key=lambda m: m.name)]
             for a in onto.ActionUnit.instances()}
    pairs = [tuple(sorted(m.name for m in p.pairMember)) for p in onto.ConfusablePair.instances()]
    return dict(AU_CONFIG=cfg, AU_REGIONS=regions, SUBSTITUTE=sub, AU_BLENDSHAPES=blend, CONFUSABLE=pairs)


def check(path):
    from owlready2 import default_world, sync_reasoner
    onto = load(path)
    with onto:
        sync_reasoner(infer_property_values=True, debug=0)            # HermiT; raises if inconsistent
    out = ["HermiT: the ontology is consistent",
           f"classes {len(list(onto.classes()))}, object properties {len(list(onto.object_properties()))}, "
           f"data properties {len(list(onto.data_properties()))}, individuals {len(list(onto.individuals()))}"]
    q = lambda s: list(default_world.sparql(f"PREFIX x: <{IRI}> " + s))
    out.append("CQ1 Which emotions does action unit 12 support? " + ", ".join(
        r[0].name for r in q("SELECT ?e WHERE { ?e x:hasConfiguration ?c . ?c x:includesAU x:AU12 }")))
    out.append("CQ2 Which facial regions does the fear configuration move? " + ", ".join(sorted({
        r[0].name for r in q("SELECT ?r WHERE { x:fear x:hasConfiguration ?c . ?c x:includesAU ?a . ?a x:movesRegion ?r }")})))
    out.append("CQ3 Which emotion pairs are confusable? " + "; ".join(
        f"{r[0].name}-{r[1].name}" for r in q("SELECT ?a ?b WHERE { ?p x:pairMember ?a . ?p x:pairMember ?b . "
                                             "FILTER(STR(?a) < STR(?b)) }")))
    out.append("CQ4 Which action units cannot be measured here, and what replaces them? " + "; ".join(
        f"{r[0].name} -> {r[1].name}" for r in q("SELECT ?a ?s WHERE { ?a x:substitutedBy ?s }")))
    out.append("CQ5 Which emotions use the brows (inferred by HermiT)? " + ", ".join(
        sorted(e.name for e in onto.BrowEmotion.instances())))
    out.append("CQ6 Which sources is A correlated with (inferred from the symmetric property)? " + ", ".join(
        s.name for s in onto.A.correlatedWith))
    return out


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "build":
        build(path); print("wrote", path)
    else:
        print("\n".join(check(path)))
