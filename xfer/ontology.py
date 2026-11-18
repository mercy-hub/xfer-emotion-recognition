"""The ontology knowledge used by the reasoner: action-unit configurations, the facial regions of each action unit,
confusable pairs, and how MediaPipe blendshapes stand in for action units.

The configurations are the core ones of Table 11 (provisional). Two substitutions are forced by the measurements on
100 x 100 RAF-DB faces, where MediaPipe's cheekSquint and noseSneer scores are always close to zero:
AU6 (cheek raiser) is read from eyeSquint, the same orbital tightening that AU7 uses, and AU9 (nose wrinkler) is
replaced by AU10 (upper lip raiser), the variant that Table 11 lists for disgust.
"""
from .labels import CLASSES

AU_CONFIG = {
    "happiness": ["AU6", "AU12"],
    "sadness": ["AU1", "AU4", "AU15"],
    "surprise": ["AU1", "AU2", "AU5", "AU26"],
    "fear": ["AU1", "AU2", "AU4", "AU5", "AU7", "AU20", "AU26"],
    "anger": ["AU4", "AU5", "AU7", "AU23"],
    "disgust": ["AU9", "AU15", "AU17"],
    "neutral": [],
}

# blendshape(s) measuring each action unit (left and right are averaged; the maximum is taken across names)
AU_BLENDSHAPES = {
    "AU1": [["browInnerUp"]],
    "AU2": [["browOuterUpLeft", "browOuterUpRight"]],
    "AU4": [["browDownLeft", "browDownRight"]],
    "AU5": [["eyeWideLeft", "eyeWideRight"]],
    "AU6": [["cheekSquintLeft", "cheekSquintRight"]],
    "AU7": [["eyeSquintLeft", "eyeSquintRight"]],
    "AU9": [["noseSneerLeft", "noseSneerRight"]],
    "AU10": [["mouthUpperUpLeft", "mouthUpperUpRight"]],
    "AU12": [["mouthSmileLeft", "mouthSmileRight"]],
    "AU15": [["mouthFrownLeft", "mouthFrownRight"]],
    "AU17": [["mouthShrugLower"]],
    "AU20": [["mouthStretchLeft", "mouthStretchRight"]],
    "AU23": [["mouthPressLeft", "mouthPressRight"]],          # AU24 lip pressor, the measurable neighbour of AU23
    "AU26": [["jawOpen"], ["mouthLowerDownLeft", "mouthLowerDownRight"]],
}
SUBSTITUTE = {"AU6": "AU7", "AU9": "AU10"}                    # used when an action unit is not measurable

# facial regions (xfer.regions.REGIONS) that each action unit moves
AU_REGIONS = {
    "AU1": ["brows", "forehead"], "AU2": ["brows", "forehead"], "AU4": ["brows"], "AU5": ["eyes"],
    "AU6": ["eyes", "cheeks"], "AU7": ["eyes"], "AU9": ["nose"], "AU10": ["nose", "mouth"], "AU12": ["mouth", "cheeks"],
    "AU15": ["mouth", "chin"], "AU17": ["chin"], "AU20": ["mouth"], "AU23": ["mouth"], "AU26": ["mouth", "chin"],
}

CONFUSABLE = [("fear", "surprise"), ("anger", "disgust"), ("sadness", "neutral")]


def class_regions(c):
    regs = []
    for au in AU_CONFIG[c]:
        for r in AU_REGIONS[au]:
            if r not in regs:
                regs.append(r)
    return regs


def confusable_index():
    return {frozenset((CLASSES.index(a), CLASSES.index(b))) for a, b in CONFUSABLE}


SOURCE_OF_KNOWLEDGE = "python"


def use_owl(path):
    """Replace the dictionaries above with the knowledge read from the OWL file, so that the reasoner's rules come
    from the ontology itself. Returns the path used."""
    global AU_CONFIG, AU_REGIONS, SUBSTITUTE, AU_BLENDSHAPES, CONFUSABLE, SOURCE_OF_KNOWLEDGE
    from .owl import knowledge
    k = knowledge(path)
    order = list(AU_BLENDSHAPES)                      # keep the column order stable
    AU_CONFIG = {c: sorted(k["AU_CONFIG"][c], key=order.index) for c in CLASSES}
    AU_REGIONS = k["AU_REGIONS"]
    SUBSTITUTE = k["SUBSTITUTE"]
    AU_BLENDSHAPES = {a: k["AU_BLENDSHAPES"][a] for a in order}
    CONFUSABLE = k["CONFUSABLE"]
    SOURCE_OF_KNOWLEDGE = path
    return path
