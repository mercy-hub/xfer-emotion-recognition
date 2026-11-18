"""Canonical class order used everywhere in the project.

RAF-DB basic-expression ids are 1..7 in this order:
1 Surprise, 2 Fear, 3 Disgust, 4 Happiness, 5 Sadness, 6 Anger, 7 Neutral.
Index k in every probability vector corresponds to CLASSES[k].
Check this against a few images per class (scripts/01_prepare_data.py writes a contact sheet) before trusting it.
"""

CLASSES = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
K = len(CLASSES)
RAF_ID_TO_IDX = {i + 1: i for i in range(K)}

# Folder or CSV names that different dataset copies use for the same class.
NAME_ALIASES = {
    "surprise": "surprise", "surprised": "surprise",
    "fear": "fear", "fearful": "fear", "scared": "fear",
    "disgust": "disgust", "disgusted": "disgust",
    "happiness": "happiness", "happy": "happiness", "joy": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "anger": "anger", "angry": "anger",
    "neutral": "neutral", "calm": "neutral",
}

# Words offered to the generative VLM (capitalised, one word each).
VLM_WORDS = {"surprise": "Surprise", "fear": "Fear", "disgust": "Disgust", "happiness": "Happiness",
             "sadness": "Sadness", "anger": "Anger", "neutral": "Neutral"}

# Wording used in CLIP prompts: (adjective, noun phrase).
CLIP_WORDS = {
    "surprise": ("surprised", "surprise"),
    "fear": ("fearful", "fear"),
    "disgust": ("disgusted", "disgust"),
    "happiness": ("happy", "happiness"),
    "sadness": ("sad", "sadness"),
    "anger": ("angry", "anger"),
    "neutral": ("neutral", "a neutral expression"),
}


def to_index(value):
    """Map a raw label (RAF id 1..7, 0-based int, or a class name) to a 0-based class index."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v.isdigit():
            return to_index(int(v))
        if v in NAME_ALIASES:
            return CLASSES.index(NAME_ALIASES[v])
        raise ValueError(f"Unknown label name: {value!r}")
    v = int(value)
    if v in RAF_ID_TO_IDX:
        return RAF_ID_TO_IDX[v]
    raise ValueError(f"Label id out of range 1..7: {value!r}")


def prob_columns():
    return [f"p_{c}" for c in CLASSES]
