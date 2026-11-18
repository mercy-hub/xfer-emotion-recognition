"""Step 3 of the new design: the VLM's written claims, checked against MediaPipe's measurements.

The VLM describes brows, eyes, nose, cheeks and mouth with fixed options (xfer/sources/vlm_zs.py). Each option is a
claim about action units. A positive claim ("brows raised") is verified when one of its action units is active on
the face (blendshape score at or above its 75th training percentile); "relaxed" is verified when none of the
field's action units is active. Unmeasurable action units use the ontology's substitutes (AU6 -> AU7, AU9 -> AU10).
"""
import re

import numpy as np

from . import ontology as ONT

# option -> action units it asserts (any one active verifies the claim)
CLAIM_AUS = {
    ("brows", "raised"): ["AU1", "AU2"],
    ("brows", "lowered"): ["AU4"],
    ("brows", "drawn together"): ["AU4"],
    ("eyes", "wide open"): ["AU5"],
    ("eyes", "narrowed"): ["AU7", "AU6"],
    ("eyes", "closed"): ["AU7"],
    ("nose", "wrinkled"): ["AU9", "AU10"],
    ("cheeks", "raised"): ["AU6", "AU12"],
    ("mouth", "smiling"): ["AU12"],
    ("mouth", "open"): ["AU26"],
    ("mouth", "corners down"): ["AU15"],
    ("mouth", "lips pressed"): ["AU23"],
    ("mouth", "upper lip raised"): ["AU10"],
}
FIELDS = ["brows", "eyes", "nose", "cheeks", "mouth"]
FIELD_AUS = {f: sorted({a for (g, _), aus in CLAIM_AUS.items() if g == f for a in aus}) for f in FIELDS}
# facial regions (coarse, xfer.regions.REGIONS) each field talks about
FIELD_REGION = {"brows": "brows", "eyes": "eyes", "nose": "nose", "cheeks": "cheeks", "mouth": "mouth"}


def options(field, text):
    """All options named in one field's text, e.g. 'open, upper lip raised' -> ['open', 'upper lip raised']."""
    t = str(text).lower()
    if t in ("", "nan"):
        return []
    found = [o for (f, o) in CLAIM_AUS if f == field and re.search(r"\b" + re.escape(o) + r"\b", t)]
    if not found and "relaxed" in t:
        return ["relaxed"]
    return found


def resolve(au, measurable_map):
    return ONT.SUBSTITUTE.get(au, au) if not measurable_map.get(au, True) else au


def check_claims(row, active, measurable_map):
    """row: dict-like with the five fields; active: dict AU -> bool for this face.
    Returns (n_claims, n_verified, list of (field, option, verified))."""
    out = []
    for f in FIELDS:
        for o in options(f, row.get(f, "")):
            if o == "relaxed":
                ok = not any(active.get(resolve(a, measurable_map), False) for a in FIELD_AUS[f])
            else:
                ok = any(active.get(resolve(a, measurable_map), False) for a in CLAIM_AUS[(f, o)])
            out.append((f, o, bool(ok)))
    return len(out), sum(v for *_, v in out), out


def claim_table(desc, au_active, au_names, measurable):
    """desc: DataFrame of VLM descriptions (image_id + five fields); au_active: (n, A) bool aligned with desc.
    Returns arrays: consistency (share of claims verified; NaN if no claim), n_claims, positive-claim fields,
    and per-field verification flags (n, 5) with NaN where the field made no claim."""
    mmap = dict(zip(au_names, measurable))
    n = len(desc)
    cons, ncl = np.full(n, np.nan), np.zeros(n, int)
    flags = np.full((n, len(FIELDS)), np.nan)
    pos_fields = []
    for i, row in enumerate(desc.to_dict("records")):
        act = dict(zip(au_names, au_active[i]))
        k, v, items = check_claims(row, act, mmap)
        ncl[i] = k
        if k:
            cons[i] = v / k
        pf = []
        for f, o, ok in items:
            j = FIELDS.index(f)
            flags[i, j] = ok if np.isnan(flags[i, j]) else min(flags[i, j], ok)
            if o != "relaxed":
                pf.append(f)
        pos_fields.append(sorted(set(pf)))
    return cons, ncl, pos_fields, flags
