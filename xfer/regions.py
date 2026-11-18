"""Facial regions and landmark geometry (source D).

Seven regions R are used by every region explanation (section 3.4): forehead, brows, eyes, nose, cheeks, mouth, chin.
With landmarks (MediaPipe face mesh, 478 points) each region is the convex hull of its mesh points; without them a
fixed template for 100 x 100 aligned RAF-DB faces is used. Regions never overlap: later regions in DRAW_ORDER win.
"""
import numpy as np

REGIONS = ["forehead", "brows", "eyes", "nose", "cheeks", "mouth", "chin"]
M = len(REGIONS)
SIZE = 100

# MediaPipe face-mesh indices (canonical 468-point topology; 468-477 are the irises)
MESH = {
    "forehead": [[10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 151, 9, 108, 69, 104, 68, 71, 337, 299, 333, 298, 301]],
    "brows": [[70, 63, 105, 66, 107, 55, 65, 52, 53, 46], [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]],
    "eyes": [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
             [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]],
    "nose": [[168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 48, 64, 129, 358]],
    "cheeks": [[116, 117, 118, 119, 100, 142, 203, 206, 207, 187, 147, 123, 50, 36],
               [345, 346, 347, 348, 329, 371, 423, 426, 427, 411, 376, 352, 280, 266]],
    "mouth": [[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 57, 287]],
    "chin": [[152, 148, 176, 149, 150, 377, 400, 378, 379, 18, 200, 199, 175, 83, 313, 201, 421, 208, 428]],
}
DRAW_ORDER = ["forehead", "cheeks", "chin", "nose", "brows", "eyes", "mouth"]

# Template boxes (x0, y0, x1, y1) for 100 x 100 aligned RAF-DB crops, used when no landmarks were found
TEMPLATE = {
    "forehead": [(20, 0, 80, 20)],
    "brows": [(18, 20, 82, 33)],
    "eyes": [(18, 33, 82, 48)],
    "cheeks": [(10, 48, 36, 78), (64, 48, 90, 78)],
    "nose": [(36, 45, 64, 68)],
    "mouth": [(28, 68, 72, 86)],
    "chin": [(28, 86, 72, 100)],
}


def _hull_fill(points, size):
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon
    mask = np.zeros((size, size), bool)
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return mask
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        return mask
    rr, cc = polygon(h[:, 1], h[:, 0], shape=(size, size))
    mask[rr, cc] = True
    return mask


def region_map(landmarks=None, size=SIZE):
    """Integer map (size x size): region index 0..M-1, or -1 for pixels outside every region.

    landmarks: array (478, 2) of (x, y) pixel coordinates in this image, or None for the template."""
    lab = -np.ones((size, size), int)
    for name in DRAW_ORDER:
        k = REGIONS.index(name)
        if landmarks is not None:
            for idx in MESH[name]:
                lab[_hull_fill(landmarks[idx], size)] = k
        else:
            s = size / SIZE
            for x0, y0, x1, y1 in TEMPLATE[name]:
                lab[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = k
    return lab


def region_masks(landmarks=None, size=SIZE):
    lab = region_map(landmarks, size)
    return np.stack([lab == k for k in range(M)])


# ----------------------------------------------------------------------------- geometry (source D)
GEOMETRY = ["inner_brow_raise", "outer_brow_raise", "brow_gap", "eye_opening", "mouth_opening", "mouth_width",
            "lip_corner_lift", "nose_to_lip"]


def geometry(lm):
    """Ratios of landmark distances (Equation 3.1), each divided by the inter-ocular distance so that they do not
    depend on face size. lm: (478, 2) pixel coordinates (y grows downwards)."""
    lm = np.asarray(lm, float)
    eye_l = lm[[33, 133]].mean(0)
    eye_r = lm[[362, 263]].mean(0)
    iod = np.linalg.norm(eye_r - eye_l) + 1e-6
    vd = lambda a, b: abs(lm[a][1] - lm[b][1]) / iod
    hd = lambda a, b: np.linalg.norm(lm[a] - lm[b]) / iod
    lip_centre_y = (lm[0][1] + lm[17][1]) / 2
    corners_y = (lm[61][1] + lm[291][1]) / 2
    return dict(
        inner_brow_raise=(vd(107, 159) + vd(336, 386)) / 2,    # AU1: inner brow above upper eyelid
        outer_brow_raise=(vd(70, 33) + vd(300, 263)) / 2,      # AU2: outer brow above outer eye corner
        brow_gap=hd(107, 336),                                 # AU4 lowers and draws the brows together
        eye_opening=(vd(159, 145) + vd(386, 374)) / 2,         # AU5 widens, AU6/7 narrow
        mouth_opening=vd(13, 14),                              # AU25/26/27
        mouth_width=hd(61, 291),                               # AU12/AU20 widen
        lip_corner_lift=(lip_centre_y - corners_y) / iod,      # AU12 > 0, AU15 < 0
        nose_to_lip=vd(2, 0),                                  # AU9/AU10 shorten it
    )


BLENDSHAPE_AU = {  # MediaPipe blendshape -> nearest FACS action unit (approximate correspondence)
    "browInnerUp": "AU1", "browOuterUpLeft": "AU2", "browOuterUpRight": "AU2", "browDownLeft": "AU4",
    "browDownRight": "AU4", "eyeWideLeft": "AU5", "eyeWideRight": "AU5", "cheekSquintLeft": "AU6",
    "cheekSquintRight": "AU6", "eyeSquintLeft": "AU7", "eyeSquintRight": "AU7", "noseSneerLeft": "AU9",
    "noseSneerRight": "AU9", "mouthUpperUpLeft": "AU10", "mouthUpperUpRight": "AU10", "mouthSmileLeft": "AU12",
    "mouthSmileRight": "AU12", "mouthDimpleLeft": "AU14", "mouthDimpleRight": "AU14", "mouthFrownLeft": "AU15",
    "mouthFrownRight": "AU15", "mouthLowerDownLeft": "AU16", "mouthLowerDownRight": "AU16",
    "mouthShrugLower": "AU17", "mouthPucker": "AU18", "mouthStretchLeft": "AU20", "mouthStretchRight": "AU20",
    "mouthPressLeft": "AU24", "mouthPressRight": "AU24", "jawOpen": "AU26",
}


# ----------------------------------------------------------------------------- fine regions (new design, step 2)
# Ten regions for the shared occlusion explanation: the paired regions are split into the person's right (_r) and
# left (_l) halves. MediaPipe's first mesh list of each pair (points 33, 70, 116, ...) is the person's right, which
# is the image-left half of the template. COARSE_OF maps each fine region back to the seven regions above.
FINE = ["forehead", "brow_l", "brow_r", "eye_l", "eye_r", "nose", "cheek_l", "cheek_r", "mouth", "chin"]
COARSE_OF = {"forehead": "forehead", "brow_l": "brows", "brow_r": "brows", "eye_l": "eyes", "eye_r": "eyes",
             "nose": "nose", "cheek_l": "cheeks", "cheek_r": "cheeks", "mouth": "mouth", "chin": "chin"}
_FINE_PARTS = {"forehead": ("forehead", 0), "brow_r": ("brows", 0), "brow_l": ("brows", 1), "eye_r": ("eyes", 0),
               "eye_l": ("eyes", 1), "nose": ("nose", 0), "cheek_r": ("cheeks", 0), "cheek_l": ("cheeks", 1),
               "mouth": ("mouth", 0), "chin": ("chin", 0)}


def fine_region_masks(landmarks=None, size=SIZE):
    """(10, size, size) bool masks, one per FINE region; non-overlapping (drawn in the coarse DRAW_ORDER)."""
    lab = -np.ones((size, size), int)
    for coarse in DRAW_ORDER:
        for fname, (c, part) in _FINE_PARTS.items():
            if c != coarse:
                continue
            k = FINE.index(fname)
            if landmarks is not None:
                lab[_hull_fill(np.asarray(landmarks)[MESH[c][part]], size)] = k
            else:
                s = size / SIZE
                boxes = TEMPLATE[c]
                if len(boxes) == 2:                       # cheeks: one box per side
                    boxes = [boxes[part]]
                elif c in ("brows", "eyes"):              # one wide box: split it at the vertical midline
                    x0, y0, x1, y1 = boxes[0]
                    mid = (x0 + x1) // 2
                    boxes = [(x0, y0, mid, y1)] if part == 0 else [(mid, y0, x1, y1)]
                for x0, y0, x1, y1 in boxes:
                    lab[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = k
    return np.stack([lab == k for k in range(len(FINE))])
