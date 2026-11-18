"""Source C: hand-crafted HOG and LBP features with a support vector machine.

Features are computed on a 100x100 grayscale crop. The face is divided into a G x G grid of cells; HOG and a
uniform-LBP histogram are computed per cell, so every feature belongs to a known facial region. That is what lets
SHAP attributions be grouped by region later (section 3.4).
Probabilities come from Platt (sigmoid) scaling fitted by internal cross-validation on the training split.
"""
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

SIZE = 100
GRID = 5            # 5 x 5 regions of 20 x 20 pixels
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2  # uniform patterns


def load_gray(path, size=SIZE):
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def features(path):
    return features_from_gray(load_gray(path))


def features_from_image(img):
    g = np.asarray(img.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    return features_from_gray(g)


def features_from_gray(g):
    cell = SIZE // GRID
    # HOG with non-overlapping 1-cell blocks of 10x10 pixels: each region holds exactly 2 x 2 HOG cells
    h = hog(g, orientations=9, pixels_per_cell=(cell // 2, cell // 2), cells_per_block=(1, 1),
            block_norm="L2-Hys", feature_vector=False)  # shape (10, 10, 1, 1, 9)
    lbp = local_binary_pattern((g * 255).astype(np.uint8), LBP_P, LBP_R, method="uniform")
    feats = []
    for r in range(GRID):
        for c in range(GRID):
            # HOG blocks whose top-left cell falls inside region (r, c)
            hb = h[2 * r:2 * r + 2, 2 * c:2 * c + 2].ravel()
            patch = lbp[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            hist, _ = np.histogram(patch, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
            feats.append(np.concatenate([hb, hist]))
    return np.concatenate(feats).astype(np.float32)


def feature_regions():
    """Region index (0..GRID*GRID-1) of every feature, in the order produced by features()."""
    g = np.zeros((SIZE, SIZE), np.float32)
    cell = SIZE // GRID
    h = hog(g, orientations=9, pixels_per_cell=(cell // 2, cell // 2), cells_per_block=(1, 1), feature_vector=False)
    out = []
    for r in range(GRID):
        for c in range(GRID):
            n = h[2 * r:2 * r + 2, 2 * c:2 * c + 2].size + LBP_BINS
            out += [r * GRID + c] * n
    return np.array(out)


def feature_matrix(paths, n_jobs=-1):
    from joblib import Parallel, delayed
    return np.stack(Parallel(n_jobs=n_jobs)(delayed(features)(p) for p in paths))


def build_svm(C=10.0, gamma="scale", seed=42):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    svc = SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced", random_state=seed)
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False))


class SvmPredictor:
    """predict_proba_images for the region explainer."""
    def __init__(self, model_path):
        import joblib
        self.clf = joblib.load(model_path)

    def predict_proba_images(self, imgs):
        return self.clf.predict_proba(np.stack([features_from_image(im) for im in imgs]))
