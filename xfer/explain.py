"""Region explanations shared by every source: Kernel SHAP and LIME from one shared sample of masked images.

For an image with M = 7 facial regions, 32 coalitions z in {0,1}^M are evaluated once:
  all regions kept, no region kept, the 7 single regions, the 7 single removals, and 16 further coalitions of size
  2 to 5 drawn in complementary pairs in proportion to the Shapley-kernel weight of each size.
A removed region is replaced by a heavily blurred copy of the image. The same model outputs feed
  * Kernel SHAP (Lundberg and Lee, 2017): weighted least squares with the efficiency constraint, giving phi_r for the
    target class (an estimate of Equation 3.3), and
  * LIME (Ribeiro, Singh and Guestrin, 2016): ridge regression weighted by an exponential kernel on cosine distance.
The function predict(images) must return class probabilities (n, K).
"""
import hashlib
from math import comb

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

N_SAMPLES = 32
BLUR_SIGMA = 6.0
LIME_WIDTH = 0.25


def image_seed(image_id):
    return int(hashlib.md5(str(image_id).encode()).hexdigest()[:8], 16)


def coalitions(M, n=N_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    rows = [np.ones(M, int), np.zeros(M, int)]
    eye = np.eye(M, dtype=int)
    rows += list(eye) + list(1 - eye)
    sizes = np.arange(2, M - 1)                                   # 2..M-2
    w = np.array([(M - 1) / (s * (M - s)) for s in sizes])        # kernel mass of each size (summed over subsets)
    w = w / w.sum()
    seen = {tuple(r) for r in rows}
    tries = 0
    while len(rows) < n and tries < 10000:
        tries += 1
        s = rng.choice(sizes, p=w)
        z = np.zeros(M, int)
        z[rng.choice(M, s, replace=False)] = 1
        for c in (z, 1 - z):
            if tuple(c) not in seen and len(rows) < n:
                rows.append(c); seen.add(tuple(c))
    return np.stack(rows)


def blurred(img):
    a = np.asarray(img.convert("RGB"), np.float32)
    return np.stack([gaussian_filter(a[..., c], BLUR_SIGMA) for c in range(3)], -1)


def perturb(img, masks, Z):
    """masks: (M, H, W) bool at the image size; Z: (n, M). Returns n PIL images."""
    a = np.asarray(img.convert("RGB"), np.float32)
    b = blurred(img)
    out = []
    for z in Z:
        removed = np.zeros(a.shape[:2], bool)
        for k in np.where(z == 0)[0]:
            removed |= masks[k]
        x = np.where(removed[..., None], b, a)
        out.append(Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)))
    return out


def kernel_weight(M, s):
    if s == 0 or s == M:
        return 0.0
    return (M - 1) / (comb(M, s) * s * (M - s))


def kernel_shap(Z, y):
    """Z (n, M) with the all-ones and all-zeros rows present; y (n,). Returns phi (M,) with sum(phi) = f(1) - f(0)."""
    M = Z.shape[1]
    s = Z.sum(1)
    f1 = y[s == M][0]
    f0 = y[s == 0][0]
    keep = (s > 0) & (s < M)
    Zk, yk = Z[keep].astype(float), y[keep] - f0
    w = np.array([kernel_weight(M, int(v)) for v in s[keep]])
    # eliminate phi_M using the efficiency constraint
    X = Zk[:, :-1] - Zk[:, [-1]]
    t = yk - Zk[:, -1] * (f1 - f0)
    W = np.sqrt(w)[:, None]
    phi_head, *_ = np.linalg.lstsq(X * W, t * W[:, 0], rcond=None)
    return np.append(phi_head, (f1 - f0) - phi_head.sum())


def lime(Z, y, alpha=1.0):
    M = Z.shape[1]
    s = Z.sum(1)
    d = 1 - np.sqrt(s / M)                                       # cosine distance to the all-ones vector
    w = np.exp(-(d ** 2) / LIME_WIDTH ** 2)
    X = Z.astype(float)
    Xm = np.average(X, axis=0, weights=w)
    ym = np.average(y, weights=w)
    Xc, yc = X - Xm, y - ym
    A = (Xc * w[:, None]).T @ Xc + alpha * np.eye(M)
    coef = np.linalg.solve(A, (Xc * w[:, None]).T @ yc)
    return coef, ym - Xm @ coef


def explain_image(img, masks, predict, image_id, target=None, n=N_SAMPLES):
    """Returns dict with Z, P (n, K), target, phi, lime_coef, lime_intercept, spearman."""
    Z = coalitions(masks.shape[0], n, image_seed(image_id))
    P = np.asarray(predict(perturb(img, masks, Z)), float)
    if target is None:
        target = int(P[0].argmax())                               # row 0 is the unmasked image
    y = P[:, target]
    phi = kernel_shap(Z, y)
    coef, b = lime(Z, y)
    rho = spearmanr(phi, coef).correlation if np.std(phi) > 0 and np.std(coef) > 0 else np.nan
    return dict(Z=Z, P=P, target=target, phi=phi, lime_coef=coef, lime_intercept=b, spearman=rho)
