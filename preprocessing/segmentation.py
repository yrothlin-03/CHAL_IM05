import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)


def scharr_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    return mag.astype(np.float32)


def median_filter_channel(ch: np.ndarray, k: int = 7) -> np.ndarray:
    return cv2.medianBlur(ch, k)


def coarse_kmeans_mask(img_rgb: np.ndarray, k: int = 8, sample_pixels: int = 50000, seed: int = 42) -> np.ndarray:
    h, w, _ = img_rgb.shape
    hsv = to_hsv(img_rgb)

    feat = np.concatenate([img_rgb.astype(np.float32), hsv.astype(np.float32)], axis=2)
    feat = feat.reshape(-1, feat.shape[2])

    rng = np.random.default_rng(seed)
    idx = rng.choice(feat.shape[0], size=min(sample_pixels, feat.shape[0]), replace=False)
    feat_s = feat[idx]

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
    km.fit(feat_s)

    labels = km.predict(feat).reshape(h, w)

    border = np.zeros((h, w), np.uint8)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1

    border_labels = labels[border.astype(bool)]
    uniq, counts = np.unique(border_labels, return_counts=True)
    bg_label = uniq[np.argmax(counts)]

    mask = (labels != bg_label).astype(np.uint8)

    mask = cv2.medianBlur(mask * 255, 7)
    mask = (mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + np.argmax(areas)
    return (lab == best).astype(np.uint8)


def build_pixel_features(img_rgb: np.ndarray, med_ksize: int = 7) -> np.ndarray:
    hsv = to_hsv(img_rgb)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    v = hsv[:, :, 2]
    v_med = median_filter_channel(v, med_ksize).astype(np.float32) / 255.0

    edge = scharr_magnitude(gray)

    rgb_f = img_rgb.astype(np.float32) / 255.0
    hsv_f = hsv.astype(np.float32)
    hsv_f[:, :, 0] = hsv_f[:, :, 0] / 179.0
    hsv_f[:, :, 1] = hsv_f[:, :, 1] / 255.0
    hsv_f[:, :, 2] = hsv_f[:, :, 2] / 255.0

    feats = np.concatenate(
        [
            rgb_f,
            hsv_f,
            v_med[..., None],
            edge[..., None],
        ],
        axis=2
    )
    return feats.reshape(-1, feats.shape[2]).astype(np.float32)


def sample_training_pixels(features: np.ndarray, mask_coarse: np.ndarray, n_pos: int = 20000, n_neg: int = 20000, seed: int = 42):
    y = mask_coarse.reshape(-1).astype(np.uint8)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng = np.random.default_rng(seed)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Coarse mask degenerate (no pos or no neg).")

    pos_s = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    neg_s = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)

    idx = np.concatenate([pos_s, neg_s])
    rng.shuffle(idx)

    X = features[idx]
    y = y[idx]
    return X, y


def train_svm_pixel_classifier(X: np.ndarray, y: np.ndarray, C: float = 10.0, gamma: str = "scale"):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    svm = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")
    svm.fit(Xs, y)
    return scaler, svm


def predict_mask(img_rgb: np.ndarray, scaler: StandardScaler, svm: SVC, batch: int = 200000) -> np.ndarray:
    h, w, _ = img_rgb.shape
    feats = build_pixel_features(img_rgb)
    out = np.zeros((feats.shape[0],), dtype=np.uint8)

    for i in range(0, feats.shape[0], batch):
        Xb = feats[i:i+batch]
        Xb = scaler.transform(Xb)
        pb = svm.predict(Xb).astype(np.uint8)
        out[i:i+batch] = pb

    return out.reshape(h, w).astype(np.uint8)


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = largest_component(mask)
    return mask.astype(np.uint8)


def segment_wbc(img_rgb: np.ndarray, k: int = 8, seed: int = 42) -> np.ndarray:
    mask_coarse = coarse_kmeans_mask(img_rgb, k=k, seed=seed)
    mask_coarse = largest_component(mask_coarse)

    feats = build_pixel_features(img_rgb)
    X, y = sample_training_pixels(feats, mask_coarse, n_pos=20000, n_neg=20000, seed=seed)

    scaler, svm = train_svm_pixel_classifier(X, y, C=10.0, gamma="scale")

    mask_refined = predict_mask(img_rgb, scaler, svm)
    mask_refined = postprocess_mask(mask_refined)

    return mask_refined


if __name__ == "__main__":
    path = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/train/train_28899.png"
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mask = segment_wbc(rgb, k=8, seed=42)

    vis = (mask * 255).astype(np.uint8)
    cv2.imwrite("/home/infres/yrothlin-24/CHAL_IM05/mask.png", vis)