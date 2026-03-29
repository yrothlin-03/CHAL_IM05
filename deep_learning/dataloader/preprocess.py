import cv2
import numpy as np
from PIL import Image


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), p=0.5):
        self.p = p
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        img_np = np.array(img)

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(out)


def extract_wbc_crop2(
    img_rgb: np.ndarray,
    pad: int = 20,
    min_area_frac: float = 0.0005,
    q: float = 0.92,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    hdist = np.minimum(np.abs(H - 140.0), 180.0 - np.abs(H - 140.0)) / 90.0
    hscore = 1.0 - np.clip(hdist, 0.0, 1.0)
    sscore = np.clip((S - 30.0) / 225.0, 0.0, 1.0)
    vscore = 1.0 - np.clip(V / 255.0, 0.0, 1.0)

    score = 0.55 * hscore + 0.25 * sscore + 0.20 * vscore
    thr = float(np.quantile(score, q))
    nuc = (score >= thr).astype(np.uint8)

    nuc = cv2.medianBlur(nuc * 255, 5)
    nuc = (nuc > 0).astype(np.uint8)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_OPEN, k1, iterations=1)
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_CLOSE, k1, iterations=2)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(nuc.astype(np.uint8), connectivity=8)
    if num <= 1:
        return img_rgb

    h, w = nuc.shape
    min_area = int(min_area_frac * h * w)

    cx0, cy0 = w * 0.5, h * 0.5
    best_i = None
    best_score = -1e18
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx = float(stats[i, cv2.CC_STAT_LEFT] + 0.5 * stats[i, cv2.CC_STAT_WIDTH])
        cy = float(stats[i, cv2.CC_STAT_TOP] + 0.5 * stats[i, cv2.CC_STAT_HEIGHT])
        d2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
        s = np.log1p(area) - 0.0008 * d2
        if s > best_score:
            best_score = s
            best_i = i

    if best_i is None:
        return img_rgb

    nuc = (lab == best_i).astype(np.uint8)

    kbig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    mask = cv2.dilate(nuc, kbig, iterations=1)

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return img_rgb

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_rgb.shape[1], x2 + pad)
    y2 = min(img_rgb.shape[0], y2 + pad)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return img_rgb

    return img_rgb[y1:y2, x1:x2]
