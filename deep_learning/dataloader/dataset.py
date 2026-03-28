import os
import sys
import contextlib
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10),
        scale=(0.85, 1.15),
        shear=8,
    ),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.0,
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.12),
        ratio=(0.3, 3.3),
        value="random",
    ),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

tta_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

@contextlib.contextmanager
def _filter_stderr(substr: str):
    r_fd, w_fd = os.pipe()
    orig_fd = os.dup(2)
    os.dup2(w_fd, 2)
    os.close(w_fd)
    try:
        yield
    finally:
        os.dup2(orig_fd, 2)
        os.close(orig_fd)
        with os.fdopen(r_fd, "r") as r:
            for line in r:
                if substr not in line:
                    sys.stderr.write(line)

def imread_silent(path: str):
    with _filter_stderr("Corrupt JPEG data"):
        return cv2.imread(path, cv2.IMREAD_COLOR)

label2id = {
    "SNE": 0, "LY": 1, "MO": 2, "EO": 3, "BA": 4, "VLY": 5, "BNE": 6,
    "MMY": 7, "MY": 8, "PMY": 9, "BL": 10, "PC": 11, "PLY": 12,
}

def get_filespath(dataset_dir: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith(".png"):
                files.append(os.path.join(dirpath, filename))
    files.sort()
    return files

def get_labels(csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        labels[str(row["ID"])] = label2id[str(row["label"])]
    return labels


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


class IM05_Dataset(Dataset):
    def __init__(
        self,
        files: List[str],
        labels: Optional[Dict[str, int]] = None,
        evaluation: bool = False,
        train: bool = True,
        transform=None,
    ):
        self.files = files
        self.labels = labels
        self.evaluation = evaluation
        self.transform = transform if transform is not None else (train_tfms if train else val_tfms)
        if not self.evaluation and self.labels is None:
            raise ValueError

    def __len__(self):
        return len(self.files)

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = extract_wbc_crop2(img)
        img = self.transform(img)
        return img.contiguous()

    def __getitem__(self, idx: int):
        img_path = self.files[idx]
        img = imread_silent(img_path)
        if img is None:
            raise RuntimeError(img_path)
        img = self._preprocess_image(img)
        filename = os.path.basename(img_path)
        if self.evaluation:
            return img, filename
        label = self.labels[filename]
        return img, label