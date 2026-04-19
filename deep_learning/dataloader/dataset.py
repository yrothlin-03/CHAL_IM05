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


from .preprocess import CLAHETransform, extract_wbc_crop2, AddGaussianNoise


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# img_size = (224, 224)
img_size = (294, 294)

train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((300, 300)),
    transforms.Resize(img_size),

    # CLAHETransform(p=0.5), 

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
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.0,
    ),
    transforms.ToTensor(),

    # AddGaussianNoise(p=0.25, std_range=(0.003, 0.015)),

    # transforms.RandomErasing(
    #     p=0.25,
    #     scale=(0.02, 0.12),
    #     ratio=(0.3, 3.3),
    #     value="random",
    # ),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((300, 300)),
    transforms.Resize(img_size),
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
    # transforms.Resize((300, 300)),
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

tta_tfms = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((300, 300)),
    transforms.Resize(img_size),
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
        # img = extract_wbc_crop2(img, resize_mode="resize", output_size=img_size)
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