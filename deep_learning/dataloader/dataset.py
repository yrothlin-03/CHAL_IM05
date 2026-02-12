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

# train_tfms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((338, 338)),
#     transforms.Lambda(lambda img: F.equalize(img)),
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.RandomVerticalFlip(0.5),
#     transforms.RandomRotation(180, fill=128),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])

# val_tfms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((338, 338)),
#     transforms.Lambda(lambda img: F.equalize(img)),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])



train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(180, fill=128),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
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
    ):
        self.files = files
        self.labels = labels
        self.evaluation = evaluation
        self.transform = train_tfms if train else val_tfms
        if not self.evaluation and self.labels is None:
            raise ValueError

    def __len__(self):
        return len(self.files)

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img

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

def split_files_stratified(
    files: List[str],
    labels: Dict[str, int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    y = [labels[os.path.basename(f)] for f in files]
    train_files, val_files = train_test_split(
        files,
        train_size=train_ratio,
        random_state=seed,
        stratify=y
    )
    return list(train_files), list(val_files)



def split_files_kfold(
    files: List[str],
    labels: Dict[str, int],
    n_splits: int = 5,
    fold_index: int = 0,
    seed: int = 42,
):
    y = np.array([labels[os.path.basename(f)] for f in files])

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    splits = list(skf.split(files, y))
    train_idx, val_idx = splits[fold_index]

    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]

    return train_files, val_files



def get_loaders(
    test: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
    num_classes: int = 13,
    sampler_power: float = 0.5,
    n_splits: int = 5,
    fold_index: int = 0,
):

    if test:
        dataset_dir = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/test"
        label_path = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/test_metadata.csv"
    else:
        dataset_dir = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/train"
        label_path = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/train_metadata.csv"

    files = get_filespath(dataset_dir)
    if not test:
        labels = get_labels(label_path) if (label_path and os.path.exists(label_path)) else None

    if not test:
        train_files, val_files = split_files_stratified(files, labels, train_ratio, seed)
        # train_files, val_files = split_files_kfold(
        #     files,
        #     labels,
        #     n_splits=n_splits,
        #     fold_index=fold_index,
        #     seed=seed
        # )
        train_dataset = IM05_Dataset(train_files, labels=labels, evaluation=False, train=True)
        val_dataset = IM05_Dataset(val_files, labels=labels, evaluation=False, train=False)

        sampler = None
        if use_weighted_sampler:
            y_train = np.array([labels[os.path.basename(f)] for f in train_files], dtype=np.int64)
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            class_w = 1.0 / np.power(counts + 1e-6, sampler_power)
            sample_w = class_w[y_train]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_w, dtype=torch.double),
                num_samples=len(sample_w),
                replacement=True,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    test_dataset = IM05_Dataset(files, labels=None, evaluation=True, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader

if __name__ == "__main__":

    train_loader, val_loader = get_loaders(
        test=False,
        train_ratio=0.8,
        seed=42,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        use_weighted_sampler=True,
        num_classes=13,
        sampler_power=0.5,
    )


    test_loader = get_loaders(
        test=True,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
    )