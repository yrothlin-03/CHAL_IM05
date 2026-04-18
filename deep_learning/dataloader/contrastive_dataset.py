import os
import sys
import cv2
import contextlib
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

label2id = {
    "SNE": 0,
    "LY": 1,
    "MO": 2,
    "EO": 3,
    "BA": 4,
    "VLY": 5,
    "BNE": 6,
    "MMY": 7,
    "MY": 8,
    "PMY": 9,
    "BL": 10,
    "PC": 11,
    "PLY": 12,
}


contrastive_train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(
        size=294,
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1),
    ),
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
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


contrastive_val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((294, 294)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


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


def split_files(
    files: List[str],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)
    split = int(len(files) * train_ratio)
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    return train_files, val_files


class IM05_ContrastiveDataset(Dataset):
    def __init__(self, files: List[str], labels: Dict[str, int], transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform if transform is not None else TwoCropsTransform(contrastive_train_tfms)

    def __len__(self):
        return len(self.files)

    def _load_image(self, img_path: str):
        img = imread_silent(img_path)
        if img is None:
            raise RuntimeError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx: int):
        img_path = self.files[idx]
        img = self._load_image(img_path)
        x1, x2 = self.transform(img)
        filename = os.path.basename(img_path)
        label = self.labels[filename]
        return x1.contiguous(), x2.contiguous(), torch.tensor(label, dtype=torch.long)


def get_contrastive_loaders(
    dataset_dir: str,
    label_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
    train_ratio: float = 0.9,
    seed: int = 42,
):
    files = get_filespath(dataset_dir)
    labels = get_labels(label_path)

    train_files, val_files = split_files(
        files,
        train_ratio=train_ratio,
        seed=seed,
    )

    train_dataset = IM05_ContrastiveDataset(
        files=train_files,
        labels=labels,
        transform=TwoCropsTransform(contrastive_train_tfms),
    )

    val_dataset = IM05_ContrastiveDataset(
        files=val_files,
        labels=labels,
        transform=TwoCropsTransform(contrastive_val_tfms),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    dataset_dir = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train"
    label_path = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train_metadata.csv"

    train_loader, val_loader = get_contrastive_loaders(
        dataset_dir=dataset_dir,
        label_path=label_path,
        batch_size=8,
        train_ratio=0.9,
        seed=42,
    )

    x1, x2, y = next(iter(train_loader))
    print("Train batch:", x1.shape, x2.shape, y.shape)

    x1, x2, y = next(iter(val_loader))
    print("Val batch:", x1.shape, x2.shape, y.shape)