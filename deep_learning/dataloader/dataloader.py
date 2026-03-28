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
from .dataset import (
    IM05_Dataset,
    train_tfms,
    val_tfms,
    tta_tfms,
)


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
    use_weighted_sampler: bool = False,
    num_classes: int = 13,
    sampler_power: float = 0.5,
    n_splits: int = 5,
    fold_index: int = 0,
    with_tta: bool = False,
):
    if test:
        dataset_dir = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/test"
        label_path = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/test_metadata.csv"
    else:
        dataset_dir = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train"
        label_path = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train_metadata.csv"

    files = get_filespath(dataset_dir)

    if not test:
        labels = get_labels(label_path) if (label_path and os.path.exists(label_path)) else None

        if n_splits > 1:
            print("[DATALOADER]: Using KFOLD")
            train_files, val_files = split_files_kfold(
                files,
                labels,
                n_splits=n_splits,
                fold_index=fold_index,
                seed=seed,
            )
        else:
            print("[DATALOADER]: Using single split")
            train_files, val_files = split_files_stratified(
                files, labels, train_ratio, seed
            )

        train_dataset = IM05_Dataset(
            train_files,
            labels=labels,
            evaluation=False,
            train=True,
            transform=train_tfms,
        )
        val_dataset = IM05_Dataset(
            val_files,
            labels=labels,
            evaluation=False,
            train=False,
            transform=val_tfms,
        )

        sampler = None
        if use_weighted_sampler:
            print("[DATALOADER]: Using weighted sampling")
            y_train = np.array(
                [labels[os.path.basename(f)] for f in train_files],
                dtype=np.int64,
            )
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

        if with_tta:
            print("[DATALOADER]: Using TTA")
            val_tta_dataset = IM05_Dataset(
                val_files,
                labels=labels,
                evaluation=False,
                train=False,
                transform=tta_tfms,
            )
            val_tta_loader = DataLoader(
                val_tta_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            return train_loader, val_loader, val_tta_loader, None, None

        return train_loader, val_loader, None, None, None

    test_dataset = IM05_Dataset(
        files,
        labels=None,
        evaluation=True,
        train=False,
        transform=val_tfms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if with_tta:
        print("[DATALOADER]: Using TTA")
        test_tta_dataset = IM05_Dataset(
            files,
            labels=None,
            evaluation=True,
            train=False,
            transform=tta_tfms,
        )
        test_tta_loader = DataLoader(
            test_tta_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return None, None, None, test_loader, test_tta_loader

    return None, None, None, test_loader, None




if __name__ == "__main__":
    train_loader, val_loader, val_tta_loader = get_loaders(batch_size=4, with_tta=True)

    x, y = next(iter(val_tta_loader))
    print("TTA batch:", x.shape, y.shape)