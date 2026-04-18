import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold

from .dataset import (
    IM05_Dataset,
    train_tfms,
    val_tfms,
    tta_tfms,
)

label2id = {
    "SNE": 0, "LY": 1, "MO": 2, "EO": 3, "BA": 4, "VLY": 5, "BNE": 6, "MMY": 7, "MY": 8, "PMY": 9, "BL": 10, "PC": 11, "PLY": 12,
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


def get_base_id(filename: str) -> str:
    p = Path(filename)
    stem = p.stem
    suffix = p.suffix
    marker = "_aug_"
    if marker in stem:
        stem = stem.split(marker)[0]
    return f"{stem}{suffix}"


def group_files_by_base_id(files: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for f in files:
        fname = os.path.basename(f)
        base_id = get_base_id(fname)
        groups.setdefault(base_id, []).append(f)
    return groups


def get_group_labels(
    groups: Dict[str, List[str]],
    labels: Dict[str, int],
) -> Dict[str, int]:
    group_labels = {}
    for base_id in groups:
        if base_id not in labels:
            raise ValueError(f"Base ID {base_id} not found in labels.")
        group_labels[base_id] = labels[base_id]
    return group_labels


def _labels_from_files(files: List[str], labels: Dict[str, int]) -> np.ndarray:
    y = []
    for f in files:
        fname = os.path.basename(f)
        base_id = get_base_id(fname)
        y.append(labels[base_id])
    return np.array(y, dtype=np.int64)


def _class_distribution(files: List[str], labels: Dict[str, int], num_classes: int = 13) -> np.ndarray:
    y = _labels_from_files(files, labels)
    return np.bincount(y, minlength=num_classes)


def _print_split_stats(
    train_files: List[str],
    val_files: List[str],
    labels: Dict[str, int],
    num_classes: int = 13,
) -> None:
    tr = _class_distribution(train_files, labels, num_classes)
    va = _class_distribution(val_files, labels, num_classes)
    print(f"[DATALOADER] Train distribution: {tr.tolist()}")
    print(f"[DATALOADER] Val distribution  : {va.tolist()}")


def split_grouped_stratified(
    files: List[str],
    labels: Dict[str, int],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    groups = group_files_by_base_id(files)
    group_labels = get_group_labels(groups, labels)

    base_ids = sorted(groups.keys())
    y = np.array([group_labels[bid] for bid in base_ids], dtype=np.int64)

    train_base_ids, val_base_ids = train_test_split(
        base_ids,
        train_size=train_ratio,
        random_state=seed,
        stratify=y,
    )

    train_files = []
    val_files = []

    for bid in train_base_ids:
        train_files.extend(groups[bid])
    for bid in val_base_ids:
        val_files.extend(groups[bid])

    return train_files, val_files


def split_grouped_kfold(
    files: List[str],
    labels: Dict[str, int],
    n_splits: int = 5,
    fold_index: int = 0,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    groups = group_files_by_base_id(files)
    group_labels = get_group_labels(groups, labels)

    base_ids = sorted(groups.keys())
    y = np.array([group_labels[bid] for bid in base_ids], dtype=np.int64)

    counts = np.bincount(y)
    positive_counts = counts[counts > 0]
    if len(positive_counts) == 0:
        raise ValueError("No labeled samples found.")

    min_count = int(positive_counts.min())
    safe_n_splits = max(2, min(n_splits, min_count))
    if safe_n_splits != n_splits:
        print(
            f"[DATALOADER]: Requested n_splits={n_splits} "
            f"but rarest grouped class is too small. Using n_splits={safe_n_splits} instead."
        )
        n_splits = safe_n_splits

    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f"fold_index={fold_index} is invalid for n_splits={n_splits}")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits = list(skf.split(base_ids, y))
    train_idx, val_idx = splits[fold_index]

    train_base_ids = [base_ids[i] for i in train_idx]
    val_base_ids = [base_ids[i] for i in val_idx]

    train_files = []
    val_files = []

    for bid in train_base_ids:
        train_files.extend(groups[bid])
    for bid in val_base_ids:
        val_files.extend(groups[bid])

    return train_files, val_files


def undersample_files(
    files: List[str],
    labels: Dict[str, int],
    num_classes: int = 13,
    mode: str = "median",
    target_count: int | None = None,
    seed: int = 42,
) -> List[str]:
    rng = np.random.default_rng(seed)

    per_class = {c: [] for c in range(num_classes)}
    for f in files:
        y = labels[get_base_id(os.path.basename(f))]
        per_class[y].append(f)

    counts = np.array([len(per_class[c]) for c in range(num_classes) if len(per_class[c]) > 0])

    if target_count is None:
        if mode == "min":
            target_count = int(counts.min())
        elif mode == "median":
            target_count = int(np.median(counts))
        elif mode == "mean":
            target_count = int(np.mean(counts))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    kept = []
    for c in range(num_classes):
        cls_files = per_class[c]
        if len(cls_files) <= target_count:
            kept.extend(cls_files)
        else:
            idx = rng.choice(len(cls_files), size=target_count, replace=False)
            kept.extend([cls_files[i] for i in idx])

    rng.shuffle(kept)
    return kept


def get_loaders(
    test: bool = False,
    train_ratio: float = 0.9,
    seed: int = 42,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    use_undersampling: bool = False,
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
        # dataset_dir = "/home/infres/yrothlin-24/CHAL_IM05/IMA205-challenge_resampled/train"
        # label_path = "/home/infres/yrothlin-24/CHAL_IM05/IMA205-challenge_resampled/train_metadata.csv"
        dataset_dir = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train"
        label_path = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train_metadata.csv"

    files = get_filespath(dataset_dir)

    if not test:
        labels = get_labels(label_path) if (label_path and os.path.exists(label_path)) else None
        if labels is None:
            raise ValueError("Training labels could not be loaded.")

        if n_splits > 1:
            print("[DATALOADER]: Using grouped KFOLD")
            train_files, val_files = split_grouped_kfold(
                files,
                labels,
                n_splits=n_splits,
                fold_index=fold_index,
                seed=seed,
            )
        else:
            print("[DATALOADER]: Using grouped single split")
            train_files, val_files = split_grouped_stratified(
                files,
                labels,
                train_ratio=train_ratio,
                seed=seed,
            )

        _print_split_stats(train_files, val_files, labels, num_classes=num_classes)

        if use_undersampling and not use_weighted_sampler:
            print("[DATALOADER]: Using undersampling")
            train_files = undersample_files(
                train_files,
                labels,
                num_classes=num_classes,
                mode="median",
                target_count=None,
                seed=seed,
            )
            print("[DATALOADER]: Train distribution after undersampling:")
            print(_class_distribution(train_files, labels, num_classes).tolist())

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
        if use_weighted_sampler and not use_undersampling:
            print("[DATALOADER]: Using weighted sampling")
            y_train = _labels_from_files(train_files, labels)
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            class_w = 1.0 / np.power(counts + 1e-6, sampler_power)
            class_w = class_w / class_w.mean()
            # class_w = np.clip(class_w, None, 3.0)
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