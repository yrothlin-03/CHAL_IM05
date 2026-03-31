import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

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


def _group_distribution(base_ids: List[str], group_labels: Dict[str, int], num_classes: int = 13) -> np.ndarray:
    y = np.array([group_labels[bid] for bid in base_ids], dtype=np.int64)
    return np.bincount(y, minlength=num_classes)


def _print_split_stats(
    train_files: List[str],
    val_files: List[str],
    labels: Dict[str, int],
    num_classes: int = 13,
) -> None:
    tr = _class_distribution(train_files, labels, num_classes)
    va = _class_distribution(val_files, labels, num_classes)
    print(f"[DATALOADER] Train distribution (files): {tr.tolist()}")
    print(f"[DATALOADER] Val distribution   (files): {va.tolist()}")


def _print_group_split_stats(
    train_base_ids: List[str],
    val_base_ids: List[str],
    group_labels: Dict[str, int],
    num_classes: int = 13,
) -> None:
    tr = _group_distribution(train_base_ids, group_labels, num_classes)
    va = _group_distribution(val_base_ids, group_labels, num_classes)
    print(f"[DATALOADER] Train distribution (groups): {tr.tolist()}")
    print(f"[DATALOADER] Val distribution   (groups): {va.tolist()}")


def split_grouped_stratified_safe(
    files: List[str],
    labels: Dict[str, int],
    val_ratio: float = 0.2,
    seed: int = 42,
    num_classes: int = 13,
) -> Tuple[List[str], List[str]]:
    """
    Split grouped by base_id, stratified per class at group level.

    Rules per class:
    - 1 group  -> all in train
    - 2 groups -> 1 train / 1 val
    - >=3      -> roughly val_ratio in val, but:
                 * at least 1 val
                 * at least 1 train
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    rng = np.random.default_rng(seed)

    groups = group_files_by_base_id(files)
    group_labels = get_group_labels(groups, labels)

    # base ids per class
    class_to_base_ids: Dict[int, List[str]] = {c: [] for c in range(num_classes)}
    for base_id, y in group_labels.items():
        class_to_base_ids[y].append(base_id)

    train_base_ids: List[str] = []
    val_base_ids: List[str] = []

    print("[DATALOADER] Safe grouped stratified split")
    for cls in range(num_classes):
        cls_base_ids = sorted(class_to_base_ids[cls])
        n = len(cls_base_ids)

        if n == 0:
            continue

        cls_base_ids = list(rng.permutation(cls_base_ids))

        if n == 1:
            n_val = 0
        elif n == 2:
            n_val = 1
        else:
            n_val = int(round(n * val_ratio))
            n_val = max(1, n_val)
            n_val = min(n_val, n - 1)

        cls_val = cls_base_ids[:n_val]
        cls_train = cls_base_ids[n_val:]

        train_base_ids.extend(cls_train)
        val_base_ids.extend(cls_val)

        print(
            f"  class={cls:2d} | total_groups={n:3d} | "
            f"train_groups={len(cls_train):3d} | val_groups={len(cls_val):3d}"
        )

    train_files: List[str] = []
    val_files: List[str] = []

    for bid in train_base_ids:
        train_files.extend(groups[bid])
    for bid in val_base_ids:
        val_files.extend(groups[bid])

    _print_group_split_stats(train_base_ids, val_base_ids, group_labels, num_classes=num_classes)

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
    n_splits: int = 1,          # <- on n'utilise plus le KFold par défaut
    fold_index: int = 0,        # conservé pour compatibilité API
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
        if labels is None:
            raise ValueError("Training labels could not be loaded.")

        if n_splits > 1:
            print(
                "[DATALOADER] Warning: n_splits > 1 requested, "
                "but for ultra-rare classes a single safe grouped split is recommended. "
                "Using safe grouped split anyway."
            )

        train_files, val_files = split_grouped_stratified_safe(
            files=files,
            labels=labels,
            val_ratio=1.0 - train_ratio,
            seed=seed,
            num_classes=num_classes,
        )

        _print_split_stats(train_files, val_files, labels, num_classes=num_classes)

        if use_undersampling and not use_weighted_sampler:
            print("[DATALOADER] Using undersampling")
            train_files = undersample_files(
                train_files,
                labels,
                num_classes=num_classes,
                mode="median",
                target_count=None,
                seed=seed,
            )
            print("[DATALOADER] Train distribution after undersampling:")
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
            print("[DATALOADER] Using weighted sampling")
            y_train = _labels_from_files(train_files, labels)
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            class_w = 1.0 / np.power(counts + 1e-6, sampler_power)
            class_w = class_w / class_w.mean()
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
            print("[DATALOADER] Using TTA")
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
        print("[DATALOADER] Using TTA")
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