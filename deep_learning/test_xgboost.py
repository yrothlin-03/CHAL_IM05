from pathlib import Path
from copy import deepcopy
from typing import Dict, Any

import numpy as np
import torch
from yaml import safe_load
from xgboost import XGBClassifier

from .models import Model
from .utils.metrics import Metrics
from .dataloader import get_loaders
from .dataloader.dataset import IM05_Dataset, val_tfms


NUM_CLASSES = 13


def get_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} is empty or invalid YAML.")
    return config


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    msd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(msd, strict=True)
    model.eval()
    return model


def build_eval_loader_from_dataset(dataset, batch_size: int, num_workers: int, pin_memory: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_feature_loaders_from_existing_split(data_config: Dict[str, Any]):
    data_config = data_config.copy()
    data_config["with_tta"] = False

    train_loader, val_loader, _, _, _ = get_loaders(
        **data_config,
        test=False,
    )

    train_files = train_loader.dataset.files
    val_files = val_loader.dataset.files
    labels = train_loader.dataset.labels

    train_dataset_eval = IM05_Dataset(
        files=train_files,
        labels=labels,
        evaluation=False,
        train=False,
        transform=val_tfms,
    )

    val_dataset_eval = IM05_Dataset(
        files=val_files,
        labels=labels,
        evaluation=False,
        train=False,
        transform=val_tfms,
    )

    batch_size = int(data_config.get("batch_size", 64))
    num_workers = int(data_config.get("num_workers", 4))
    pin_memory = bool(data_config.get("pin_memory", True))

    train_eval_loader = build_eval_loader_from_dataset(
        train_dataset_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_eval_loader = build_eval_loader_from_dataset(
        val_dataset_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_eval_loader, val_eval_loader


@torch.no_grad()
def extract_features_and_labels(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = False,
):
    model.eval()

    feats = []
    labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            x = model.backbone(inputs)

        # Important: XGBoost attend un tableau 2D [N, D]
        # Si le backbone retourne [B, C, H, W], on fait un flatten
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)

        feats.append(x.detach().cpu().numpy())
        labels.append(targets.numpy())

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    return X, y


def build_xgb_classifier(
    random_state: int,
    num_classes: int = NUM_CLASSES,
    use_gpu: bool = True,
):
    kwargs = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=random_state,
        tree_method="hist",
    )

    if use_gpu:
        kwargs["device"] = "cuda"

    return XGBClassifier(**kwargs)


def fit_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    use_gpu: bool = True,
):
    clf = build_xgb_classifier(
        random_state=random_state,
        num_classes=NUM_CLASSES,
        use_gpu=use_gpu,
    )
    clf.fit(X_train, y_train)
    return clf


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = Metrics()
    metrics.update(y_true, y_pred)
    return metrics.compute()


def format_metrics(m: dict, top_k: int = 5) -> str:
    bacc = float(m.get("bacc", 0.0))
    macro_f1 = float(m.get("macro_f1", 0.0))
    prec = np.asarray(m.get("precision", []), dtype=float)
    rec = np.asarray(m.get("recall", []), dtype=float)
    f1 = np.asarray(m.get("f1", []), dtype=float)

    s = f"bacc={bacc:.3f} macro_f1={macro_f1:.3f}"
    if f1.size:
        worst = np.argsort(f1)[:top_k].tolist()
        best = np.argsort(-f1)[:top_k].tolist()

        def pack(idxs):
            return " ".join([f"{i}(p{prec[i]:.2f} r{rec[i]:.2f} f{f1[i]:.2f})" for i in idxs])

        s += "\n  best : " + pack(best)
        s += "\n  worst: " + pack(worst)

    return s


def print_distribution(name: str, y: np.ndarray, num_classes: int = NUM_CLASSES):
    counts = np.bincount(y, minlength=num_classes)
    print(f"{name} distribution: {counts.tolist()}")


def main_xgb_on_features(config: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = deepcopy(config["data"])
    model_config = deepcopy(config["model"])
    training_config = deepcopy(config["training"])

    ckpt_path = training_config["resume_from"]
    use_amp = bool(training_config.get("use_amp", False))
    random_state = int(data_config.get("seed", 42))
    use_gpu_xgb = bool(torch.cuda.is_available())

    print("-" * 80)
    print("MODE: FEATURE EXTRACTION + XGBOOST")
    print(f"DEVICE: {device}")
    print(f"CHECKPOINT: {ckpt_path}")
    print("-" * 80)

    model = Model(**model_config).to(device)
    model = load_model_checkpoint(model, ckpt_path, device)

    train_eval_loader, val_eval_loader = build_feature_loaders_from_existing_split(data_config)

    print(f"Train eval size: {len(train_eval_loader.dataset)}")
    print(f"Val eval size  : {len(val_eval_loader.dataset)}")

    X_train, y_train = extract_features_and_labels(
        model=model,
        loader=train_eval_loader,
        device=device,
        use_amp=use_amp,
    )

    X_val, y_val = extract_features_and_labels(
        model=model,
        loader=val_eval_loader,
        device=device,
        use_amp=use_amp,
    )

    print(f"Train features shape: {X_train.shape}")
    print(f"Val features shape  : {X_val.shape}")

    print_distribution("Train", y_train)
    print_distribution("Val", y_val)

    clf = fit_xgb_classifier(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        use_gpu=use_gpu_xgb,
    )

    train_pred = clf.predict(X_train).astype(np.int64)
    val_pred = clf.predict(X_val).astype(np.int64)

    train_metrics = compute_metrics(y_train, train_pred)
    val_metrics = compute_metrics(y_val, val_pred)

    print("\n[TRAIN METRICS - XGBoost on train features]")
    print(format_metrics(train_metrics))

    print("\n[VAL METRICS - XGBoost on val features]")
    print(format_metrics(val_metrics))


if __name__ == "__main__":
    config_path = Path("/home/infres/yrothlin-24/CHAL_IM05/configs/training.yaml")
    config = get_config(config_path)
    main_xgb_on_features(config)