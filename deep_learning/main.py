from pathlib import Path
from copy import deepcopy
from typing import Any, Dict

import pandas as pd
import torch
from yaml import safe_load

from .utils import Trainer
from .models import Model
from .dataloader import get_loaders


ID2LABEL = {
    0: "SNE",
    1: "LY",
    2: "MO",
    3: "EO",
    4: "BA",
    5: "VLY",
    6: "BNE",
    7: "MMY",
    8: "MY",
    9: "PMY",
    10: "BL",
    11: "PC",
    12: "PLY",
}


def get_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} is empty or invalid YAML.")
    return config


def create_submission_file(preds: Dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"ID": filename, "label": ID2LABEL[int(pred_id)]}
        for filename, pred_id in sorted(preds.items(), key=lambda x: x[0])
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def get_val_distribution(val_loader) -> Dict[int, int]:
    distribution = {i: 0 for i in range(13)}
    for _, labels in val_loader:
        for label in labels:
            distribution[int(label)] += 1
    return distribution


def print_dataset_sizes(
    train_loader=None,
    val_loader=None,
    test_loader=None,
) -> None:
    train_size = len(train_loader.dataset) if train_loader is not None else "N/A"
    val_size = len(val_loader.dataset) if val_loader is not None else "N/A"
    test_size = len(test_loader.dataset) if test_loader is not None else "N/A"

    print(
        f"Successfully loaded data with sizes: "
        f"train={train_size}, val={val_size}, test={test_size}"
    )


def build_model(model_config: Dict[str, Any], device: torch.device):
    model = Model(**model_config).to(device)
    return model


def run_single_train(
    device: torch.device,
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> None:
    model = build_model(model_config, device)

    train_loader, val_loader, val_tta_loader, test_loader, test_tta_loader = get_loaders(
        **data_config,
        test=False,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "val_tta": val_tta_loader,
        "test": test_loader,
        "test_tta": test_tta_loader,
    }

    print_dataset_sizes(train_loader, val_loader, test_loader)
    print(f"Validation set distribution: {get_val_distribution(val_loader)}")

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        config=training_config,
        evaluation=False,
    )
    trainer.train()


def run_kfold_train(
    device: torch.device,
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> None:
    n_splits = int(data_config.get("n_splits", 5))

    print(f"Training with {n_splits}-fold cross-validation...")

    for fold in range(n_splits):
        print("=" * 80)
        print(f"FOLD {fold + 1}/{n_splits}")
        print("=" * 80)

        fold_model_config = deepcopy(model_config)
        fold_training_config = deepcopy(training_config)
        fold_training_config["ckpt_dir"] = str(
            Path(training_config.get("ckpt_dir", "checkpoints")) / f"fold_{fold}"
        )

        model = build_model(fold_model_config, device)

        train_loader, val_loader, val_tta_loader, test_loader, test_tta_loader = get_loaders(
            **data_config,
            test=False,
            fold_index=fold,
        )

        dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "val_tta": val_tta_loader,
            "test": test_loader,
            "test_tta": test_tta_loader,
        }

        print_dataset_sizes(train_loader, val_loader, test_loader)
        print(f"Validation set distribution: {get_val_distribution(val_loader)}")
        print(f"Checkpoint directory: {fold_training_config['ckpt_dir']}")

        trainer = Trainer(
            model=model,
            dataloaders=dataloaders,
            config=fold_training_config,
            evaluation=False,
        )
        trainer.train()


def run_single_eval(
    device: torch.device,
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> Dict[str, int]:
    model = build_model(model_config, device)

    train_loader, val_loader, val_tta_loader, test_loader, test_tta_loader = get_loaders(
        **data_config,
        test=True,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "val_tta": val_tta_loader,
        "test": test_loader,
        "test_tta": test_tta_loader,
    }

    print_dataset_sizes(test_loader=test_loader)

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        config=training_config,
        evaluation=True,
    )
    preds = trainer.evaluate()
    return preds


def run_kfold_eval(
    device: torch.device,
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> Dict[str, int]:
    n_splits = int(data_config.get("n_splits", 5))
    logits_sum: Dict[str, torch.Tensor] = {}

    train_loader, val_loader, val_tta_loader, test_loader, test_tta_loader = get_loaders(
        **data_config,
        test=True,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "val_tta": val_tta_loader,
        "test": test_loader,
        "test_tta": test_tta_loader,
    }

    print_dataset_sizes(test_loader=test_loader)

    for fold in range(n_splits):
        print("=" * 80)
        print(f"EVAL FOLD {fold + 1}/{n_splits}")
        print("=" * 80)

        fold_model_config = deepcopy(model_config)
        fold_training_config = deepcopy(training_config)
        fold_training_config["resume_from"] = str(
            Path(training_config.get("ckpt_dir", "checkpoints")) / f"fold_{fold}" / "best.pt"
        )

        model = build_model(fold_model_config, device)

        trainer = Trainer(
            model=model,
            dataloaders=dataloaders,
            config=fold_training_config,
            evaluation=True,
        )

        preds_logits = trainer.evaluate(return_logits=True)

        for filename, logits in preds_logits.items():
            logits_t = torch.as_tensor(logits).detach().cpu()
            if filename in logits_sum:
                logits_sum[filename] += logits_t
            else:
                logits_sum[filename] = logits_t.clone()

    preds = {
        filename: int(torch.argmax(logits / n_splits).item())
        for filename, logits in logits_sum.items()
    }
    return preds


def main(config: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = deepcopy(config["data"])
    model_config = deepcopy(config["model"])
    training_config = deepcopy(config["training"])

    evaluation = bool(config.get("evaluation", False))
    n_splits = int(data_config.get("n_splits", 1))

    print("-" * 80)
    print(f"BACKBONE: {model_config.get('backbone_name', 'N/A')}")
    print(f"MODE: {'evaluation' if evaluation else 'training'}")
    print(f"DEVICE: {device}")
    print("-" * 80)

    if evaluation:
        if n_splits > 1:
            preds = run_kfold_eval(device, model_config, data_config, training_config)
        else:
            preds = run_single_eval(device, model_config, data_config, training_config)

        output_path = Path(config.get("submission_path", "submissions/submission.csv"))
        create_submission_file(preds, output_path)
        print(f"Submission saved to: {output_path}")

        for i, (k, v) in enumerate(sorted(preds.items())):
            if i >= 20:
                break
            print(k, v)

    else:
        if n_splits > 1:
            run_kfold_train(device, model_config, data_config, training_config)
        else:
            run_single_train(device, model_config, data_config, training_config)


if __name__ == "__main__":
    config_path = Path("/home/infres/yrothlin-24/CHAL_IM05/configs/training.yaml")
    config = get_config(config_path)
    main(config)