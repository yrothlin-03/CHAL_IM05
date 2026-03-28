from pathlib import Path
from typing import Dict, Any
from yaml import safe_load
import torch

from .utils import ContrastiveTrainer
from .models import ContrastiveModel
from .dataloader import get_contrastive_loaders


def get_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Config file {config_path} is empty or invalid YAML.")
    return config


def main(config: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    backbone_name = model_config.get("backbone_name", "UNKNOWN")

    print("-----" * 20)
    print("-----" * 20)
    print("-----" * 20)
    print(f"CONTRASTIVE PRETRAINING WITH BACKBONE: {backbone_name}")
    print("-----" * 20)
    print("-----" * 20)
    print("-----" * 20)

    current_model_config = dict(model_config)

    model = ContrastiveModel(**current_model_config).to(device)

    train_loader, val_loader = get_contrastive_loaders(**data_config)

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    print(
        f"Successfully loaded contrastive data with sizes: "
        f"train={len(train_loader.dataset)}"
        + (f", val={len(val_loader.dataset)}" if val_loader is not None else "")
    )

    trainer = ContrastiveTrainer(
        model=model,
        dataloaders=dataloaders,
        config=training_config,
    )

    trainer.train()


if __name__ == "__main__":
    config_path = Path("/home/infres/yrothlin-24/CHAL_IM05/configs/contrastive.yaml")
    config = get_config(config_path)
    main(config)