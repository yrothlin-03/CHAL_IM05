from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional
from yaml import safe_load
import torch

from .utils import Trainer
from .models import Model
from .dataloader import get_loaders

from typing import Dict, Any
from copy import deepcopy


id2label = {
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


def get_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Config file {config_path} is empty or invalid YAML.")
    return config



def create_submission_file(preds: Dict[str, Any], output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"ID": filename, "label": id2label[int(pred_id)]}
        for filename, pred_id in sorted(preds.items(), key=lambda x: x[0])
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)



def get_val_distribution(val_loader):
    distribution = {i: 0 for i in range(13)}
    for _, labels in val_loader:
        for label in labels:
            distribution[int(label)] += 1
    return distribution


BACKBONE_NAME = [
    'RESNET_50',
    # 'RESNET_18',
    # 'RESNET_101',
    # 'EFFICIENTNET_B4',
    # 'EFFICIENTNET_B7',
    # 'EFFICIENTNET_V2_S',
    # 'EFFICIENTNET_V2_M',
    # 'REGNET',
    # 'VIT_B_16',
    # 'CONVNEXT_BASE',
    # 'CONVNEXT_SMALL',
    # 'DENSENET_169',
    # 'SWIN_T',
]


def main(config: Dict[str, Any]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    for backbone_name in BACKBONE_NAME:
        print("-----"*20)
        print("-----"*20)
        print("-----"*20)
        print(f"TEST WITH BACKBONE : {backbone_name}")
        print("-----"*20)
        print("-----"*20)
        print("-----"*20)


        model_config["backbone_name"]=backbone_name
        model = Model(**model_config).to(device)

        train_loader, val_loader, val_tta_loader, test_loader, test_tta_loader = get_loaders(**data_config, test=False)
        dataloaders = {"train": train_loader, "val": val_loader, "val_tta": val_tta_loader, "test": test_loader, "test_tta": test_tta_loader}

        print(
            f"Succesfully loaded data with sizes : "
            f"train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}"
        )

        val_distri = get_val_distribution(val_loader)
        print(f"Validation set distribution: {val_distri}")

        trainer = Trainer(
            model=model,
            dataloaders=dataloaders,
            config=training_config,
            evaluation=False,
        )

        trainer.train()





def main_bis(config: Dict[str, Any]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = deepcopy(config["data"])
    model_config = deepcopy(config["model"])
    training_config = deepcopy(config["training"])

    evaluation = bool(config.get("evaluation", False))

    print("-----" * 20)
    print(f"MODE : {'EVALUATION' if evaluation else 'TRAINING'}")
    print(f"DEVICE : {device}")
    print("-----" * 20)

    
    if not evaluation:

        model = Model(**model_config).to(device)

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

        print(
            f"Successfully loaded data with sizes : "
            f"train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}"
        )

        val_distri = get_val_distribution(val_loader)
        print(f"Validation set distribution: {val_distri}")

        trainer = Trainer(
            model=model,
            dataloaders=dataloaders,
            config=training_config,
            evaluation=False,
        )

        trainer.train()

    else:

        model = Model(**model_config).to(device)

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

        print(
            f"Successfully loaded data with sizes : "
            f"test={len(test_loader.dataset) if test_loader is not None else 'N/A'}"
        )

        trainer = Trainer(
            model=model,
            dataloaders=dataloaders,
            config=training_config,
            evaluation=True,
        )

        preds = trainer.evaluate()

        output_path = Path(config.get("submission_path", "submission.csv"))
        create_submission_file(preds, output_path)

        print(f"Submission saved to: {output_path}")

        print("\nSample predictions:")
        for i, (k, v) in enumerate(sorted(preds.items())):
            if i >= 20:
                break
            print(k, v)




if __name__ == "__main__":
    config_path = Path("/home/infres/yrothlin-24/CHAL_IM05/configs/training.yaml")
    config = get_config(config_path)
    # main(config)
    main_bis(config)