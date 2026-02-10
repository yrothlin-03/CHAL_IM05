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


def main(config: Dict[str, Any]):

    evaluation = config.get("evaluation", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    if evaluation:
        dataloader = get_loaders(**data_config, test=True)
        dataloaders = {"train": None, "val": None, "test": dataloader}
        print(f"Succesfully loaded test data with size: {len(dataloaders['test'].dataset) if dataloaders['test'] else 'N/A'}")
    else:
        train_loader, val_loader = get_loaders(**data_config, test=False)
        dataloaders = {"train": train_loader, "val": val_loader, "test": None}
        print(f"Succesfully loaded data with sizes : train={len(dataloaders['train'].dataset) if dataloaders['train'] else 'N/A'}, val={len(dataloaders['val'].dataset) if dataloaders['val'] else 'N/A'}, test={len(dataloaders['test'].dataset) if dataloaders['test'] else 'N/A'}")

    model = Model(**model_config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")

    trainer = Trainer(model=model, dataloaders=dataloaders, config=training_config, evaluation=evaluation)
    print(f"Trainer initialized with config: {training_config}")

    if evaluation:
        preds = trainer.evaluate()
        create_submission_file(preds, Path("/home/infres/yrothlin-24/CHAL_IM05/submissions/submission1.csv"))
    else:
        trainer.train()
    
    for i, (k, v) in enumerate(preds.items()):
        if i >= 50:
            break
        print(k, v)


    


if __name__ == "__main__":
    config_path = Path("/home/infres/yrothlin-24/CHAL_IM05/configs/training.yaml")
    config = get_config(config_path)
    main(config)