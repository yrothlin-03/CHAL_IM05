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

def main(config: Dict[str, Any]):

    evaluation = config.get("evaluation", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]


    if evaluation:
        n_folds = 5
        logits_sum = {}

        test_loader = get_loaders(**data_config, test=True)
        dataloaders = {"train": None, "val": None, "test": test_loader}

        for fold in range(n_folds):
            model = Model(**model_config).to(device)

            fold_config = deepcopy(training_config)
            fold_config["resume_from"] = str(
                Path(training_config.get("ckpt_dir", "checkpoints")) / f"fold_{fold}" / "best.pt"
            )
            print(f"!!!!!!! Resume checkpoint for fold {fold}: {fold_config['resume_from']}")

            trainer = Trainer(
                model=model,
                dataloaders=dataloaders,
                config=fold_config,
                evaluation=True,
            )

            preds_logits = trainer.evaluate(return_logits=True)

            for filename, logits in preds_logits.items():
                if not torch.is_tensor(logits):
                    logits = torch.tensor(logits)
                logits = logits.detach().cpu()

                if filename in logits_sum:
                    logits_sum[filename] += logits
                else:
                    logits_sum[filename] = logits.clone()

        preds = {
            fn: int(torch.argmax(lg / n_folds).item())
            for fn, lg in logits_sum.items()
        }



        # model = Model(**model_config).to(device)
        # dataloader = get_loaders(**data_config, test=True)
        # dataloaders = {"train": None, "val": None, "test": dataloader}
        # print(f"Succesfully loaded test data with size: {len(dataloaders['test'].dataset) if dataloaders['test'] else 'N/A'}")
        # trainer = Trainer(model=model, dataloaders=dataloaders, config=training_config, evaluation=evaluation)
        # print(f"Trainer initialized with config: {training_config}")
        # preds = trainer.evaluate()


        create_submission_file(preds, Path("/home/infres/yrothlin-24/CHAL_IM05/submissions/submission9.csv"))

        for i, (k, v) in enumerate(preds.items()):
            if i >= 50:
                break
            print(k, v)

    else:
        for fold in range(5):
            fold_config = deepcopy(training_config)
            fold_config["ckpt_dir"] = str(Path(training_config.get("ckpt_dir", "checkpoints")) / f"fold_{fold}")
            model = Model(**model_config).to(device)
            train_loader, val_loader = get_loaders(**data_config, test=False, n_splits=5, fold_index=fold)
            dataloaders = {"train": train_loader, "val": val_loader, "test": None}
            val_distri = get_val_distribution(val_loader)
            print(f"Validation set distribution: {val_distri}")
            print(f"Succesfully loaded data with sizes : train={len(dataloaders['train'].dataset) if dataloaders['train'] else 'N/A'}, val={len(dataloaders['val'].dataset) if dataloaders['val'] else 'N/A'}, test={len(dataloaders['test'].dataset) if dataloaders['test'] else 'N/A'}")
            print(f"!!!!!!! Checkpoint directory for fold {fold}: {fold_config['ckpt_dir']}")
            trainer = Trainer(model=model, dataloaders=dataloaders, config=fold_config, evaluation=evaluation)
            print(f"Trainer initialized with config: {fold_config}")
            trainer.train()

        # model = Model(**model_config).to(device)
        # train_loader, val_loader = get_loaders(**data_config, test=False)
        # dataloaders = {"train": train_loader, "val": val_loader, "test": None}
        # print(f"Succesfully loaded data with sizes : train={len(dataloaders['train'].dataset) if dataloaders['train'] else 'N/A'}, val={len(dataloaders['val'].dataset) if dataloaders['val'] else 'N/A'}, test={len(dataloaders['test'].dataset) if dataloaders['test'] else 'N/A'}")
        # trainer = Trainer(model=model, dataloaders=dataloaders, config=training_config, evaluation=evaluation)
        # print(f"Trainer initialized with config: {training_config}")
        # trainer.train()




    


if __name__ == "__main__":
    config_path = Path("/home/infres/yrothlin-24/CHAL_IM05/configs/training.yaml")
    config = get_config(config_path)
    main(config)