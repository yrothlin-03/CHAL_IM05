from pathlib import Path
import logging
import json 
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional



def init_logger(log_dir: Path, logger_name: str) -> logging.Logger:
    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  

    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    log_path = log_dir / f"{logger_name}.log"

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger



class TrainLogger:
    def __init__(
        self,
        log_dir: Path,
        run_name: str,
        mode: str = "downstream",
        pretrain: Optional[str] = "tueg_2500",
        dataset: Optional[str] = None,
        debug: bool = False,
    ):  
        self.root = Path(log_dir).expanduser()
        parts = [Path(log_dir).expanduser(), mode, pretrain, dataset, run_name]
        self.log_dir = Path(*[p for p in parts if p]) 

        self.run_name = run_name
        self.is_debug = debug
        self.mode = mode
        self.pretrain = pretrain
        self.dataset = dataset

        self.logger = self._init_logger(self.log_dir, self.run_name)
        self.info(f"[INIT LOGGING] Logger initialized at {self.log_dir}")
        level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(level)
        for h in self.logger.handlers:
            h.setLevel(level)

    def _get_log_path(self) -> Path:
        parts = [self.log_dir, f"{self.run_name}.log"]
        return Path(*[p for p in parts if p])
        
    def _results_path(self, root: Path)-> Path:
        parts = [root, self.mode, self.pretrain, self.dataset, "results.json"]
        return Path(*[p for p in parts if p])


    def _init_logger(self, log_dir: Path, run_name: str) -> logging.Logger:
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(run_name)
        logger.propagate = False

        if logger.handlers:
            for h in list(logger.handlers):
                logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        log_path = log_dir / f"{run_name}.log"

        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        ch = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def plot_loss_curves(self, train_losses, val_losses, masked_losses=None, visible_losses=None, spectral_losses=None) -> None:
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        if masked_losses is not None:
            plt.plot(epochs, masked_losses, label="Masked Loss")
        if visible_losses is not None:
            plt.plot(epochs, visible_losses, label="Visible Loss")
        if spectral_losses is not None:
            plt.plot(epochs, spectral_losses, label="Spectral Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plot_path = self.log_dir / f"{self.run_name}_loss_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Loss curves saved to {plot_path}")
    
    def save_history(self, run_name: str, history: dict) -> None:
        results_path = self._results_path(self.root)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        if results_path.exists():
            with open(results_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = {}

        all_results[run_name] = history

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)

        self.logger.info(f"Results saved to {results_path}")

    
