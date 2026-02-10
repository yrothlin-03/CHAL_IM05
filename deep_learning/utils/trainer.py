import os
import time
import torch
import torch.nn as nn
import numpy as np
import math
from .metrics import Metrics

class Trainer:
    def __init__(self, config, model, dataloaders, evaluation=False):
        self.config = config
        self.model = model
        self.evaluation = evaluation

        self.device = next(model.parameters()).device
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.test_loader = dataloaders.get("test", None)

        self.use_amp = config.get("use_amp", False)
        self.grad_clip = config.get("grad_clip", None)
        self.optimizer_name = config.get("optimizer", "adamw").lower()
        self.scheduler_name = config.get("scheduler", "cosine").lower()

        self.weight_decay = config.get("weight_decay", 0.01)
        self.lr = config.get("lr", 1e-3)
        self.num_epochs = config.get("num_epochs", 20)

        self.warmup_steps = config.get("warmup_steps", 0)
        self.warmup_ratio = config.get("warmup_ratio", 0.1)

        self.loss_name = config.get("loss", "cross_entropy").lower()

        self.gamma = config.get("gamma", 0.1)
        self.step_size_steps = config.get("step_size_steps", 1000)
        self.log_step = config.get("log_step", 50)

        self.max_step_train = config.get("max_step_train", None)
        self.max_step_val = config.get("max_step_val", None)
        self.max_step_test = config.get("max_step_test", None)
        self.label_smoothing = config.get("label_smoothing", 0.1)
        self.weights = config.get("weights", None)

        self.save_ckpt = bool(config.get("save_ckpt", False))
        self.ckpt_dir = str(config.get("ckpt_dir", "checkpoints"))
        self.ckpt_every = int(config.get("ckpt_every", 1))
        self.monitor = str(config.get("monitor", "macro_f1"))
        self.best_score = float("-inf")
        if self.save_ckpt:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        
        if not self.evaluation:
            steps_per_epoch = len(self.train_loader) if self.max_step_train is None else min(len(self.train_loader), self.max_step_train)
            self.total_steps = max(1, self.num_epochs * steps_per_epoch)
            self.global_step = 0
            self.global_epoch = 0

            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            self.criterion = self._build_criterion(label_smoothing=self.label_smoothing, loss_name=self.loss_name, alpha=self.weights)

            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

            self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}

        self.resume_from = config.get("resume_from", None)
        self.resume = bool(config.get("resume", True))

        if self.resume_from:
            self._load_checkpoint(self.resume_from, resume=self.resume)
            print(f"Resumed from checkpoint: {self.resume_from}")

    def _load_checkpoint(self, path: str, resume: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        msd = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(msd, strict=True)

        # if not resume:
        #     return

        # if ckpt.get("optimizer_state_dict") is not None and self.optimizer is not None:
        #     self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # if ckpt.get("scheduler_state_dict") is not None and self.scheduler is not None:
        #     self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # if ckpt.get("scaler_state_dict") is not None and self.scaler is not None:
        #     self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        # self.global_epoch = int(ckpt.get("epoch", 0))
        # self.global_step = int(ckpt.get("global_step", 0))
        # self.best_score = float(ckpt.get("best_score", self.best_score))

    def _save_checkpoint(self, name: str, val_metrics: dict | None = None):
        if not self.save_ckpt:
            return
        path = os.path.join(self.ckpt_dir, name)
        torch.save(
            {
                "epoch": self.global_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
                "config": self.config,
                "val_metrics": val_metrics,
                "best_score": self.best_score,
            },
            path,
        )

    def _build_optimizer(self):
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        raise ValueError(f"Optimizer {self.optimizer_name} not supported.")

    def _build_scheduler(self):
        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_steps)

        if self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size_steps, gamma=self.gamma)

        if self.scheduler_name == "warmup_cosine":
            warmup = int(self.warmup_steps)
            warmup = max(0, min(warmup, self.total_steps - 1))

            def lr_lambda(current_step: int):
                if warmup == 0:
                    return 0.5 * (1 + math.cos(math.pi * current_step / self.total_steps))
                if current_step < warmup:
                    return self.warmup_ratio + (1 - self.warmup_ratio) * (current_step / warmup)
                progress = (current_step - warmup) / (self.total_steps - warmup)
                return 0.5 * (1 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        raise ValueError(f"Scheduler {self.scheduler_name} not supported.")

    def _build_criterion(self, loss_name="cross_entropy", label_smoothing=0.1, gamma=2.0, alpha=None):
        w = self.weights
        w = torch.tensor(w, dtype=torch.float32, device=self.device) if w is not None else None

        if loss_name == "cross_entropy":
            if label_smoothing and label_smoothing > 0:
                return nn.CrossEntropyLoss(weight=w, label_smoothing=float(label_smoothing))
            return nn.CrossEntropyLoss(weight=w)

        if loss_name == "focal":
            alpha_t = None
            if alpha is not None:
                alpha_t = torch.tensor(alpha, dtype=torch.float32, device=self.device)
            elif w is not None:
                alpha_t = w

            def focal_loss(logits, targets):
                ce = nn.functional.cross_entropy(logits, targets, reduction="none")
                pt = torch.exp(-ce)
                if alpha_t is not None:
                    at = alpha_t.gather(0, targets)
                    return (at * (1 - pt) ** gamma * ce).mean()
                return ((1 - pt) ** gamma * ce).mean()

            return focal_loss

        raise ValueError(f"Unknown loss_name: {loss_name}")

    def _format_metrics(self, m: dict, top_k: int = 5):
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

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batch = len(self.train_loader) if self.max_step_train is None else min(len(self.train_loader), self.max_step_train)

        data_times = []
        infer_times = []
        it = iter(self.train_loader)

        for step in range(n_batch):
            t0 = time.perf_counter()
            inputs, targets = next(it)
            dt = time.perf_counter() - t0
            if step < 30:
                data_times.append(dt)

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if step < 30 and self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            if step < 30 and self.device.type == "cuda":
                torch.cuda.synchronize()
            itime = time.perf_counter() - t1
            if step < 30:
                infer_times.append(itime)

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            if (step + 1) % self.log_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {step+1}/{n_batch} | loss={loss.item():.4f} lr={lr:.6f}")

        if data_times and infer_times:
            print(f"Timing (first {min(30, n_batch)} train batches) | data={sum(data_times)/len(data_times):.6f}s infer={sum(infer_times)/len(infer_times):.6f}s")

        return total_loss / max(1, n_batch)

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0.0
        metrics = Metrics()
        n_batch = len(self.val_loader) if self.max_step_val is None else min(len(self.val_loader), self.max_step_val)
        dist = {}

        for step, (inputs, targets) in enumerate(self.val_loader):
            if self.max_step_val is not None and step >= self.max_step_val:
                break

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            t = targets.detach().cpu().numpy()
            metrics.update(t, preds)

            for lab in t:
                lab = int(lab)
                dist[lab] = dist.get(lab, 0) + 1

        print(f"Val dist: {dist}")
        m = metrics.compute()
        print(self._format_metrics(m))
        return total_loss / max(1, n_batch), m

    @torch.no_grad()
    def evaluate(self):
        if self.test_loader is None:
            raise ValueError("No test dataloader provided (dataloaders['test']).")

        self.model.eval()
        preds = {}

        for step, (inputs, filenames) in enumerate(self.test_loader):
            if self.max_step_test is not None and step >= self.max_step_test:
                break

            inputs = inputs.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)

            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for fname, pred in zip(filenames, batch_preds):
                preds[str(fname)] = int(pred)

        return preds

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.global_epoch = epoch
            train_loss = self.train_one_epoch()
            val_loss, val_metrics = self.validate_one_epoch()

            print(f"Epoch {epoch}/{self.num_epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            if self.save_ckpt and (epoch % self.ckpt_every == 0):
                self._save_checkpoint("last.pt", val_metrics)

            score = float(val_metrics.get(self.monitor, float("-inf")))
            if self.save_ckpt and score > self.best_score:
                self.best_score = score
                self._save_checkpoint("best.pt", val_metrics)