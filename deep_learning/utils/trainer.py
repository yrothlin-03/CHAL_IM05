import os
import time
import torch
import torch.nn as nn
import numpy as np
import math
from .metrics import Metrics
import matplotlib
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = float(gamma)
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=float(label_smoothing),
            reduction="none",
        )

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()



class Trainer:
    def __init__(self, config, model, dataloaders, evaluation=False):
        self.config = config
        print(f"Trainer config: {self.config}")
        self.model = model
        self.evaluation = evaluation
        print(f"MODE EVALUATION : {self.evaluation}")

        self.device = next(model.parameters()).device
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.val_tta_loader = dataloaders.get("val_tta", None)
        self.test_loader = dataloaders.get("test", None)
        self.test_tta_loader = dataloaders.get("test_tta", None)
        self.tta_rounds = int(config.get("tta_rounds", 0))

        if self.tta_rounds != 0:
            print(f"Using TTA with : {self.tta_rounds} rounds")
        x = None
        if self.test_loader is not None:
            x = next(iter(self.test_loader))[0].to(self.device, non_blocking=True)
        elif self.train_loader is not None:
            x = next(iter(self.train_loader))[0].to(self.device, non_blocking=True)

        with torch.no_grad():
            if x is not None:
                self.model(x)

        self.use_amp = config.get("use_amp", False)
        self.grad_clip = config.get("grad_clip", None)
        self.optimizer_name = config.get("optimizer", "adamw").lower()
        self.scheduler_name = config.get("scheduler", "cosine").lower()

        self.use_mixup = bool(config.get("use_mixup", False))
        self.mixup_alpha = float(config.get("mixup_alpha", 0.2))

        self.use_cutmix = bool(config.get("use_cutmix", False))
        self.cutmix_alpha = float(config.get("cutmix_alpha", 1.0))
        self.cutmix_prob = float(config.get("cutmix_prob", 0.5))

        self.weight_decay = config.get("weight_decay", 0.01)
        self.bb_lr = config.get("bb_lr", 0.0001)
        self.head_lr = config.get("head_lr", 0.001)

        self.num_epochs = config.get("num_epochs", 20)
        self.warmup_epochs = config.get("warmup_epochs", 5)

        self.finetune_bb_lr = config.get("finetune_bb_lr", self.bb_lr)
        self.finetune_head_lr = config.get("finetune_head_lr", self.head_lr)

        self.warmup_steps = config.get("warmup_steps", 0)
        self.warmup_ratio = config.get("warmup_ratio", 0.1)

        self.loss_name = config.get("loss", "focal").lower()
        self.use_weights = config.get("use_weights", False)

        self.gamma = config.get("gamma", 2.0)
        self.step_size_steps = config.get("step_size_steps", 1000)
        self.log_step = config.get("log_step", 50)

        self.max_step_train = config.get("max_step_train", None)
        self.max_step_val = config.get("max_step_val", None)
        self.max_step_test = config.get("max_step_test", None)
        self.label_smoothing = config.get("label_smoothing", 0.1)
        self.weights = config.get("weights", None)

        self.rare_classes_for_no_aug = torch.tensor(
            config.get("rare_classes_for_no_aug", []),
            device=self.device,
            dtype=torch.long,
        )

        self.save_ckpt = bool(config.get("save_ckpt", False))
        self.ckpt_dir = str(config.get("ckpt_dir", "checkpoints"))
        self.ckpt_every = int(config.get("ckpt_every", 1))
        self.monitor = str(config.get("monitor", "macro_f1"))

        self.early_stopping = bool(config.get("early_stopping", False))
        self.early_stopping_patience = int(config.get("early_stopping_patience", 10))
        self.early_stopping_mode = str(config.get("early_stopping_mode", "max")).lower()
        self.early_stopping_counter = 0
        if self.early_stopping:
            print(f"Early stopping enabled with patience={self.early_stopping_patience} and mode={self.early_stopping_mode}")

        self.best_score = float("-inf") if self.early_stopping_mode == "max" else float("inf")

        if self.save_ckpt:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        if not self.evaluation:
            steps_per_epoch = len(self.train_loader) if self.max_step_train is None else min(len(self.train_loader), self.max_step_train)
            self.steps_per_epoch = max(1, steps_per_epoch)

            self.warmup_phase_epochs = max(0, min(self.warmup_epochs, self.num_epochs))
            self.finetune_phase_epochs = max(0, self.num_epochs - self.warmup_phase_epochs)

            self.total_steps = max(1, self.num_epochs * self.steps_per_epoch)
            self.global_step = 0
            self.global_epoch = 0

            self.optimizer = self._build_optimizer(phase="warmup")
            self.scheduler = self._build_scheduler(phase="warmup")
            self.criterion = self._build_criterion(
                label_smoothing=self.label_smoothing,
                loss_name=self.loss_name,
                gamma=self.gamma,
                use_weights=self.use_weights,
            )

            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

            self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}

        self.resume_from = config.get("resume_from", None)
        self.resume = bool(config.get("resume", True))
        self.resume_from_pretrained = config.get("resume_from_pretrained", None)
        self.resume_pretrained = bool(config.get("resume_pretrained", True))

        if self.resume_from_pretrained is not None and self.resume_pretrained:
            self._load_checkpoint_backbone(self.resume_from_pretrained, resume=self.resume)


        if self.resume and self.evaluation:
            self._load_checkpoint(self.resume_from, resume=self.resume)
            print(f"Resumed from checkpoint: {self.resume_from}")


        self.plot_tsne = bool(config.get("plot_tsne", False))
        self.tsne_split = str(config.get("tsne_split", "val"))
        self.tsne_every = int(config.get("tsne_every", 1))
        self.tsne_max_samples = int(config.get("tsne_max_samples", 1000))
        self.tsne_perplexity = float(config.get("tsne_perplexity", 30.0))
        self.tsne_dir = str(config.get("tsne_dir", os.path.join(self.ckpt_dir, "tsne")))
        if self.plot_tsne:
            os.makedirs(self.tsne_dir, exist_ok=True)
            print(f"TSNE will be plotted every {self.tsne_every} epochs and saved to: {self.tsne_dir}")

        self.plot_confusion = bool(config.get("plot_confusion", False))
        self.confusion_split = str(config.get("confusion_split", "val"))
        self.confusion_every = int(config.get("confusion_every", 1))
        self.confusion_normalize = bool(config.get("confusion_normalize", False))
        self.confusion_dir = str(config.get("confusion_dir", os.path.join(self.ckpt_dir, "confusion_matrix")))
        self.class_names = config.get("class_names", [f"class {i}" for i in range(13)])

        if self.plot_confusion:
            os.makedirs(self.confusion_dir, exist_ok=True)
            print(f"Confusion matrices will be plotted every {self.confusion_every} epochs and saved to: {self.confusion_dir}")



    def _is_improvement(self, score: float) -> bool:
        if self.early_stopping_mode == "max":
            return score > self.best_score
        if self.early_stopping_mode == "min":
            return score < self.best_score
        raise ValueError(f"Unsupported early_stopping_mode: {self.early_stopping_mode}")


    def _load_checkpoint(self, path: str, resume: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        msd = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(msd, strict=True)

    def _load_checkpoint_backbone(self, path: str, resume: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        msd = ckpt.get("model_state_dict", ckpt)

        backbone_state_dict = {}
        for k, v in msd.items():
            if k.startswith("backbone.backbone."):
                new_key = k.replace("backbone.", "", 1)
                backbone_state_dict[new_key] = v

        missing, unexpected = self.model.backbone.load_state_dict(
            backbone_state_dict,
            strict=False,
        )

        print(f"[LOAD BACKBONE] Loaded from: {path}")
        print(f"Missing keys   : {missing}")
        print(f"Unexpected keys: {unexpected}")

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
        print(f"Checkpoint saved: {path}")



    def _get_phase_total_steps(self, phase: str) -> int:
        if phase == "warmup":
            phase_epochs = self.warmup_phase_epochs if self.warmup_phase_epochs > 0 else 1
        else:
            phase_epochs = self.finetune_phase_epochs if self.finetune_phase_epochs > 0 else 1
        return max(1, phase_epochs * self.steps_per_epoch)
    

    def _build_optimizer(self, phase: str = "warmup"):
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        if phase == "warmup":
            bb_lr = self.bb_lr
            head_lr = self.head_lr
        else:
            bb_lr = self.finetune_bb_lr
            head_lr = self.finetune_head_lr

        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": bb_lr},
                    {"params": head_params, "lr": head_lr},
                ],
                weight_decay=self.weight_decay,
            )

        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                [
                    {"params": backbone_params, "lr": bb_lr},
                    {"params": head_params, "lr": head_lr},
                ],
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

        raise ValueError(f"Optimizer {self.optimizer_name} not supported.")



    def _build_scheduler(self, phase: str = "warmup"):
        if self.scheduler_name == "cosine_restarts":
            if phase == "warmup":
                t0 = 15
            else:
                remaining = max(1, self.num_epochs - self.warmup_epochs)
                t0 = max(remaining // 3, 5)

            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t0
            )

        if self.scheduler_name == "cosine":
            phase_total_steps = self._get_phase_total_steps(phase)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=phase_total_steps
            )

        if self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.step_size_steps,
                gamma=self.gamma
            )

        if self.scheduler_name == "warmup_cosine":
            phase_total_steps = self._get_phase_total_steps(phase)
            warmup = int(self.warmup_steps)
            warmup = max(0, min(warmup, phase_total_steps - 1))

            def lr_lambda(current_step: int):
                if warmup == 0:
                    return 0.5 * (1 + math.cos(math.pi * current_step / phase_total_steps))
                if current_step < warmup:
                    return self.warmup_ratio + (1 - self.warmup_ratio) * (current_step / warmup)
                progress = (current_step - warmup) / (phase_total_steps - warmup)
                return 0.5 * (1 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        raise ValueError(f"Scheduler {self.scheduler_name} not supported.")



    def _build_criterion(self, loss_name="focal", label_smoothing=0.05, gamma=2.0, use_weights=False):
        w = self.weights if use_weights else None
        w = torch.tensor(w, dtype=torch.float32, device=self.device) if w is not None else None

        loss_name = str(loss_name).lower()

        if loss_name in {"ce", "cross_entropy"}:
            return nn.CrossEntropyLoss(
                weight=w,
                label_smoothing=float(label_smoothing),
            )

        if loss_name == "focal":
            return FocalLoss(
                weight=w,
                gamma=float(gamma),
                label_smoothing=float(label_smoothing),
            )

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


    def _freeze_backbone(self): 
        self.model.freeze_backbone()

    def _unfreeze_backbone(self):
        self.model.unfreeze_backbone()



    def _mixup_batch(self, x, y, alpha=0.2):
        if alpha <= 0:
            return x, y, y, 1.0

        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0), device=x.device)

        mixed_x = lam * x + (1.0 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def _cutmix_batch(self, x, y, alpha=1.0):
        if alpha <= 0:
            return x, y, y, 1.0

        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0), device=x.device)

        _, _, H, W = x.shape

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        box_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - box_area / float(H * W)

        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _apply_batch_aug(self, x, y):
        if self.use_mixup and self.use_cutmix:
            if np.random.rand() < self.cutmix_prob:
                return self._cutmix_batch(x, y, alpha=self.cutmix_alpha)
            return self._mixup_batch(x, y, alpha=self.mixup_alpha)

        if self.use_cutmix:
            return self._cutmix_batch(x, y, alpha=self.cutmix_alpha)

        if self.use_mixup:
            return self._mixup_batch(x, y, alpha=self.mixup_alpha)

        return x, y, y, 1.0
    

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        metrics = Metrics()
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


            contains_rare = torch.isin(targets,self.rare_classes_for_no_aug).any()
            use_batch_aug = (self.use_mixup or self.use_cutmix) and not contains_rare

            if use_batch_aug:
                inputs, targets_a, targets_b, lam = self._apply_batch_aug(inputs, targets)

            if step < 30 and self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                if use_batch_aug:
                    loss = lam * self.criterion(outputs, targets_a) + (1.0 - lam) * self.criterion(outputs, targets_b)
                else:
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

            if self.scheduler is not None and self.scheduler_name in {"cosine", "step", "warmup_cosine"}:
                self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            t = targets.detach().cpu().numpy()
            metrics.update(t, preds)

            if (step + 1) % self.log_step == 0:
                lr_bb = self.optimizer.param_groups[0]["lr"]
                lr_head = self.optimizer.param_groups[1]["lr"]
                print(f"Step {step+1}/{n_batch} | loss={total_loss/(step+1):.4f} bb_lr={lr_bb:.6f} head_lr={lr_head:.6f}")

        if data_times and infer_times:
            print(f"Timing (first {min(30, n_batch)} train batches) | data={sum(data_times)/len(data_times):.6f}s infer={sum(infer_times)/len(infer_times):.6f}s")

        train_metrics = metrics.compute()
        print("[ TRAIN METRICS ]:")
        print(self._format_metrics(train_metrics))

        return total_loss / max(1, n_batch), train_metrics



    @torch.no_grad()
    def validate_one_epoch_tta(self):
        self.model.eval()

        base_logits = []
        tta_logits_sum = None
        all_targets = []
        total_loss = 0.0
        dist = {}

        n_batch = len(self.val_loader) if self.max_step_val is None else min(len(self.val_loader), self.max_step_val)

        for step, (inputs, targets) in enumerate(self.val_loader):
            if self.max_step_val is not None and step >= self.max_step_val:
                break

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            base_logits.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

            t = targets.detach().cpu().numpy()
            for lab in t:
                lab = int(lab)
                dist[lab] = dist.get(lab, 0) + 1

        for _ in range(self.tta_rounds):
            current_logits = []
            for step, (inputs, targets) in enumerate(self.val_tta_loader):
                if self.max_step_val is not None and step >= self.max_step_val:
                    break

                inputs = inputs.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)

                current_logits.append(outputs.detach().cpu())

            current_logits = torch.cat(current_logits, dim=0)
            if tta_logits_sum is None:
                tta_logits_sum = current_logits
            else:
                tta_logits_sum += current_logits

        base_logits = torch.cat(base_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        logits = base_logits
        if tta_logits_sum is not None:
            logits = (base_logits + tta_logits_sum) / (1 + self.tta_rounds)

        preds = torch.argmax(logits, dim=1).numpy()
        targets = all_targets.numpy()

        metrics = Metrics()
        metrics.update(targets, preds)
        m = metrics.compute()

        print("[ VALIDATION METRICS TTA]:")
        print(f"Val dist: {dist}")
        print(self._format_metrics(m))
        return total_loss / max(1, n_batch), m, targets, preds


    @torch.no_grad()
    def validate_one_epoch(self):
        if self.val_tta_loader is not None and self.tta_rounds > 0:
            return self.validate_one_epoch_tta()

        self.model.eval()
        total_loss = 0.0
        metrics = Metrics()
        n_batch = len(self.val_loader) if self.max_step_val is None else min(len(self.val_loader), self.max_step_val)
        dist = {}

        all_targets = []
        all_preds = []

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
            all_targets.append(t)
            all_preds.append(preds)

            for lab in t:
                lab = int(lab)
                dist[lab] = dist.get(lab, 0) + 1

        all_targets = np.concatenate(all_targets, axis=0) if len(all_targets) > 0 else np.array([], dtype=np.int64)
        all_preds = np.concatenate(all_preds, axis=0) if len(all_preds) > 0 else np.array([], dtype=np.int64)

        print("[ VALIDATION METRICS]:")
        print(f"Val dist: {dist}")
        m = metrics.compute()
        print(self._format_metrics(m))
        return total_loss / max(1, n_batch), m, all_targets, all_preds
    

    @torch.no_grad()
    def evaluate_tta(self, return_logits=False):
        if self.test_loader is None:
            raise ValueError("No test dataloader provided (dataloaders['test']).")
        if self.test_tta_loader is None or self.tta_rounds <= 0:
            return self.evaluate(return_logits=return_logits)

        self.model.eval()

        base_logits = {}
        tta_logits_sum = {}

        for step, (inputs, filenames) in enumerate(self.test_loader):
            if self.max_step_test is not None and step >= self.max_step_test:
                break

            inputs = inputs.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)

            for fname, logit in zip(filenames, outputs.detach().cpu()):
                base_logits[str(fname)] = logit.clone()

        for _ in range(self.tta_rounds):
            for step, (inputs, filenames) in enumerate(self.test_tta_loader):
                if self.max_step_test is not None and step >= self.max_step_test:
                    break

                inputs = inputs.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)

                for fname, logit in zip(filenames, outputs.detach().cpu()):
                    key = str(fname)
                    if key not in tta_logits_sum:
                        tta_logits_sum[key] = logit.clone()
                    else:
                        tta_logits_sum[key] += logit

        preds = {}
        preds_logits = {}

        for fname in base_logits.keys():
            logits = base_logits[fname]
            if fname in tta_logits_sum:
                logits = (logits + tta_logits_sum[fname]) / (1 + self.tta_rounds)

            if return_logits:
                preds_logits[fname] = logits.numpy()
            else:
                preds[fname] = int(torch.argmax(logits).item())

        return preds_logits if return_logits else preds


    @torch.no_grad()
    def evaluate(self, return_logits=False):
        if self.test_tta_loader is not None and self.tta_rounds > 0:
            return self.evaluate_tta(return_logits=return_logits)

        if self.test_loader is None:
            raise ValueError("No test dataloader provided (dataloaders['test']).")

        self.model.eval()
        preds = {}
        preds_logits = {}

        for step, (inputs, filenames) in enumerate(self.test_loader):
            if self.max_step_test is not None and step >= self.max_step_test:
                break

            inputs = inputs.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                if return_logits:
                    for fname, logit in zip(filenames, outputs.cpu().numpy()):
                        preds_logits[str(fname)] = logit

            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for fname, pred in zip(filenames, batch_preds):
                preds[str(fname)] = int(pred)

        if return_logits:
            return preds_logits

        return preds


    @torch.no_grad()
    def extract_backbone_features(self, loader, max_samples: int = 1000):
        self.model.eval()

        feats = []
        labels = []

        total = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                features = self.model.backbone(inputs)

            feats.append(features.detach().cpu())
            labels.append(targets.detach().cpu())

            total += inputs.size(0)
            if total >= max_samples:
                break

        feats = torch.cat(feats, dim=0)[:max_samples]
        labels = torch.cat(labels, dim=0)[:max_samples]

        return feats.numpy(), labels.numpy()



    @torch.no_grad()
    def save_backbone_tsne(self, split: str = "val", epoch: int | None = None):
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            raise ValueError(f"Unknown split: {split}")

        if loader is None:
            print(f"[TSNE] No loader available for split={split}")
            return

        features, labels = self.extract_backbone_features(
            loader=loader,
            max_samples=self.tsne_max_samples,
        )

        if len(features) < 2:
            print("[TSNE] Not enough samples to compute t-SNE")
            return

        perplexity = min(self.tsne_perplexity, max(2, len(features) - 1))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        emb_2d = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))

        classes = np.unique(labels)
        n_classes = int(classes.max()) + 1

        cmap = plt.get_cmap("tab20", n_classes) 
        norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5, 1), cmap.N)

        scatter = plt.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            c=labels,
            cmap=cmap,
            norm=norm,
            s=12,
            alpha=0.8,
        )

        legend_elements = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=f"class {cls}",
                markerfacecolor=cmap(cls),
                markersize=8,
            )
            for cls in classes
        ]

        plt.legend(
            handles=legend_elements,
            title="Classes",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.title(f"Backbone t-SNE ({split})" + (f" - epoch {epoch}" if epoch is not None else ""))
        plt.tight_layout()

        filename = f"tsne_{split}"
        if epoch is not None:
            filename += f"_epoch_{epoch:03d}"
        filename += ".png"

        out_path = os.path.join(self.tsne_dir, filename)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[TSNE] Saved to {out_path}")


    def save_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split: str = "val",
        epoch: int | None = None,
    ):
        if len(y_true) == 0 or len(y_pred) == 0:
            print("[CONFUSION] Empty targets or predictions, skipping.")
            return

        labels = np.arange(len(self.class_names))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if self.confusion_normalize:
            cm = cm.astype(np.float32)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

        plt.figure(figsize=(10, 8))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)

        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, self.class_names)

        fmt = ".2f" if self.confusion_normalize else "d"
        thresh = cm.max() / 2.0 if cm.size > 0 else 0.0

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = format(cm[i, j], fmt)
                plt.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

        title = f"Confusion Matrix ({split})"
        if epoch is not None:
            title += f" - epoch {epoch}"
        if self.confusion_normalize:
            title += " [normalized]"

        plt.title(title)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        filename = f"confusion_{split}"
        if epoch is not None:
            filename += f"_epoch_{epoch:03d}"
        if self.confusion_normalize:
            filename += "_norm"
        filename += ".png"

        out_path = os.path.join(self.confusion_dir, filename)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[CONFUSION] Saved to {out_path}")


    def train(self):
        best_epoch = -1

        for epoch in range(1, self.num_epochs + 1):
            if epoch == 1:
                self._freeze_backbone()
                self.optimizer = self._build_optimizer(phase="warmup")
                self.scheduler = self._build_scheduler(phase="warmup")
                print(
                    f"Backbone frozen at epoch {epoch}. "
                    f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
                )

            if epoch == self.warmup_epochs + 1:
                self._unfreeze_backbone()
                self.optimizer = self._build_optimizer(phase="finetune")
                self.scheduler = self._build_scheduler(phase="finetune")
                print(
                    f"Backbone unfrozen at epoch {epoch}. "
                    f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
                )

            self.global_epoch = epoch
            train_loss, train_metrics = self.train_one_epoch()

            if self.scheduler is not None and self.scheduler_name == "cosine_restarts":
                self.scheduler.step()

            val_loss, val_metrics, val_targets, val_preds = self.validate_one_epoch()

            if self.plot_tsne and (epoch % self.tsne_every == 0 or epoch == 1):
                self.save_backbone_tsne(split=self.tsne_split, epoch=epoch)

            if self.plot_confusion and (epoch % self.confusion_every == 0 or epoch == 1):
                self.save_confusion_matrix(
                    y_true=val_targets,
                    y_pred=val_preds,
                    split=self.confusion_split,
                    epoch=epoch,
                )

            print(
                f"[ RESULTS ] : Epoch {epoch}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            if self.save_ckpt and (self.global_epoch == self.num_epochs):
                self._save_checkpoint("last.pt", val_metrics)

            if self.monitor == "val_loss":
                score = float(val_loss)
            else:
                score = float(val_metrics.get(self.monitor, float("-inf")))

            improved = self._is_improvement(score)

            if improved:
                best_epoch = epoch
                self.best_score = score
                self.early_stopping_counter = 0

                print(
                    f"New best {self.monitor}: {self.best_score:.4f} "
                    f"at epoch {epoch}."
                )

                if self.save_ckpt:
                    print("Saving best checkpoint.")
                    self._save_checkpoint("best.pt", val_metrics)

            else:
                if self.early_stopping:
                    self.early_stopping_counter += 1
                    print(
                        f"No improvement on {self.monitor}. "
                        f"Early stopping counter: "
                        f"{self.early_stopping_counter}/{self.early_stopping_patience}"
                    )

                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"Best epoch was {best_epoch} with "
                            f"{self.monitor}={self.best_score:.4f}"
                        )
                        break

        print(
            f"[FINAL]: Best score achieved at epoch {best_epoch} "
            f"with score: {self.best_score}"
        )