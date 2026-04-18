import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


class ContrastiveTrainer:
    def __init__(self, config, model, dataloaders):
        self.config = config
        print(f"ContrastiveTrainer config: {self.config}")

        self.model = model
        self.device = next(model.parameters()).device

        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders.get("val", None)

        self.use_amp = bool(config.get("use_amp", False))
        self.grad_clip = config.get("grad_clip", None)
        self.amp_dtype = torch.bfloat16

        self.optimizer_name = str(config.get("optimizer", "adamw")).lower()
        self.scheduler_name = str(config.get("scheduler", "cosine")).lower()

        self.lr = float(config.get("lr", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 1e-4))

        self.num_epochs = int(config.get("num_epochs", 20))
        self.temperature = float(config.get("temperature", 0.2))
        self.log_step = int(config.get("log_step", 50))

        self.max_step_train = config.get("max_step_train", None)
        self.max_step_val = config.get("max_step_val", None)

        self.save_ckpt = bool(config.get("save_ckpt", False))
        self.ckpt_dir = str(config.get("ckpt_dir", "checkpoints_contrastive"))
        self.best_score = float("inf")

        if self.save_ckpt:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        steps_per_epoch = len(self.train_loader)
        if self.max_step_train is not None:
            steps_per_epoch = min(steps_per_epoch, self.max_step_train)
        self.total_steps = max(1, self.num_epochs * steps_per_epoch)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.device.type == "cuda" else "cpu",
            enabled=False,
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

        self.plot_tsne = bool(config.get("plot_tsne", False))
        self.tsne_split = str(config.get("tsne_split", "val"))
        self.tsne_every = int(config.get("tsne_every", 1))
        self.tsne_max_samples = int(config.get("tsne_max_samples", 1000))
        self.tsne_perplexity = float(config.get("tsne_perplexity", 30.0))
        self.tsne_dir = str(config.get("tsne_dir", os.path.join(self.ckpt_dir, "tsne")))
        self.class_names = config.get("class_names", [f"class {i}" for i in range(13)])

        if self.plot_tsne:
            os.makedirs(self.tsne_dir, exist_ok=True)
            print(f"TSNE will be plotted every {self.tsne_every} epochs and saved to: {self.tsne_dir}")

        x1, x2, _ = next(iter(self.train_loader))
        x1 = x1.to(self.device, non_blocking=True)
        x2 = x2.to(self.device, non_blocking=True)
        with torch.no_grad():
            _ = self.model(x1[:2])
            _ = self.model(x2[:2])

        self.global_step = 0
        self.global_epoch = 0

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

        raise ValueError(f"Optimizer {self.optimizer_name} not supported.")

    def _build_scheduler(self):
        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps
            )

        if self.scheduler_name == "step":
            step_size = int(self.config.get("step_size_steps", 1000))
            gamma = float(self.config.get("gamma", 0.1))
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )

        if self.scheduler_name == "none":
            return None

        raise ValueError(f"Scheduler {self.scheduler_name} not supported.")

    def _save_checkpoint(self, name: str, val_loss: float | None = None):
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
                "val_loss": val_loss,
                "best_score": self.best_score,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

    def supcon_loss(self, features, labels):
        # print(f"Computing SupCon loss with features shape {features.shape} and labels shape {labels.shape}")
        features = nn.functional.normalize(features.float(), dim=2)
        labels = labels.contiguous().view(-1, 1)

        batch_size = labels.shape[0]
        if batch_size <= 1:
            raise ValueError("Batch size must be > 1 for SupCon.")

        mask = torch.eq(labels, labels.T).float().to(features.device)

        contrast_count = 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size * anchor_count, device=features.device).view(-1, 1),
            0,
        )

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(mask_sum, min=1.0)

        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        n_batch = len(self.train_loader)
        if self.max_step_train is not None:
            n_batch = min(n_batch, self.max_step_train)

        data_times = []
        infer_times = []

        it = iter(self.train_loader)

        for step in range(n_batch):
            t0 = time.perf_counter()
            x1, x2, y = next(it)
            dt = time.perf_counter() - t0
            if step < 30:
                data_times.append(dt)

            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if step < 30 and self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                z1 = self.model(x1)
                z2 = self.model(x2)

            features = torch.stack([z1, z2], dim=1)
            loss = self.supcon_loss(features, y)

            if step < 30 and self.device.type == "cuda":
                torch.cuda.synchronize()
            itime = time.perf_counter() - t1
            if step < 30:
                infer_times.append(itime)

            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            if (step + 1) % self.log_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Step {step+1}/{n_batch} | loss={total_loss/(step+1):.4f} lr={lr:.6f}")

        if data_times and infer_times:
            print(
                f"Timing (first {min(30, n_batch)} train batches) | "
                f"data={sum(data_times)/len(data_times):.6f}s "
                f"infer={sum(infer_times)/len(infer_times):.6f}s"
            )

        return total_loss / max(1, n_batch)

    @torch.no_grad()
    def validate_one_epoch(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0

        n_batch = len(self.val_loader)
        if self.max_step_val is not None:
            n_batch = min(n_batch, self.max_step_val)

        for step, (x1, x2, y) in enumerate(self.val_loader):
            if self.max_step_val is not None and step >= self.max_step_val:
                break

            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                z1 = self.model(x1)
                z2 = self.model(x2)

            features = torch.stack([z1, z2], dim=1)
            loss = self.supcon_loss(features, y)
            total_loss += loss.item()

        val_loss = total_loss / max(1, n_batch)
        print(f"[VAL] supcon_loss={val_loss:.4f}")
        return val_loss

    @torch.no_grad()
    def extract_features_for_tsne(self, loader, max_samples: int = 1000):
        self.model.eval()

        feats = []
        labels = []

        total = 0
        for x1, _, y in loader:
            x1 = x1.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                z = self.model.backbone(x1)

            feats.append(z.detach().cpu())
            labels.append(y.detach().cpu())

            total += x1.size(0)
            if total >= max_samples:
                break

        feats = torch.cat(feats, dim=0)[:max_samples]
        labels = torch.cat(labels, dim=0)[:max_samples]

        return feats.numpy(), labels.numpy()

    @torch.no_grad()
    def save_tsne(self, split: str = "val", epoch: int | None = None):
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            raise ValueError(f"Unknown split: {split}")

        if loader is None:
            print(f"[TSNE] No loader available for split={split}")
            return

        features, labels = self.extract_features_for_tsne(
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
        n_classes = len(self.class_names)

        cmap = plt.get_cmap("tab20", n_classes)
        norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5, 1), cmap.N)

        plt.scatter(
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
                label=self.class_names[int(cls)] if int(cls) < len(self.class_names) else f"class {int(cls)}",
                markerfacecolor=cmap(int(cls)),
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

        plt.title(f"Contrastive t-SNE ({split})" + (f" - epoch {epoch}" if epoch is not None else ""))
        plt.tight_layout()

        filename = f"tsne_{split}"
        if epoch is not None:
            filename += f"_epoch_{epoch:03d}"
        filename += ".png"

        out_path = os.path.join(self.tsne_dir, filename)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[TSNE] Saved to {out_path}")

    def train(self):
        best_epoch = None

        for epoch in range(1, self.num_epochs + 1):
            self.global_epoch = epoch

            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()

            if self.plot_tsne and (epoch % self.tsne_every == 0 or epoch == 1):
                self.save_tsne(split=self.tsne_split, epoch=epoch)

            print(
                f"[RESULTS] Epoch {epoch}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f}"
                + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if self.save_ckpt and epoch == self.num_epochs:
                self._save_checkpoint("last.pt", val_loss)

            score = val_loss if val_loss is not None else train_loss
            if self.save_ckpt and score < self.best_score:
                self.best_score = score
                best_epoch = epoch
                print(f"New best supcon loss: {self.best_score:.4f} at epoch {epoch}. Saving checkpoint.")
                self._save_checkpoint("best.pt", val_loss)

        print(f"[FINAL] Best epoch: {best_epoch} | best_loss={self.best_score:.4f}")