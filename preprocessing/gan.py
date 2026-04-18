import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


label2id = {
    "SNE": 0, "LY": 1, "MO": 2, "EO": 3, "BA": 4, "VLY": 5, "BNE": 6,
    "MMY": 7, "MY": 8, "PMY": 9, "BL": 10, "PC": 11, "PLY": 12,
}


def get_base_id(filename: str) -> str:
    p = Path(filename)
    stem = p.stem
    suffix = p.suffix
    marker = "_aug_"
    if marker in stem:
        stem = stem.split(marker)[0]
    return f"{stem}{suffix}"


def get_filespath(dataset_dir: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                files.append(os.path.join(dirpath, filename))
    files.sort()
    return files


def get_labels(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        labels[str(row["ID"])] = str(row["label"])
    return labels


class WBCClassDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        labels_csv: str,
        target_class: str,
        image_size: int = 64,
        use_augmented: bool = True,
        unique_base_only: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.labels = get_labels(labels_csv)
        self.target_class = target_class
        self.use_augmented = use_augmented
        self.unique_base_only = unique_base_only

        all_files = get_filespath(str(self.root_dir))
        selected_files = []

        seen_base_ids = set()

        for f in all_files:
            fname = os.path.basename(f)
            base_id = get_base_id(fname)

            if base_id not in self.labels:
                continue

            if self.labels[base_id] != target_class:
                continue

            if not use_augmented and fname != base_id:
                continue

            if unique_base_only:
                if base_id in seen_base_ids:
                    continue
                seen_base_ids.add(base_id)

            selected_files.append(f)

        self.files = sorted(selected_files)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        if len(self.files) == 0:
            raise ValueError(
                f"No images found for class '{target_class}' in {root_dir} "
                f"with labels from {labels_csv}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return img


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, base_channels: int = 64, img_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(z_dim, base_channels * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(base_channels // 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, base_channels: int = 64, img_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(img_channels, base_channels // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128 -> 64x64
            nn.Conv2d(base_channels // 2, base_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(base_channels * 8, base_channels * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(base_channels * 16, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradients = grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def save_samples(generator, z_dim, device, output_dir, epoch, n=64):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, z_dim, 1, 1, device=device)
        fake = generator(z).cpu()
        fake = (fake + 1) / 2
        utils.save_image(fake, output_dir / f"samples_epoch_{epoch:04d}.png", nrow=8)
    generator.train()


def train_gan(
    data_dir: str,
    labels_csv: str,
    target_class: str,
    output_dir: str = "gan_wbc_outputs",
    image_size: int = 64,
    batch_size: int = 16,
    epochs: int = 300,
    z_dim: int = 128,
    lr: float = 1e-4,
    critic_steps: int = 5,
    lambda_gp: float = 10.0,
    num_workers: int = 4,
    use_augmented: bool = True,
    unique_base_only: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir) / target_class
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = WBCClassDataset(
        root_dir=data_dir,
        labels_csv=labels_csv,
        target_class=target_class,
        image_size=image_size,
        use_augmented=use_augmented,
        unique_base_only=unique_base_only,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(dataset) >= batch_size),
    )

    G = Generator(z_dim=z_dim).to(device)
    D = Critic().to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    print(f"Device: {device}")
    print(f"Target class: {target_class}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Steps per epoch: {len(loader)}")

    for epoch in range(1, epochs + 1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        for real in loader:
            real = real.to(device)
            cur_batch_size = real.size(0)

            for _ in range(critic_steps):
                z = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)
                fake = G(z)

                d_real = D(real).mean()
                d_fake = D(fake.detach()).mean()
                gp = gradient_penalty(D, real, fake.detach(), device)

                d_loss = -(d_real - d_fake) + lambda_gp * gp

                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()

            z = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)
            fake = G(z)
            g_loss = -D(fake).mean()

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            d_loss_epoch += d_loss.item()
            g_loss_epoch += g_loss.item()

        d_loss_epoch /= len(loader)
        g_loss_epoch /= len(loader)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}]  D Loss: {d_loss_epoch:.4f}  G Loss: {g_loss_epoch:.4f}")

        if epoch % 100 == 0 or epoch == 1:
            save_samples(G, z_dim, device, output_dir, epoch)

        if epoch % 100 == 0 or epoch == epochs:
            torch.save(G.state_dict(), output_dir / f"generator_epoch_{epoch:04d}.pt")
            torch.save(D.state_dict(), output_dir / f"critic_epoch_{epoch:04d}.pt")


def generate_images(
    checkpoint_path: str,
    output_dir: str = "generated_wbc",
    n_images: int = 100,
    z_dim: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = Generator(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()

    with torch.no_grad():
        for i in range(n_images):
            z = torch.randn(1, z_dim, 1, 1, device=device)
            fake = G(z)[0].cpu()
            fake = (fake + 1) / 2
            img = fake.permute(1, 2, 0).numpy()
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"gen_{i:05d}.png"), img)


if __name__ == "__main__":
    train_gan(
        data_dir="/home/infres/yrothlin-24/CHAL_IM05/IMA205-challenge_resampled/train",
        labels_csv="/home/infres/yrothlin-24/CHAL_IM05/IMA205-challenge_resampled/train_metadata.csv",
        target_class="PLY",
        output_dir="/home/infres/yrothlin-24/CHAL_IM05/gan_wbc_outputs",
        image_size=256,
        batch_size=16,
        epochs=2000,
        z_dim=256,
        lr=1e-4,
        critic_steps=5,
        lambda_gp=10.0,
        num_workers=4,
        use_augmented=True,
        unique_base_only=False,
    )

    # generate_images(
    #     checkpoint_path="/home/infres/yrothlin-24/CHAL_IM05/gan_wbc_outputs/PLY/generator_epoch_0300.pt",
    #     output_dir="/home/infres/yrothlin-24/CHAL_IM05/generated_wbc/PLY",
    #     n_images=200,
    #     z_dim=256,
    # )