import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from typing import List
import os
import sys
import contextlib
import cv2

@contextlib.contextmanager
def _filter_stderr(substr: str):
    r_fd, w_fd = os.pipe()
    orig_fd = os.dup(2)
    os.dup2(w_fd, 2)
    os.close(w_fd)
    try:
        yield
    finally:
        os.dup2(orig_fd, 2)
        os.close(orig_fd)
        with os.fdopen(r_fd, "r") as r:
            for line in r:
                if substr not in line:
                    sys.stderr.write(line)

def imread_silent(path: str):
    with _filter_stderr("Corrupt JPEG data"):
        return cv2.imread(path, cv2.IMREAD_COLOR)


label2id = {
    "SNE": 0,   # Segmented neutrophil
    "LY": 1,    # Lymphocyte
    "MO": 2,    # Monocyte
    "EO": 3,    # Eosinophil
    "BA": 4,    # Basophil
    "VLY": 5,   # Variant lymphocyte
    "BNE": 6,   # Band-form neutrophil
    "MMY": 7,   # Metamyelocyte
    "MY": 8,    # Myelocyte
    "PMY": 9,   # Promyelocyte
    "BL": 10,   # Blast cell
    "PC": 11,   # Plasma cell
    "PLY": 12,  # Prolymphocyte
}



def get_filespath(dataset_dir: str):
    files = []
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                files.append(os.path.join(dirpath, filename))
    return files

def get_labels(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    labels = {}
    for index, row in df.iterrows():
        labels[row['ID']] = label2id[str(row['label'])]
    return labels

    

class IM05_Dataset(Dataset):
    def __init__(self, files: List[str] = None, label_path: str = None, evaluation: bool = False):
        self.files = files
        self.labels = get_labels(label_path) if not evaluation else None
        self.evaluation = evaluation

    def __len__(self):
        return len(self.files)

    
    def _preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = imread_silent(img_path)

        if img is None:
            raise RuntimeError(f"Image illisible : {img_path}")

        img = self._preprocess_image(img)
        filename = os.path.basename(img_path)
        if self.evaluation:
            return img, filename
        label = self.labels[filename] if not self.evaluation else None
        return img, label

    


def split_files(files: List[str], train_ratio: float = 0.8, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(files)
    total_files = len(files)
    train_size = int(total_files * train_ratio)
    train_files = files[:train_size]
    val_files = files[train_size:]
    return train_files, val_files


def get_loaders(
    dataset_dir: str,
    label_path: str,
    test: bool = True, 
    train_ratio: float = 0.8,
    seed: int = 42,
    batch_size: int = 10,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
):
    files = get_filespath(dataset_dir)
    if not test:
        train_files, val_files = split_files(files, train_ratio, seed)
        train_dataset = IM05_Dataset(train_files, label_path, evaluation = test)
        val_dataset = IM05_Dataset(val_files, label_path)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        return train_loader, val_loader
    else:
        dataset = IM05_Dataset(files, label_path, evaluation= test)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )     
        return dataloader



if __name__ == "__main__":
    dataset_dir = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/train"
    label_path = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/train_metadata.csv"
    train_loader, val_loader = get_loaders(dataset_dir, label_path, train_ratio = 1.0, test=False)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    y = {}
    for imgs, labels in train_loader:
        for label in labels:
            label = label.item()
            if label not in y:
                y[label] = 0
            y[label] += 1
    print(f"Train distribution : {y}")
    y = {}
    for imgs, labels in val_loader:
        for label in labels:
            label = label.item()
            if label not in y:
                y[label] = 0
            y[label] += 1
    print(f"Validation distribution : {y}")

    dataset_dir = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/test"
    label_path = "/home/infres/yrothlin-24/CHAL_IM05/data/IMA205-challenge/test_metadata.csv"
    test_loader = get_loaders(dataset_dir, label_path, train_ratio = 1.0, test = True)
    print(f"Test dataset size: {len(test_loader.dataset)}")


