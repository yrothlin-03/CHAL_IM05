import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image


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


def filter_class_files(
    data_dir: str,
    labels_csv: str,
    target_class: str,
    use_augmented: bool = True,
    unique_base_only: bool = False,
) -> List[Path]:
    labels = get_labels(labels_csv)
    all_files = get_filespath(data_dir)

    selected_files: List[Path] = []
    seen_base_ids = set()

    for f in all_files:
        f = Path(f)
        fname = f.name
        base_id = get_base_id(fname)

        if base_id not in labels:
            continue
        if labels[base_id] != target_class:
            continue
        if not use_augmented and fname != base_id:
            continue
        if unique_base_only:
            if base_id in seen_base_ids:
                continue
            seen_base_ids.add(base_id)

        selected_files.append(f)

    if len(selected_files) == 0:
        raise ValueError(f"No files found for class={target_class}")

    return selected_files


def export_stylegan_dataset(
    data_dir: str,
    labels_csv: str,
    target_class: str,
    export_dir: str,
    image_size: int = 256,
    use_augmented: bool = True,
    unique_base_only: bool = False,
) -> Path:
    export_dir = Path(export_dir)
    raw_dir = export_dir / target_class / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = filter_class_files(
        data_dir=data_dir,
        labels_csv=labels_csv,
        target_class=target_class,
        use_augmented=use_augmented,
        unique_base_only=unique_base_only,
    )

    for i, src in enumerate(files):
        img = Image.open(src).convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        dst = raw_dir / f"{i:06d}.png"
        img.save(dst)

    return raw_dir


def build_stylegan_zip(
    stylegan_repo: str,
    source_image_dir: str,
    dest_zip: str,
) -> None:
    stylegan_repo = Path(stylegan_repo)
    cmd = [
        "python",
        str(stylegan_repo / "dataset_tool.py"),
        f"--source={source_image_dir}",
        f"--dest={dest_zip}",
    ]
    subprocess.run(cmd, check=True)


def train_stylegan2_ada(
    stylegan_repo: str,
    dataset_zip: str,
    run_dir: str,
    gpus: int = 1,
    kimg: int = 2000,
    snap: int = 10,
    resume: str | None = None,
    cfg: str = "auto",
    aug: str = "ada",
    mirror: bool = True,
) -> None:
    stylegan_repo = Path(stylegan_repo)
    cmd = [
        "python",
        str(stylegan_repo / "train.py"),
        f"--outdir={run_dir}",
        f"--data={dataset_zip}",
        f"--gpus={gpus}",
        f"--cfg={cfg}",
        f"--aug={aug}",
        f"--kimg={kimg}",
        f"--snap={snap}",
    ]

    if mirror:
        cmd.append("--mirror=1")
    else:
        cmd.append("--mirror=0")

    if resume is not None:
        cmd.append(f"--resume={resume}")

    subprocess.run(cmd, check=True)


def generate_with_stylegan2(
    stylegan_repo: str,
    network_pkl: str,
    outdir: str,
    seeds: str = "0-99",
    trunc: float = 1.0,
) -> None:
    stylegan_repo = Path(stylegan_repo)
    cmd = [
        "python",
        str(stylegan_repo / "generate.py"),
        f"--network={network_pkl}",
        f"--outdir={outdir}",
        f"--seeds={seeds}",
        f"--trunc={trunc}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    DATA_DIR = "/home/infres/yrothlin-24/CHAL_IM05/IMA205-challenge_resampled/train"
    LABELS_CSV = "/home/infres/yrothlin-24/CHAL_IM05/IMA205-challenge_resampled/train_metadata.csv"

    STYLEGAN_REPO = "/home/infres/yrothlin-24/stylegan2-ada-pytorch"
    WORK_DIR = "/home/infres/yrothlin-24/CHAL_IM05/stylegan_wbc"
    TARGET_CLASS = "PLY"

    IMAGE_SIZE = 256
    USE_AUGMENTED = True
    UNIQUE_BASE_ONLY = False

    PREPARE_DATASET = True
    TRAIN = True
    GENERATE = False

    raw_dir = Path(WORK_DIR) / TARGET_CLASS / "images"
    dataset_zip = Path(WORK_DIR) / TARGET_CLASS / f"{TARGET_CLASS}.zip"
    run_dir = Path(WORK_DIR) / TARGET_CLASS / "training_runs"
    gen_dir = Path(WORK_DIR) / TARGET_CLASS / "generated"

    if PREPARE_DATASET:
        export_stylegan_dataset(
            data_dir=DATA_DIR,
            labels_csv=LABELS_CSV,
            target_class=TARGET_CLASS,
            export_dir=WORK_DIR,
            image_size=IMAGE_SIZE,
            use_augmented=USE_AUGMENTED,
            unique_base_only=UNIQUE_BASE_ONLY,
        )

        build_stylegan_zip(
            stylegan_repo=STYLEGAN_REPO,
            source_image_dir=str(raw_dir),
            dest_zip=str(dataset_zip),
        )

    if TRAIN:
        train_stylegan2_ada(
            stylegan_repo=STYLEGAN_REPO,
            dataset_zip=str(dataset_zip),
            run_dir=str(run_dir),
            gpus=1,
            kimg=2000,
            snap=10,
            resume=None,
            cfg="auto",
            aug="ada",
            mirror=True,
        )

    if GENERATE:
        network_pkl = str(run_dir / "00000" / "network-snapshot-000200.pkl")
        generate_with_stylegan2(
            stylegan_repo=STYLEGAN_REPO,
            network_pkl=network_pkl,
            outdir=str(gen_dir),
            seeds="0-199",
            trunc=1.0,
        )