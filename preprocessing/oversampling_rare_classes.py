import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

import sys
import contextlib


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


LABEL2ID = {
    "SNE": 0, "LY": 1, "MO": 2, "EO": 3, "BA": 4, "VLY": 5, "BNE": 6,
    "MMY": 7, "MY": 8, "PMY": 9, "BL": 10, "PC": 11, "PLY": 12,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def build_augmenter():
    def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = img.copy()

        if rng.random() < 0.5:
            out = cv2.flip(out, 1)

        if rng.random() < 0.5:
            out = cv2.flip(out, 0)

        angle = float(rng.uniform(-20, 20))
        h, w = out.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)

        tx = float(rng.uniform(-0.08, 0.08) * w)
        ty = float(rng.uniform(-0.08, 0.08) * h)
        rot[0, 2] += tx
        rot[1, 2] += ty

        out = cv2.warpAffine(
            out,
            rot,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        alpha = float(rng.uniform(0.9, 1.1))
        beta = float(rng.uniform(-10, 10))
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        if rng.random() < 0.3:
            noise = rng.normal(0, 4, size=out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return out

    return augment


def resample_and_export(
    src_train_dir: str,
    src_labels_csv: str,
    dst_root_dir: str,
    class_target_counts: Dict[str, int],
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    augment = build_augmenter()

    src_train_dir = Path(src_train_dir)
    src_labels_csv = Path(src_labels_csv)
    dst_root_dir = Path(dst_root_dir)

    dst_train_dir = dst_root_dir / "train"
    dst_labels_csv = dst_root_dir / "train_metadata.csv"

    dst_train_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_labels_csv)
    if "ID" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns 'ID' and 'label'.")

    grouped: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        label = str(row["label"])
        img_id = str(row["ID"])
        grouped.setdefault(label, []).append(img_id)

    rows: List[Dict[str, str]] = []
    processed = 0

    all_labels = sorted(grouped.keys())

    for label in all_labels:
        original_ids = grouped[label]
        current_count = len(original_ids)
        target_count = class_target_counts.get(label, current_count)

        if target_count <= 0:
            print(f"[INFO] Class {label}: target_count={target_count}, skipped.")
            continue

        print(f"[INFO] Class {label}: current={current_count}, target={target_count}")

        if target_count < current_count:
            kept_idx = rng.choice(current_count, size=target_count, replace=False)
            selected_ids = [original_ids[i] for i in kept_idx]

            for img_id in selected_ids:
                src_img = src_train_dir / img_id
                dst_img = dst_train_dir / img_id

                img = imread_silent(str(src_img))
                if img is None:
                    raise RuntimeError(src_img)

                ok = cv2.imwrite(str(dst_img), img)
                if not ok:
                    raise RuntimeError(dst_img)

                rows.append({"ID": img_id, "label": label})

                processed += 1
                if processed % 500 == 0:
                    print(f"[INFO] Processed {processed} files")

        else:
            for img_id in original_ids:
                src_img = src_train_dir / img_id
                dst_img = dst_train_dir / img_id

                img = imread_silent(str(src_img))
                if img is None:
                    raise RuntimeError(src_img)

                ok = cv2.imwrite(str(dst_img), img)
                if not ok:
                    raise RuntimeError(dst_img)

                rows.append({"ID": img_id, "label": label})

                processed += 1
                if processed % 500 == 0:
                    print(f"[INFO] Processed {processed} files")

            n_to_generate = target_count - current_count

            for k in range(n_to_generate):
                src_id = original_ids[rng.integers(0, current_count)]
                src_img_path = src_train_dir / src_id

                img = imread_silent(str(src_img_path))
                if img is None:
                    raise RuntimeError(src_img_path)

                aug_img = augment(img, rng)

                stem = Path(src_id).stem
                suffix = Path(src_id).suffix
                new_id = f"{stem}_aug_{label}_{k:05d}{suffix}"

                dst_img_path = dst_train_dir / new_id

                ok = cv2.imwrite(str(dst_img_path), aug_img)
                if not ok:
                    raise RuntimeError(dst_img_path)

                rows.append({"ID": new_id, "label": label})

                processed += 1
                if processed % 500 == 0:
                    print(f"[INFO] Processed {processed} files")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(dst_labels_csv, index=False)

    print(f"[INFO] Done. Total processed files: {processed}")
    print("[INFO] Final distribution:")
    print(out_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    SRC_TRAIN_DIR = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train"
    SRC_LABELS_CSV = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train_metadata.csv"
    DST_ROOT_DIR = "/home/infres/yrothlin-24/CHAL_IM05/IMA205-challenge_resampled"

    CLASS_TARGET_COUNTS = {
        "SNE": 6000,
        "LY": 5000,
        "MO": 2500,
        "EO": 800,
        "BA": 400,
        "VLY": 400,
        "BNE": 400,
        "MMY": 400,
        "MY": 400,
        "PMY": 300,
        "BL": 1500,
        "PC": 200,
        "PLY": 200,
    }

    resample_and_export(
        src_train_dir=SRC_TRAIN_DIR,
        src_labels_csv=SRC_LABELS_CSV,
        dst_root_dir=DST_ROOT_DIR,
        class_target_counts=CLASS_TARGET_COUNTS,
        seed=42,
    )