import torch
import torch.nn as nn
from typing import Dict, Any, List
import os
import pandas as pd
from pathlib import Path
import shutil
from torchvision import transforms
import numpy as np
import cv2
import contextlib



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

])

transform2 = transforms.Compose([ 
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(180, fill=128),
    transforms.ToTensor(),
])


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

label2id = {v: k for k, v in id2label.items()}





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



def get_filespath(dataset_dir: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith(".png"):
                files.append(os.path.join(dirpath, filename))
    files.sort()
    return files


def get_labels(csv_path: str) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        labels[str(row["ID"])] = label2id[str(row["label"])]
    return labels


def get_class_distribution() -> Dict[int, List[str]]:
    distribution = {i: [] for i in range(13)}
    for filename, label in labels.items():
        distribution[label].append(filename)
    return distribution

def save_new_image(label: int, augmented_image, out_dir: str):
    label_name = id2label[label]
    label_dir = os.path.join(out_dir, label_name)
    os.makedirs(label_dir, exist_ok=True)

    existing_files = os.listdir(label_dir)
    existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.startswith(label_name)]
    next_index = max(existing_indices) + 1 if existing_indices else 0

    new_filename = f"{label_name}_{next_index}.png"
    new_filepath = os.path.join(label_dir, new_filename)

    augmented_image.save(new_filepath)
    print(f"Saved augmented image: {new_filepath}")


def class_initial_images(data_dir: str, label_path: str, out_dir: str):
    labels = get_labels(label_path)
    distribution = get_class_distribution()

    for label, file_list in distribution.items():
        label_dir = os.path.join(out_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        for filepath in file_list:
            # print(f"Processing file: {filepath} for label: {id2label[label]}")
            src_path = filepath if os.path.isabs(filepath) else os.path.join(data_dir, filepath)
            filename = os.path.basename(filepath)
            dst_path = os.path.join(label_dir, filename)

            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Missing source file: {src_path} (from {filepath})")

            shutil.copy2(src_path, dst_path)



def extract_wbc_crop2(
    img_rgb: np.ndarray,
    pad: int = 20,
    min_area_frac: float = 0.0005,
    q: float = 0.92,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    hdist = np.minimum(np.abs(H - 140.0), 180.0 - np.abs(H - 140.0)) / 90.0
    hscore = 1.0 - np.clip(hdist, 0.0, 1.0)
    sscore = np.clip((S - 30.0) / 225.0, 0.0, 1.0)
    vscore = 1.0 - np.clip(V / 255.0, 0.0, 1.0)

    score = 0.55 * hscore + 0.25 * sscore + 0.20 * vscore
    thr = float(np.quantile(score, q))
    nuc = (score >= thr).astype(np.uint8)

    nuc = cv2.medianBlur(nuc * 255, 5)
    nuc = (nuc > 0).astype(np.uint8)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_OPEN, k1, iterations=1)
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_CLOSE, k1, iterations=2)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(nuc.astype(np.uint8), connectivity=8)
    if num <= 1:
        return img_rgb

    h, w = nuc.shape
    min_area = int(min_area_frac * h * w)

    cx0, cy0 = w * 0.5, h * 0.5
    best_i = None
    best_score = -1e18
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx = float(stats[i, cv2.CC_STAT_LEFT] + 0.5 * stats[i, cv2.CC_STAT_WIDTH])
        cy = float(stats[i, cv2.CC_STAT_TOP] + 0.5 * stats[i, cv2.CC_STAT_HEIGHT])
        d2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
        s = np.log1p(area) - 0.0008 * d2
        if s > best_score:
            best_score = s
            best_i = i

    if best_i is None:
        return img_rgb

    nuc = (lab == best_i).astype(np.uint8)

    kbig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    mask = cv2.dilate(nuc, kbig, iterations=1)

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return img_rgb

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_rgb.shape[1], x2 + pad)
    y2 = min(img_rgb.shape[0], y2 + pad)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return img_rgb

    return img_rgb[y1:y2, x1:x2]

    
def preprocess_original_data(
    data_dir: str,
    out_dir: str,
    label_path: str,
    clear_output_dir: bool = True
):
    if clear_output_dir and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    labels = get_labels(label_path)
    count = 0

    for filename, label in labels.items():
        src_path = os.path.join(data_dir, filename)

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source file: {src_path}")

        img = imread_silent(src_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_img = extract_wbc_crop2(img_rgb)
        transformed_img = transform(cropped_img)

        img_np = (transformed_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        label_name = id2label[label]
        label_dir = os.path.join(out_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        dst_path = os.path.join(label_dir, os.path.basename(filename))
        cv2.imwrite(dst_path, img_bgr)

        # print(f"Preprocessed and saved: {dst_path}")

        # count += 1
        # if count >= 5:
        #     break



def create_new_data(
    filepath: str, 
):
    img = imread_silent(filepath)
    if img is None:
        print(f"Warning: Could not read image {filepath}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented_img = transform2(img_rgb)
    augmented_img_np = (augmented_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    augmented_img_bgr = cv2.cvtColor(augmented_img_np, cv2.COLOR_RGB2BGR)
    return augmented_img_bgr


def augment_data_classical(
    data_dir: str = "/home/infres/yrothlin-24/CHAL_IM05/data/my_augmented_data",
    num_augmented_per_image: int = 5,
    class_to_augment = ['PLY', 'PC', 'PMY'],
):
    for cls in class_to_augment:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        images = [f for f in os.listdir(cls_dir) if f.endswith(".png")]

        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            img = imread_silent(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stem = Path(img_name).stem

            existing = [f for f in os.listdir(cls_dir) if f.startswith(stem + "_aug")]
            start_idx = len(existing)

            for i in range(num_augmented_per_image):
                aug = transform2(img_rgb)
                aug_np = (aug.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                aug_bgr = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)

                out_name = f"{stem}_aug{start_idx + i}.png"
                out_path = os.path.join(cls_dir, out_name)
                cv2.imwrite(out_path, aug_bgr)
    


    

if __name__ == "__main__":
    dataset_dir = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train"
    label_path = "/tsi/data_education/ChallengeIMA205/IMA205-challenge/train_metadata.csv"
    out_dir = "/home/infres/yrothlin-24/CHAL_IM05/data/my_augmented_data"

    files = get_filespath(dataset_dir)
    labels = get_labels(label_path)
    # distribution = get_class_distribution()
    # print("Class distribution:", {id2label[i]: len(files) for i, files in distribution.items()})
    # class_initial_images(dataset_dir, label_path, out_dir)
    # preprocess_original_data(dataset_dir, out_dir, label_path)
    augment_data_classical(out_dir, num_augmented_per_image=5, class_to_augment=['PC', 'PMY'])