# IM05 - Classification d’images médicales - Challenge 2025/2026

---

## Environnement

Les expériences ont été réalisées avec un environnement **micromamba** sur cluster.

- Avec **conda / micromamba** :
```bash
conda env create -f environment.yml
conda activate im05
```

- Avec **pip** :
```bash
pip install -r requirements.txt
```

---

## Données et prétraitement

### Dataset

Les chemins vers les données doivent être renseignés manuellement dans les fichiers de configuration.

Structure attendue :
```text
train/
train_metadata.csv
test/
test_metadata.csv
```

Mapping des labels :
```python
label2id = {
    "SNE": 0, "LY": 1, "MO": 2, "EO": 3, "BA": 4,
    "VLY": 5, "BNE": 6, "MMY": 7, "MY": 8, "PMY": 9,
    "BL": 10, "PC": 11, "PLY": 12,
}
```

---

## Première approche : Deep Learning

Le pipeline deep learning est situé dans :
```text
deep_learning/
```

### Pré-entraînement contrastif supervisé

```bash
python -m deep_learning.main_contrastive
```

### Fine-tuning pour la classification

```bash
python -m deep_learning.main
```

### Configuration requise (IMPORTANT)

Avant de lancer les scripts, il est nécessaire de modifier les fichiers YAML dans :
```text
configs/
```

#### Champs obligatoires

```yaml
data:
  dataset_dir: "/path/to/train"
  label_path: "/path/to/train_metadata.csv"

training:
  ckpt_dir: "/path/to/checkpoints"
  tsne_dir: "/path/to/logs_tsne"

submission_path: "/path/to/submission.csv"
```

### Paramètres principaux

- Optimiseur : AdamW / SGD
- Scheduler : cosine / cosine_restarts
- Loss : CrossEntropy / Focal Loss
- Data augmentation : flips, rotations, transformations affines, color jitter, random erasing
- Régularisation : label smoothing, weight decay
- Méthodes avancées : MixUp, CutMix, TTA

---

## Seconde approche : Machine Learning classique et vision par ordinateur

Notebook principal :
```text
machine_learning/pipeline_ml.ipynb
```

### Paramètres obligatoires

```python
OUT_DIR = "/path/to/output"

train_dir = "/path/to/train"
train_label_path = "/path/to/train_metadata.csv"

test_dir = "/path/to/test"
test_label_path = "/path/to/test_metadata.csv"
```

### Pipeline

- Prétraitement (amélioration du contraste, redimensionnement)
- Segmentation des globules blancs
- Extraction de caractéristiques (texture, couleur, deep features)
- Sélection de variables
- Classification (Logistic Regression, SVM, Random Forest, XGBoost)

---

## Scripts de prétraitement et d’augmentation

Situés dans :
```text
preprocessing/
```

Scripts principaux :

- `oversampling_rare_classes.py` → augmentation offline
- `gan.py` → génération d’images (WGAN-GP)
- `prepare_data_stylegan.py` → préparation StyleGAN (expérimental)

⚠️ Les chemins doivent être adaptés manuellement.

---

## Remarques

- Les performances sont sensibles à :
  - la séparation train/val/test
  - la seed aléatoire
  - le déséquilibre des classes

- L’entraînement est coûteux en calcul (GPU recommandé)

- Vérifier systématiquement :
  - les chemins des données
  - les chemins des checkpoints
  - la cohérence des fichiers YAML