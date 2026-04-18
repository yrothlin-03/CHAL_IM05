#!/bin/bash
#SBATCH -p A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --job-name=convnext_tiny_without_contrastive_pretraining
#SBATCH --output=logs/IM05/%x_%j.out
#SBATCH --error=logs/IM05/%x_%j.err

set -euo pipefail
mkdir -p logs/IM05/
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

eval "$(micromamba shell hook --shell=bash)"
micromamba activate IM05

cd /home/infres/yrothlin-24/CHAL_IM05

# python -m deep_learning.main

python -m deep_learning.main \
  --evaluation false \
  --submission_path /home/infres/yrothlin-24/CHAL_IM05/submissions/submission.csv \
  --training.save_ckpt true \
  --training.ckpt_dir /home/infres/yrothlin-24/CHAL_IM05/checkpoints/tests \
  --training.ckpt_every 1 \
  --training.resume_from /home/infres/yrothlin-24/CHAL_IM05/checkpoints/tests/best.pt \
  --training.resume true \
  --training.resume_from_pretrained /home/infres/yrothlin-24/CHAL_IM05/checkpoints/contrastive_pretraining/best.pt \
  --training.resume_pretrained false \
  --training.monitor macro_f1 \
  --training.early_stopping true \
  --training.early_stopping_patience 10 \
  --training.early_stopping_mode max \
  --training.use_mixup true \
  --training.mixup_alpha 0.2 \
  --training.use_cutmix true \
  --training.cutmix_alpha 1.0 \
  --training.cutmix_prob 0.5 \
  --training.loss focal \
  --training.gamma 2.0 \
  --training.bb_lr 0.0003 \
  --training.head_lr 0.0003 \
  --training.finetune_bb_lr 0.0001 \
  --training.finetune_head_lr 0.0001 \
  --training.weight_decay 0.0001 \
  --training.num_epochs 60 \
  --training.warmup_epochs 0 \
  --training.optimizer adamw \
  --training.scheduler cosine_restarts \
  --training.use_amp false \
  --training.grad_clip 1.0 \
  --training.log_step 50 \
  --training.tta_rounds 4 \
  --training.label_smoothing 0.05 \
  --training.use_weights false \
  --training.weights 0.131885 0.167166 0.287122 0.512761 0.738572 0.786459 0.760902 0.792986 0.716469 1.409173 0.335431 1.824578 4.536496 \
  --training.plot_tsne true \
  --training.tsne_split val \
  --training.tsne_every 5 \
  --training.tsne_max_samples 1000 \
  --training.tsne_perplexity 30 \
  --training.tsne_dir /home/infres/yrothlin-24/CHAL_IM05/logs/tests/TSNE \
  --training.plot_confusion true \
  --training.confusion_split val \
  --training.confusion_every 5 \
  --training.confusion_normalize true \
  --training.confusion_dir /home/infres/yrothlin-24/CHAL_IM05/logs/tests/confusion \
  --training.class_names SNE LY MO EO BA VLY BNE MMY MY PMY BL PC PLY \
  --data.train_ratio 0.9 \
  --data.seed 42 \
  --data.batch_size 64 \
  --data.shuffle true \
  --data.num_workers 4 \
  --data.pin_memory true \
  --data.n_splits 1 \
  --data.use_weighted_sampler true \
  --data.use_undersampling false \
  --data.sampler_power 0.5 \
  --data.with_tta true \
  --model.num_classes 13 \
  --model.backbone_name CONVNEXT_TINY \
  --model.freeze_backbone false \
  --model.pretrained true