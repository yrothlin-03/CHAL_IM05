#!/bin/bash
#SBATCH -p A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --job-name=efficientNet
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

python -m deep_learning.main