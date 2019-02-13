#! /usr/bin/env bash
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=optimize_am
set -e

stepsext/optimize_acoustic_model.sh \
  --verbose true \
  --device cuda \
  exp/matrix ${SLURM_ARRAY_TASK_ID:-1}
