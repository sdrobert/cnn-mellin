#! /usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mem=25G
#SBATCH --output=logs/timit/stage12.%J.log
#SBATCH --time=infinite
#SBATCH --array=1-20

./scripts/slurm/timit_wrapper.sh -s 12 -x -q
