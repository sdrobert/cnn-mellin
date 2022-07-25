#! /usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mem=25G
#SBATCH --output=logs/timit/stage04.%J.log
#SBATCH --time=infinite
#SBATCH --array=1-20

./scripts/slurm/timit_wrapper.sh -s 04 -x -q
