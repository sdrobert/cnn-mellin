#! /usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --export=ALL
#SBATCH --mem=4G
#SBATCH --output=logs/timit/stage02-%a.log
#SBATCH --time=1:00:00
#SBATCH --array=1-2

./scripts/slurm/timit_wrapper.sh -s 02 -x -w
