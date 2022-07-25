#! /usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --export=ALL
#SBATCH --mem=4G
#SBATCH --output=logs/timit/stage01.log
#SBATCH --time=0:10:00

./scripts/slurm/timit_wrapper.sh -s 01 -i "$1" -x -w
