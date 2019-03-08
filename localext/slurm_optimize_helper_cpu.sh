#! /usr/bin/env bash
#SBATCH --output=/dev/null
#SBATCH --error=matrix_%a.err
#SBATCH --open-mode=append
#SBATCH --mem=20G
#SBATCH --cpus-per-task=20
#SBATCH --job-name=opt_am

set -e

if [ ! -d exp/logs ]; then
  mkdir -p exp/logs
fi

# this helps ensure not everyone is querying/creating the database at the same
# time
sleep ${SLURM_ARRAY_TASK_ID:-1}

trial_dir="$(sed "${SLURM_ARRAY_TASK_ID:-1}q;d" exp/matrix)"
# name and trial num
trial_name=$(basename "$(dirname "${trial_dir}")")_$(basename "${trial_dir}")

stepsext/optimize_acoustic_model.sh \
  --verbose true \
  --device cpu \
  "${trial_dir}" &>> "exp/logs/${trial_name}.log"
