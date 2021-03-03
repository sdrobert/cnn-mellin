#! /usr/bin/env bash
#SBATCH --output=/dev/null
#SBATCH --error=matrix_%a.err
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=decode_am

set -e

if [ ! -d exp/logs ]; then
  mkdir -p exp/logs
fi

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "Needs array" 1>&2
  exit 1
fi

echo "What is happening?"

trial_dir="$(sed "${SLURM_ARRAY_TASK_ID}q;d" exp/matrix)"
# name and trial num
trial_name=$(basename "$(dirname "${trial_dir}")")_$(basename "${trial_dir}")

echo "${trial_dir} ${trial_name}"

stepsext/decode_acoustic_model.sh \
  --device cuda \
  "${trial_dir}" &>> "exp/logs/decode_${trial_name}.log"
