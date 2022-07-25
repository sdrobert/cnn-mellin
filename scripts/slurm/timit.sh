#! /usr/bin/env bash

mkdir -p logs/timit

export CUDA_HOME=/pkgs/cuda-11.3
cpu_opts="-p cpu"
gpu_opts="-p p100"

set -e

[ ! -f "data/timit/local/.complete" ] && \
  sbatch $cpu_opts --wait scripts/slurm/timit_stage_01.sh ~/Databases/TIMIT

[ ! -f "data/timit/fbank-81-10ms/.complete" ] || \
[ ! -f "data/timit/sigbank-41-2ms/.complete" ] && \
  sbatch $cpu_opts --wait scripts/slurm/timit_stage_02.sh

# hyperparameter search (can be skipped)
if true; then
  # XXX(sdrobert): the even-numbered stages are the actual hyperparameter
  # searches. We assign them an infinite time allotment by default as it's
  # unclear how long they'll take. You can safely decrease these allotments to
  # a few hours and, whenever the job is killed, simply restart it. Note that
  # doing so 
  for s in {3..13..2}; do
    Sa="$(printf "%02d\n" $s)"
    Sb="$(printf "%02d\n" $((s + 1)))"
    [ ! -f "exp/timit/optim/completed_stages/$Sa" ] && \
      sbatch $cpu_opts --wait scripts/slurm/timit_stage_$Sa.sh
    [ ! -f "exp/timit/optim/completed_stages/$Sb" ] && \
      sbatch $gpu_opts --wait scripts/slurm/timit_stage_$Sb.sh
  done
fi