#! /usr/bin/env bash

mkdir -p logs/timit

export CUDA_HOME=/pkgs/cuda-11.3
timit_dir="${1:"~/Databases/TIMIT"}"
cpu_opts="${2:-"-p cpu"}"
gpu_opts="${3:-"-p p100"}"
do_hyperparam=1

set -e

python -c 'import asr'

[ ! -f "data/timit/local/.complete" ] && \
  sbatch $cpu_opts -c 1 -W --mem=1G \
    scripts/slurm/timit_wrapper.sh -s 1 -i "$timit_dir"

[ ! -f "data/timit/fbank-81-10ms/.complete" ] || \
[ ! -f "data/timit/sigbank-41-2ms/.complete" ] && \
  sbatch $cpu_opts -c 1 -a 1-2 -W --mem=5G -t 3:0:0 \
    scripts/slurm/timit_wrapper.sh -s 2

# hyperparameter search (can be skipped)
if ((do_hyperparam)); then
  # XXX(sdrobert): the even-numbered stages are the actual hyperparameter
  # searches. We assign them an infinite time allotment by default as it's
  # unclear how long they'll take. You can safely decrease these allotments to
  # a few hours and, whenever the job is killed, simply restart it.
  for s in {3..13..2}; do
    Sa="$(printf "%02d\n" $s)"
    Sb="$(printf "%02d\n" $((s + 1)))"
    if [ ! -f "exp/timit/completed_stages/$Sa" ]; then
      sbatch $cpu_opts --mem=256M -c 1 -W -t 0:10:0 \
        scripts/slurm/timit_wrapper.sh -s $s
      touch "exp/timit/completed_stages/$Sa"
    fi
    if [ ! -f "exp/timit/completed_stages/$Sb" ]; then
      sbatch $gpu_opts -c 4 -a 1-64 -W --gres=gpu:1 --mem=25G \
        scripts/slurm/timit_wrapper.sh -s $((s + 1))
      touch "exp/timit/completed_stages/$Sb"
    fi
  done
fi