#! /usr/bin/env bash

mkdir -p logs/timit

export CUDA_HOME=/pkgs/cuda-11.3
timit_dir=~/Databases/TIMIT
cpu_opts="-p cpu --gres=gpu:0"
gpu_opts="-p p100 --gres=gpu:1"
do_hyperparam=1

set -e

python -c 'import asr'

[ ! -f "data/timit/local/.complete" ] && \
  sbatch $cpu_opts -c 1 -W scripts/slurm/timit_wrapper.sh -s 1 -i "$timit_dir"

[ ! -f "data/timit/fbank-81-10ms/.complete" ] || \
[ ! -f "data/timit/sigbank-41-2ms/.complete" ] && \
  sbatch $cpu_opts -c 1 -a 1-2 -W scripts/slurm/timit_wrapper.sh -s 2

# hyperparameter search (can be skipped)
if ((do_hyperparam)); then
  # XXX(sdrobert): the even-numbered stages are the actual hyperparameter
  # searches. We assign them an infinite time allotment by default as it's
  # unclear how long they'll take. You can safely decrease these allotments to
  # a few hours and, whenever the job is killed, simply restart it. Note that
  # doing so 
  for s in {3..13..2}; do
    Sa="$(printf "%02d\n" $s)"
    Sb="$(printf "%02d\n" $((s + 1)))"
    [ ! -f "exp/timit/optim/completed_stages/$Sa" ] && \
      sbatch $cpu_opts -c 1 -W scripts/slurm/timit_wrapper.sh -s $s
    [ ! -f "exp/timit/optim/completed_stages/$Sb" ] && \
      sbatch $gpu_opts -c 4 -a 1-20 -W scripts/slurm/timit_wrapper.sh -s $s
  done
fi