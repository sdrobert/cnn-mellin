#! /usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output=logs/timit/slurm-%J.log

# XXX(sdrobert): If TIMIT_CKPT_DIR is set, it will be used to store checkpoints
# of models during hyperparameter tuning. When a trial completes successfully,
# the subdirectory for the trial is removed. If the files are absent (e.g. they
# were cleared), the trial will restart from epoch 1.
# If TIMIT_CKPT_DIR is not set, it will default to the optim/ckpts subdirectory
# of the experiment folder.
if [ -d "/checkpoint/${USER}/${SLURM_JOB_ID}" ]; then
  export TIMIT_CKPT_DIR="/checkpoint/${USER}/${SLURM_JOB_ID}"
fi

stage="$(echo "$*" | sed -n 's/.*-s \([0-9]*\).*/\1/p')"
if [ -z "$stage" ]; then
  echo "Did not specify a stage!" 2>&1
  exit 1
fi

# We use the job array for parallelism. Don't muck with these.
if [ ! -z "${SLURM_ARRAY_TASK_ID}" ]; then
  export TIMIT_OFFSET="$(( ${SLURM_ARRAY_TASK_ID} - 1 ))"
  export TIMIT_STRIDE="${SLURM_ARRAY_TASK_COUNT}"
fi

echo "Starting stage $stage"
./timit.sh -q -x "$@"
if [ $? -eq 0 ]; then
  echo "Succeeded"
  exit 0
else
  echo "Failed"
  exit 1
fi
