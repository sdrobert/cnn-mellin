#! /usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output=logs/timit/slurm-%J.log

global_args=( )

if [ -d "/checkpoint/${USER}/${SLURM_JOB_ID}" ]; then
  export TIMIT_CKPT_DIR="/checkpoint/${USER}/${SLURM_JOB_ID}/ckpt"
  export TORCH_EXTENSIONS_DIR="/checkpoint/${USER}/${SLURM_JOB_ID}/torch_extensions"
  mkdir -p "$TIMIT_CKPT_DIR" "$TORCH_EXTENSIONS_DIR"
  global_args+=( "-O" "$TIMIT_CKPT_DIR" )
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

./timit.sh -q -x "$@" "${global_args[@]}"
