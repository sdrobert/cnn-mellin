#! /usr/bin/env bash

function check_variables_are_set() {
  while [ $# -gt 0 ]; do
    if [[ ! -v "$1" ]]; then
      return 1
    fi
    shift
  done
}

function unset_variables() {
  while [ $# -gt 0 ]; do
    unset $1
    shift
  done
}

[ -f path.sh ] && . path.sh

echo "$0 $*"

device=
train_num_data_workers=
verbose=false
help_message="Optimize acoustic model parameters from an experiment matrix

Usage: $0 [options] (<trial-dir> | <matrix-file> <line>)
e.g.: $0 exp/matrix 1

This script wraps 'optimize-acoustic-model' for easy use with an experiment
matrix generated by stepsext/generate_matrix.sh. The optimized configuration
will be written to the 'optim_config' variable in the 'variables' file of the
trial directory. For convenience, the path to this file will be printed upon
completion of the script. This trial (and any other trial in the matrix file)
will not use the optimized config file; it is up to the user to re-generate
matrices with the updated config.

Where <trial-dir> is a directory generated by stepsext/generate_matrix.sh and
contains a file called 'variables'.

Alternatively, <matrix-file> is the matrix file generated by
stepsext/generate_matrix.sh and <line> is a natural number that indexes
a line in <matrix-file> that lists <trial-dir>.

Options:
--device <STR>                 : A torch device string, such as 'cpu' or
                                 'cuda:1'. If unset, 'cuda' will be used if
                                 pytorch can access a GPU, otherwise 'cpu'
--train-num-data-workers <INT> : The number of worker threads to spawn to
                                 handle loading training data. If unset, will
                                 use the train-acoustic-model default
--verbose {true,false}
"
. parse_options.sh

if [ $# != 1 ] && [ $# != 2 ]; then
  echo "${help_message}" | grep "Usage" 1>&2
  echo "$0 --help for more info" 1>&2
  exit 1
fi

set -e

if [ $# = 1 ]; then
  trial_dir="$1"
else
  trial_dir="$(sed "${2}q;d" "$1")"
fi

if [ ! -f "${trial_dir}/variables" ]; then
  echo "No file '${trial_dir}/variables'" 1>&2
  exit 1
fi

trial_dir_vars=(
  "freq_dim"
  "target_dim"
  "HCLG"
  "gmm_mdl"
  "words"
  "log_prior"
  "train_data"
  "dev_data"
  "test_data"
  "dev_ref"
  "test_ref"
  "model_cfg"
  "state_dir"
  "state_csv"
  "decode_dev"
  "decode_test"
  "decoding_states"
  "min_active"
  "max_active"
  "max_mem"
  "beam"
  "lattice_beam"
)
optim_variables=(
  "optim_out_config"
  "optim_history_url"
  "optim_partitions"
  "optim_use_val_partition"
  "optim_data_set"
  "optim_agent"
)
unset_variables "${trial_dir_vars[@]}" "${optim_variables[@]}" "weight_file"
. "${trial_dir}/variables"
check_variables_are_set "${trial_dir_vars[@]}"
if ! check_variables_are_set "${optim_variables[@]}" ; then
  echo "\
Unable to find optimization variables (such as ${optim_variables[0]}) in
'${trial_dir}/variables'. This likely means you called
stepsext/generate_matrix.sh without first having installed optuna.
Try 'pip install optuna'" 1>&2
  exit 1
fi

if [ -z "${device}" ]; then
  if python -c 'import sys; import torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    device=cuda
  else
    device=cpu
  fi
  echo "Inferred device: ${device}"
fi

if ! $use_optim_val_partition; then
  no_optim_val_partition=x
fi

data_dir="$(eval "echo \${${optim_data_set}_data}")"

if $verbose ; then
  verbose=--verbose
else
  verbose=
fi

optimize-acoustic-model \
  --config "${model_cfg}" \
  --device "${device}" \
  --history-url "${optim_history_url}" \
  --agent-name "${optim_agent}" \
  ${verbose} \
  ${no_optim_val_partition+--no-val-partition} \
  ${train_num_data_workers:+--train-num-data-workers "${train_num_data_workers}"} \
  ${weight_file:+--weight-tensor-file "${weight_file}"} \
  "${data_dir}" "${optim_partitions[@]}" "${optim_out_config}"

echo "Optimized configuration stored at '${optim_out_config}'"
