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
help_message="Train an acoustic model from an experiment matrix

Usage: $0 [options] (<trial-dir> | <matrix-file> <line>)
e.g.: $0 exp/matrix 1

This script wraps 'train-acoustic-model' for easy use with an experiment
matrix generated by stepsext/generate_matrix.sh

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
  "log_prior"
  "train_data"
  "dev_data"
  "test_data"
  "model_cfg"
  "state_dir"
  "state_csv"
  "scoring_dir"
  "weigh_training_samples"
  "decoding_states"
)
unset_variables "${trial_dir_vars[@]}" weight_file
. "${trial_dir}/variables"
check_variables_are_set "${trial_dir_vars[@]}"
if $weigh_training_samples ; then
  check_variables_are_set weight_file
else
  # unset it so we know not to use it
  weight_file=
fi

if [ -z "${device}" ]; then
  if python -c 'import sys; import torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    device=cuda
  else
    device=cpu
  fi
  echo "Inferred device: ${device}"
fi

train-acoustic-model \
  --config "${model_cfg}" \
  --device "${device}" \
  --state-csv "${state_csv}" \
  ${weight_file:+--weight-tensor-file "${weight_file}"} \
  "${state_dir}" "${train_data}" "${dev_data}"
