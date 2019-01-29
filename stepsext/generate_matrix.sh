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

exp_dir=exp
num_trials=10
decoding_states=best
append_to_matrix=false
min_active=200
max_active=7000
max_mem=50000000
beam=13.0
lattice_beam=8.0
optim_data_set=train
optim_num_partitions=3
optim_use_val_partition=true
help_message="Generate an experiment matrix for cnn-mellin

Usage: $0 [options] <torch-data-group> [<config-group-1> [<config-group-2 [...]]]
e.g.: $0 \\
      data/torch/fbank,data/torch/tonebank \\
      conf/partials/mconv.cfg,conf/partials/conv.cfg

Where a group is a comma-delimited list of paths.

<torch-data-dir> is a directory that contains the file 'variables' which
contains key=value pairs for the variables:

freq_dim    : the number of coefficients per frame in time
target_dim  : the number of emission pdfs (senones) the acoustic model must
              output
HCLG        : path to the test-time decoding graph used by Kaldi
gmm_mdl     : path to the '.mdl' file Kaldi uses as part of the decoding
              process
log_prior   : path to a FloatTensor containing log prior probabilities of each
              target
train_data  : absolute path to (torch) training data directory
dev_data    : absolute path to (torch) validation data directory
test_data   : absolute path to (torch) test data directory
dev_ref     : absolute path to kaldi validation data directory, where 'text'
              and 'stm' files are stored
test_ref    : absolute path to kaldi test data directory, where 'text' and
              'stm' files are stored
weight_file : path to a FloatTensor containing the relative weights of each
              target (optional)

<config-group-x> is a comma-delimited list of files containing partial
configuration files (in .INI format) that are mutually exclusive of one
another. A model configuration for the trial will be created by loading up
the files, one from each group, such that <config-group-x> clobbers
<config-group-y> if <config-group-x> comes before <config-group-y> on the
command line.

Options
--exp-dir <DIR>                   : Root directory of outputs. A file
                                    'exp_dir/matrix' will list trials and all
                                    trials will be in a subdirectory of exp_dir
                                    (deft: ${exp_dir})
--num-trials <INT>                : Total number of trials per configuration to
                                    generate (deft: ${num_trials})
--decoding-states {best,last}     : Whether test-time decoding should use the
                                    best model parameters or last model
                                    parameters at test time
                                    (def: ${decoding_states})
--append-to-matrix <BOOL>         : If a matrix file already exists and true,
                                    will only append the new entries to the
                                    file. If false, the file will be
                                    overwritten (deft: ${append_to_matrix})
--min-active <INT>                : Used in decoding. The minimum number of
                                    active states (deft: ${min_active})
--max-active <INT>                : Used in decoding. The maximum number of
                                    active states (deft: ${max_active})
--max-mem <INT>                   : Used in decoding. The maximum approximate
                                    memory usage in determinization
                                    (deft: ${max_mem})
--beam <FLOAT>                    : Decoding beam width (deft: ${beam})
--lattice-beam <FLOAT>            : Lattice generation beam width
                                    (deft: ${lattice_beam})
--optim-data-set {train,dev,test} : The partition to perform hyperparameter
                                    optimization on (deft: ${optim_data_set})
--optim-partitions <INT>          : The number of partitions to split the data
                                    set for hyperparameter optimization.
                                    localext/split_torch_data_dir.sh will be
                                    used for the actual split, if available
                                    (deft: ${optim_num_partitions})
--optim-use-val-partition <BOOL>  : Whether to use one partition for
                                    validation during hyperparameter
                                    optimization
                                    (deft: ${optim_use_val_partition})
"

. parse_options.sh

if [ $# -lt 1 ]; then
  echo "${help_message}" | grep "Usage" 1>&2
  echo "$0 --help for more info" 1>&2
  exit 1
fi

min_optim_partitions=2
if ${optim_use_val_partition} ; then
  min_optim_partitions=3
fi
if [ "${optim_num_partitions}" -lt ${min_optim_partitions} ]; then
  echo "\
With --optim-use-val-partition set to ${optim_use_val_partition},
--optim-partitions must be at least ${min_optim_partitions}" 1>&2
  exit 1
fi

if python -c "import pydrobert.gpyopt" 2> /dev/null; then
  write_optim_config=true
else
  write_optim_config=false
fi

tmp="$(mktemp -d)"
trap "rm -rf '${tmp}'" EXIT

function get_cfg_dir_name() {
  # input: feat dir path + paths to config files used to generate the model
  #        configuration
  # output: a unique ascii identifier for the configuration
  tmpf="${tmp}/a"
  echo "\
feat_dir=$1
decoding_states=${decoding_states}
min_active=${min_active}
max_active=${max_active}
max_mem=${max_mem}
beam=${beam}
lattice_beam=${lattice_beam}
optim_data_set=${optim_data_set}
optim_num_partitions=${optim_num_partitions}
optim_use_val_partition=${optim_use_val_partition}
" >> "${tmpf}"
  name="$(basename "$1")"
  shift
  while [ $# -gt 0 ]; do
    name="${name}_$(basename "$1" | cut -d'.' -f 1)"
    sort "$1" | sed '/^$/d' >> "${tmpf}"
    shift
  done
  echo "${name}_$(sha1sum "${tmpf}" | head -c 10)"
}

set -e

exp_dir="$(mkdir -p "${exp_dir}"; cd "${exp_dir}"; pwd -P)"

IFS=',' read -a data_group <<< "$1"
shift

cfg_sample_file_last="${tmp}/cfgl"
cfg_sample_file="${tmp}/cfg"
echo "" > "${cfg_sample_file_last}"
while [ $# -gt 0 ]; do
  > "${cfg_sample_file}"
  IFS=',' read -a group <<< "$1"
  shift
  for elem in "${group[@]}"; do
    if [ ! -f "${elem}" ]; then
      echo "'${elem}' does not exist" 1>&2
      exit 1
    fi
    awk '{if ($1 == "") { d="" } else { d=","} printf $1d"'"${elem}"'\n"}' "${cfg_sample_file_last}" \
      >> "${cfg_sample_file}"
  done
  sort "${cfg_sample_file}" > "${cfg_sample_file_last}"
done
cp "${cfg_sample_file_last}" "${cfg_sample_file}"

data_dir_vars=(
  "freq_dim"
  "target_dim"
  "HCLG"
  "gmm_mdl"
  "log_prior"
  "train_data"
  "dev_data"
  "test_data"
  "dev_ref"
  "test_ref"
)

tmp_exp_dir="${tmp}/exp"
mkdir "${tmp_exp_dir}"

for data_dir in "${data_group[@]}"; do
  if [ ! -f "${data_dir}/variables" ]; then
    echo "No file '${data_dir}/variables'" 1>&2
    exit 1
  fi

  unset_variables "${data_dir_vars[@]}" "weight_file"
  . "${data_dir}/variables"
  if ! check_variables_are_set "${data_dir_vars[@]}"; then
    echo "\
Not all variables that were supposed to be set by '${data_dir}/variables' were
set (call '$0 --help' for more info): ${data_dir_vars[*]}" 1>&2
    exit 1
  fi
  feat_cfg="${tmp}/fcfg"
  echo "\
[model]
freq_dim = ${freq_dim}
target_dim = ${target_dim}
" > "${feat_cfg}"

  if $write_optim_config ; then
    if [ -f "localext/split_torch_data_dir.sh" ]; then
      tmp_optim_partitions=( )
      for n in $(seq 1 ${optim_num_partitions}); do
        tmp_optim_partitions=( "${tmp_optim_partitions[@]}" "${tmp}/p${n}.txt" )
      done
      localext/split_torch_data_dir.sh \
        "$(eval echo "\${${optim_data_set}_data}")" "${tmp_optim_partitions[@]}"
    else
      echo "No localext/split_torch_data_dir.sh, optim_partitions will be random"
      tmp_optim_partitions=( ${optim_num_partitions} )
    fi
  fi

  while read line ; do
    IFS=',' read -a cfgs <<< "${line}"
    if [ -z "${cfgs[0]}" ]; then
      cfgs=()
    fi
    cfg_dir_name=$(get_cfg_dir_name "${data_dir}" "${cfgs[@]}")
    for trial in $(seq 1 ${num_trials}); do
      trial_prefix="${cfg_dir_name}/${trial}"
      tmp_trial_path="${tmp_exp_dir}/${trial_prefix}"
      trial_cfg="${tmp}/tcfg"
      echo "\
[model]
seed = $(echo "10 * ${trial}" | bc)

[training]
seed = $(echo "10 * ${trial} + 1" | bc)

[train_data]
seed = $(echo "10 * ${trial} + 2" | bc)

[val_data]
seed = $(echo "10 * ${trial} + 3" | bc)

[pdfs_data]
seed = $(echo "10 * ${trial} + 4" | bc)
" > "${trial_cfg}"

      mkdir -p "${tmp_trial_path}"
      echo "\
model_cfg='${exp_dir}/${trial_prefix}/model.cfg'
state_dir='${exp_dir}/${trial_prefix}/states'
state_csv='${exp_dir}/${trial_prefix}/training.csv'
decode_dev='${exp_dir}/${trial_prefix}/decode_dev'
decode_test='${exp_dir}/${trial_prefix}/decode_test'
decoding_states=${decoding_states}
min_active=${min_active}
max_active=${max_active}
max_mem=${max_mem}
beam=${beam}
lattice_beam=${lattice_beam}" > "${tmp_trial_path}/variables"
cat "${data_dir}/variables" >> "${tmp_trial_path}/variables"

      if $write_optim_config ; then
        echo "\

[optim]
seed = $(echo "10 * ${trial} + 5" | bc)" >> "${trial_cfg}"
        echo -n "\
optim_out_config='${exp_dir}/${trial_prefix}/optimized.cfg'
optim_history_csv='${exp_dir}/${trial_prefix}/optim_history.csv'
optim_data_set=${optim_data_set}
optim_use_val_partition=${optim_use_val_partition}
optim_partitions=( " >> "${tmp_trial_path}/variables"
        if [ "${#tmp_optim_partitions[@]}" = 1 ]; then
          echo "${tmp_optim_partitions[0]} )" >> "${tmp_trial_path}/variables"
        else
          for n in $(seq 1 ${optim_num_partitions}); do
            cp "${tmp_optim_partitions[n-1]}" "${tmp_trial_path}/p${n}.txt"
            echo -n "'${exp_dir}/${trial_prefix}/p${n}.txt' " \
              >> "${tmp_trial_path}/variables"
          done
          echo ")" >> "${tmp_trial_path}/variables"
        fi
      fi

      print-parameters-as-ini \
        "${cfgs[@]}" "${feat_cfg}" "${trial_cfg}" \
        > "${tmp_trial_path}/model.cfg"
      echo "${trial_prefix}" >> "${tmp_exp_dir}/prefixes"
    done
  done < "${cfg_sample_file}"
done

tmp_matrix="${tmp}/matrix"
if $append_to_matrix && [ -s "${exp_dir}/matrix" ]; then
  cp "${exp_dir}/matrix" "${tmp_matrix}"
fi
while read prefix; do
  if [ -s "${exp_dir}/${prefix}/training.csv" ]; then
    echo "\
'${exp_dir}/${prefix}' is being overwritten but its 'training.csv' file is
non-empty. Leaving it as-is, but you might want to delete it yourself if you
want to restart the experiment (or the configuration is stale).
" 1>&2
  fi
  mkdir -p "${exp_dir}/${prefix}"
  rsync -r "${tmp_exp_dir}/${prefix}/" "${exp_dir}/${prefix}"
  echo "${exp_dir}/${prefix}" >> "${tmp_matrix}"
done < "${tmp_exp_dir}/prefixes"
sort -u "${tmp_matrix}" > "${exp_dir}/matrix"
