#! /usr/bin/env bash

# Usage: scripts/hyperparam.sh [db_url] [step]
# 
# Similar to a Kaldi script. You can try to run the whole thing at once or
# resume from a step.

[ -f ./db_creds.sh ] && source db_creds.sh

if [ -z "$db_url" ]; then
  if [ $# -lt 1 ]; then
    echo "Do not have a db_url value. Either specify in db_creds.sh:

      db_url=my-url-here
    
    or pass the value as the first argument to this script.
    " 1>&2
    exit 1
  fi
  db_url="$1"
  shift
fi

step="${1:-0}"

declare -A num_epochs_map=( [sm]=40 [md]=50 [lg]=60 )
declare -A num_trials_map=( [sm]=256 [md]=128 [lg]=64 )
declare -A top_k_map=( [md]=10 [lg]=5 )
declare -A pruner_map=( [sm]=none [md]=none [lg]=hyperband )
declare -A sampler_map=( [sm]=random [md]=random [lg]=tpe )
declare -A prev_sz_map=( [md]=sm [lg]=md )
declare -A gpu_mem_limit_map=( [sm]="$(python -c 'print(6 * (1024 ** 3))')" [md]="$(python -c 'print(9 * (1024 ** 3))')" [lg]="$(python -c 'print(12 * (1024 ** 3))')" )
do_async=1
optim_command="sbatch scripts/cnn_mellin.slrm"

# infer the available model types from conf/optim folder (it's probably
# mcorr and lcorr)
model_types=( $(echo conf/optim/model-*-sm.ini | tr ' ' $'\n' | cut -d - -f 2 | sort) )
# infer the available feature types from the data/timit folder
feature_types=( $(echo data/timit/* | tr ' ' $'\n' | sed 's:data/timit/::g; /timit/d; /local/d' | sort) )

mkdir -p exp/{logs,conf}

is_complete () {
  [[ "$(optuna trials --study-name $1 --storage $db_url -f yaml 2> /dev/null | grep -e 'state: COMPLETE' | wc -l)" -ge "$2" ]]
  return $?
}

get_best_prior () {
  # determine the best parameter setting up to and excluding the current
  # e.g.: get_best_prior mcorr fbank-81-10ms model-sm
  local model_type="$1"
  local feature_type="$2"
  if [ "$3" = "model-sm" ]; then
    echo "conf/optim/model-${model_type}-sm.ini"
    return 0
  fi
  local upto_sty="${3%-*}"
  local upto_sz="${3#*-}"
  pairs=( )
  if [ "$upto_sz" != "sm" ]; then
    pairs+=( "model-sm" "train-sm" )
    if [ "$upto_sz" = "lg" ]; then
      pairs+=( "model-md" "train-md" )
    fi
  fi
  if [ "$upto_sty" = "train" ]; then
    pairs+=( "model-$upto_sz" )
  fi
  local best=1000
  local best_name=
  for pair in "${pairs[@]}"; do
    local cur_name="${pair/-/-${model_type}-}-${feature_type}"
    local cur_trials="${num_trials_map[${pair#*-}]}"
    if ! is_complete $cur_name $cur_trials; then
      echo "$cur_name is not complete. Cannot get best prior" 1>&2
      return 1
    fi
    local cur_best="$(optuna best-trial --study-name ${cur_name} --storage $db_url -f yaml 2> /dev/null | grep -e 'value: ' | cut -d ' ' -f 2)"
    [ -z "$cur_best" ] && return 1
    if (( $(echo "${cur_best} < ${best}" | bc -l) )); then
      best="${cur_best}"
      best_name="${cur_name}"
    fi
  done
  echo "$best_name"
  return 0
}

init_study () {
  # e.g. init_study mcorr fbank-81-10ms model-sm
  local model_type="$1"
  local feature_type="$2"
  local sty="${3%-*}"
  local sz="${3#*-}"
  local study_name="${3/-/-${model_type}-}-${feature_type}"
  local blacklist=()
  local pruner="${pruner_map[$sz]}"
  local num_epochs="${num_epochs_map[$sz]}"
  local gpu_mem_limit="${gpu_mem_limit_map[$sz]}"
  local num_trials="${num_trials_map[$sz]}"
  local prev_sz="${prev_sz_map[$sz]}"
  local top_k="${top_k_map[$sz]}"
  if [ "$3" = "model-sm" ]; then
    local prior_cmd="cat conf/optim/model-${model_type}-sm.ini"
    blacklist+=( 'training.*' 'data.*' 'model.convolutional_mellin' )
  else
    if [ "$3" = "train-sm" ]; then
      blacklist+=( 'model.*' 'training.max_.*_mask' 'training.num_epochs' 'training.early_.*' )
    fi
    local best_prior="$(get_best_prior $1 $2 $3)"
    [ -z "$best_prior" ] && return 1
    local prior_cmd="cnn-mellin optim --study-name $best_prior $db_url best -"
  fi
  if [ "$sty" = "model" ]; then
    num_epochs=$(( $num_epochs - 10 ))
  fi
  $prior_cmd | \
    sed 's/\(max_.*_mask\)[ ]*=.*/\1 = 10000/g;s/num_epochs[ ]*=.*/num_epochs = '"${num_epochs}"'/g' \
    > exp/conf/${study_name}.ini
  if [ "$sz" = "sm" ]; then
    local select_args=( --blacklist "${blacklist[@]}" )
  else
    cnn-mellin \
      optim \
        --study-name ${sty}-${model_type}-${prev_sz}-${feature_type} \
        "$db_url" \
        important \
          --top-k=${top_k} \
          exp/conf/${study_name}.params
    local select_args=( --whitelist $(cat exp/conf/${study_name}.params) )
  fi
  cnn-mellin \
    --read-ini exp/conf/${study_name}.ini \
    --device cuda \
    optim \
      --study-name ${study_name} \
      "$db_url" \
      init \
        data/timit/${feature_type}/train \
        ${select_args[@]} \
        --num-data-workers 4 \
        --num-trials ${num_trials} \
        --mem-limit-bytes ${gpu_mem_limit} \
        --pruner ${pruner}
  return $?
}

run_study () {
  local model_type="$1"
  local feature_type="$2"
  local sz="${3#*-}"
  local study_name="${3/-/-${model_type}-}-${feature_type}"
  local num_trials="${num_trials_map[$sz]}"
  local sampler="${sampler_map[$sz]}"
  if is_complete $study_name $num_trials; then
    echo "Already done ${num_trials} trials for ${study_name}"
    return 0
  fi
  $optim_command \
    optim --study-name "${study_name}" "$db_url" run \
    --sampler "$sampler" &
  [ $do_async != 1 ] && wait $!
  return 0
}

if [ $step -le 1 ]; then
  for model_type in "${model_types[@]}"; do
    for feature_type in "${feature_types[@]}"; do
      init_study $model_type $feature_type model-sm || exit 1
    done
  done
fi

if [ $step -le 2 ]; then
  for model_type in "${model_types[@]}"; do
    for feature_type in "${feature_types[@]}"; do
      run_study $model_type $feature_type model-sm || exit 1
    done
  done
  wait
fi

if [ $step -le 3 ]; then
  for model_type in "${model_types[@]}"; do
    for feature_type in "${feature_types[@]}"; do
      init_study $model_type $feature_type train-sm || exit 1
    done
  done
fi

if [ $step -le 4 ]; then
  for model_type in "${model_types[@]}"; do
    for feature_type in "${feature_types[@]}"; do
      run_study $model_type $feature_type train-sm || exit 1
    done
  done
  wait
fi

if [ $step -le 5 ]; then
  for model_type in "${model_types[@]}"; do
    for feature_type in "${feature_types[@]}"; do
      init_study $model_type $feature_type model-md || exit 1
    done
  done
fi

if [ $step -le 6 ]; then
  for model_type in "${model_types[@]}"; do
    for feature_type in "${feature_types[@]}"; do
      run_study $model_type $feature_type model-md || exit 1
    done
  done
  wait
fi

if [ $step -le 7 ]; then
  for model_type in "${model_types[@]}"; do
    for feature_type in "${feature_types[@]}"; do
      init_study $model_type $feature_type train-md || exit 1
    done
  done
fi

if [ $step -le 8 ]; then
  for model_type in "${model_types[@]}"; do
    for feature_type in "${feature_types[@]}"; do
      run_study $model_type $feature_type train-md || exit 1
    done
  done
  wait
fi