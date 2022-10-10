if [ -z "$stage" ]; then
  echo \
    "This script should not be run directly. Use './timit.sh -q'" 1>&2
  exit 1
fi

opt="$exp/optim"
ckpt="${TIMIT_CKPT_DIR:-"$opt/ckpts"}"
mkdir -p "$opt/conf" "$ckpt" "$exp/completed_stages"

[ -f ./db_creds.sh ] && source db_creds.sh
if [ -z "$db_url" ]; then
  echo \
    "variable 'db_url' unset. Please create a file db_creds.sh in the folder" \
    "containing timit.sh and set the variable there" 2>&1
  exit 1
fi

max_retries=10
declare -A num_epochs_map=( [sm]=40 [md]=50 [lg]=60 )
declare -A num_trials_map=( [sm]=512 [md]=256 [lg]=128 )
declare -A top_k_map=( [md]=10 [lg]=5 )
declare -A pruner_map=( [sm]=none [md]=none [lg]=hyperband )
declare -A sampler_map=( [sm]=random [md]=random [lg]=tpe )
declare -A prev_sz_map=( [md]=sm [lg]=md )
declare -A gpu_mem_limit_map=( \
  [sm]="$(python -c 'print(6 * (1024 ** 3))')" \
  [md]="$(python -c 'print(9 * (1024 ** 3))')" \
  [lg]="$(python -c 'print(12 * (1024 ** 3))')" \
)

# usually one would use bc, but it isn't installed on Azure...
float_lt () [[ $(python -c "print(int($1 < $2))") -eq 1 ]]

is_complete () [[ "$(optuna trials --study-name $1 --storage $db_url -f yaml 2> /dev/null | grep -e 'state: COMPLETE' | wc -l)" -ge "$2" ]]

get_best_prior () {
  # determine the best parameter setting up to and excluding the current
  # e.g.: get_best_prior mcorr fbank-81-10ms model-sm
  local model="$1"
  local feat="$2"
  if [ "$3" = "model-sm" ]; then
    echo "conf/optim/model-${model}-sm.ini"
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
    local cur_name="${pair/-/-${model}-}-${feat}"
    local cur_trials="${num_trials_map[${pair#*-}]}"
    if ! is_complete $cur_name $cur_trials; then
      echo "$cur_name is not complete. Cannot get best prior" 1>&2
      return 1
    fi
    local cur_best="$(optuna best-trial --study-name ${cur_name} --storage $db_url -f yaml 2> /dev/null | grep -e 'value: ' | cut -d ' ' -f 2)"
    [ -z "$cur_best" ] && return 1
    if float_lt "${cur_best}" "${best}"; then
      best="${cur_best}"
      best_name="${cur_name}"
    fi
  done
  echo "$best_name"
  return 0
}

init_study () {
  # e.g. init_study mcorr fbank-81-10ms model-sm
  local model="$1"
  local feat="$2"
  local sty="${3%-*}"
  local sz="${3#*-}"
  local study_name="${3/-/-${model}-}-${feat}"
  local blacklist=()
  local pruner="${pruner_map[$sz]}"
  local num_epochs="${num_epochs_map[$sz]}"
  local gpu_mem_limit="${gpu_mem_limit_map[$sz]}"
  local num_trials="${num_trials_map[$sz]}"
  local prev_sz="${prev_sz_map[$sz]}"
  local top_k="${top_k_map[$sz]}"
  if [ "$3" = "model-sm" ]; then
    local prior_cmd="cat conf/optim/model-${model}-sm.ini"
    blacklist+=( 'training.*' 'data.*' 'model.convolutional_mellin' )
  else
    if [ "$3" = "train-sm" ]; then
      blacklist+=( 'model.*' 'training.max_.*_mask' 'training.num_epochs' 'training.early_.*' )
    fi
    local best_prior="$(get_best_prior $1 $2 $3)"
    [ -z "$best_prior" ] && return 1
    local prior_cmd="python asr.py optim --study-name $best_prior $db_url best -"
  fi
  if [ "$sty" = "model" ]; then
    num_epochs=$(( $num_epochs - 10 ))
  fi
  $prior_cmd | sed -n '/No CUDA runtime is found/!p' | \
    sed 's/\(max_.*_mask\)[ ]*=.*/\1 = 10000/g;s/num_epochs[ ]*=.*/num_epochs = '"${num_epochs}"'/g' \
    > "$opt/conf/${study_name}.ini"
  if [ "$sz" = "sm" ]; then
    local select_args=( --blacklist "${blacklist[@]}" )
  else
    python asr.py \
      optim \
        --study-name ${sty}-${model}-${prev_sz}-${feat} \
        "$db_url" \
        important \
          --top-k=${top_k} \
          "$opt/conf/${study_name}.params"
    local select_args=( --whitelist $(cat "$opt/conf/${study_name}.params") )
  fi
  python asr.py \
    --read-ini "$opt/conf/${study_name}.ini" \
    --device $device \
    optim \
      --study-name ${study_name} \
      "$db_url" \
      init \
        "$data/$feat/train" \
        ${select_args[@]} \
        --num-data-workers 4 \
        --num-trials ${num_trials} \
        --mem-limit-bytes ${gpu_mem_limit} \
        --pruner ${pruner}
  return $?
}

run_study () {
  local model="$1"
  local feat="$2"
  local sz="${3#*-}"
  local study_name="${3/-/-${model}-}-${feat}"
  local num_trials="${num_trials_map[$sz]}"
  local sampler="${sampler_map[$sz]}"
  if is_complete $study_name $num_trials; then
    echo "Already done ${num_trials} trials for ${study_name}"
    return 0
  fi
  for n in $(seq 1 $max_retries); do
    echo "Sleeping 2min to wear out heartbeat"
    sleep $(( 120 + OFFSET ))
    echo "Attempt $n/$max_retries to optimize '${study_name}'"
    python asr.py \
      --model-dir "$ckpt" \
      optim --study-name "${study_name}" "$db_url" run \
      --sampler "$sampler" && echo "Run $n/$max_retries succeeded!" && break
    echo "Run $n/$max_retries failed!"
  done
  return $?
}

for s in {03..14}; do
  if [ $stage -le $s ]; then
    if [ ! -f "$exp/completed_stages/$s" ]; then
      echo "Beginning stage $s"
      ss=$((10#$s - 3))
      ((ss % 2)) && cmd=run_study || cmd=init_study
      ss=$((ss / 2))
      ((ss % 2)) && m_or_t=train || m_or_t=model
      ss=$((ss / 2))
      if [ $ss -eq 0 ]; then
        sz=sm
      elif [ $ss -eq 1 ]; then
        sz=md
      else
        sz=lg
      fi
      all_done=1
      for model in "${models[@]}"; do
        for feat in "${feats[@]}"; do
          $cmd $model $feat $m_or_t-$sz
          [ "$cmd" = "run_study" ] && \
            is_complete "$m_or_t-$model-$sz-$feat" "${num_trials_map[$sz]}" || \
            all_done=0
        done
      done
      # run_study returns when all remaining trials are either completed or
      # being run by someone else. Only touch the flag when the study is
      # actually completed.
      ((all_done)) && touch "$exp/completed_stages/$s"
      echo "Finished stage $s"
    else
      echo "'$exp/completed_stages/$s' exists. Skipping stage $s"
    fi
    ((only)) && exit 0
  fi
done
