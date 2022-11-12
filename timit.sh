#! /usr/bin/env bash

# XXX(sdrobert): The awkward handling of foreach-loops allows for independent
# parallelization over tasks via striding. If N processes in total need to
# complete M independent tasks, then if process n is assigned tasks i according
# to the for-loop
# 
#   for (i=n; i < M; i += N) { ... }
# 
# then all M tasks will be distributed roughly evenly over the N processes.
# 
# To perform only this work, a process can set environment variables prior
# to running this script
# 
#   export TIMIT_OFFSET=n
#   export TIMIT_STRIDE=N
# 
# We don't expose the offset and stride as flags to the user because the logic
# of the script can break. This can happen when the process is allowed to
# continue to the next stage as there's no guarantee that the work it just
# finished is the only work it depends on. This script doesn't have
# synchronization blocks, only early termination via the flag -x. The intended
# synchronization method is therefore to run the script a stage at a time,
# blocking between stages when necessary.

set -e

source scripts/utils.sh

usage () {
  cat << EOF 1>&2
Usage: $0 [options]

Options:
  -s N            Run from stage N.
                  Value: '$stage'
  -i PTH          Location of TIMIT data directory (unprocessed).
                  Value: '$timit'
  -d PTH          Location of processed data directory.
                  Value: '$data'
  -o PTH          Location to store experiment artifacts.
                  Value: '$exp'
  -O PTH          If set, specifies seperate location to store model
                  checkpoints. Otherwise, store with other experiment
                  artifacts (-o).
  -b 'A [B ...]'  The beam widths to test for decoding
  -n N            Number of repeated trials to perform.
                  Value: '$seeds'
  -k N            Offset (inclusive) of the seed to start from.
                  Value: '$offset'
  -c DEVICE       Device to run experiments on.
                  Value: '$device'
  -f 'A [B ...]'  Feature configurations to experiment with.
  -m 'A [B ...]'  Model configurations to experiment with.
                  Value: '${models[*]}'
  -q              Perform hyperparameter optimization step (must begin on or
                  before stage 2).
  -x              Run only the current stage.
  -w              Do not raise on failure to compile JIT scripts.
  -h              Display usage info and exit.
EOF
  exit ${1:-1}
}


check_config() {
  if [[ " ${INVALIDS[*]} " =~ " $1_$2 " ]] || \
     [[ " ${INVALIDS[*]} " =~ " $2_$3 " ]] || \
     [[ " ${INVALIDS[*]} " =~ " $1_$3 " ]] ; then
    return 1
  fi
  return 0
}

build_yaml() {
  mkdir -p "$confdir"
  yml="$confdir/$1_$2_$3.yaml"
  combine-yaml-files \
    --nested --quiet \
    conf/proto/{base,model_$1,estimator_$2,lm_$3}.yaml "$yml"
}

# constants
#
# XXX(sdrobert): do not use underscores in the names of parts of configurations
# (feats, models, etcs). Underscores are used to join those parts into a name.
ALL_FEATS=( fbank-81-10ms sigbank-41-2ms )
ALL_MODELS=( lcorr mcorr )
# invalids are regexes 
INVALIDS=( )
OFFSET="${TIMIT_OFFSET:-0}"
STRIDE="${TIMIT_STRIDE:-1}"

# variables
stage=0
timit=
data=data/timit
exp=exp/timit
seeds=20
max_retries=50
offset=1
device=cuda
feats=( "${ALL_FEATS[@]}" )
models=( "${ALL_MODELS[@]}" )
beam_widths=( 1 2 4 8 16 32 )
only=0
do_hyperparam=0
check_jit=1

while getopts "qxhws:k:i:d:o:O:b:n:c:f:m:" opt; do
  case $opt in
    q)
      do_hyperparam=1
      ;;
    x)
      only=1
      ;;
    w)
      check_jit=0
      ;;
    h)
      echo "Shell recipe to perform experiments on TIMIT." 1>&2
      usage 0
      ;;
    s)
      argcheck_is_nat $opt "$OPTARG"
      stage=$OPTARG
      ;;
    k)
      argcheck_is_nat $opt "$OPTARG"
      offset=$OPTARG
      ;;
    i)
      argcheck_is_readable $opt "$OPTARG"
      timit="$OPTARG"
      ;;
    d)
      # we check permissions for the -d and -o directories when we actually
      # need them. This is largely for Azure, which will only mount these
      # with appropriate permissions and when necessary.
      data="$OPTARG"
      ;;
    o)
      exp="$OPTARG"
      ;;
    O)
      ckpt_dir="$OPTARG"
      ;;
    b)
      argcheck_all_nat $opt "$OPTARG"
      beam_widths=( $OPTARG )
      ;;
    n)
      argcheck_is_nat $opt "$OPTARG"
      seeds=$OPTARG
      ;;
    c)
      device="$OPTARG"
      ;;
    f)
      argcheck_all_a_choice $opt "${ALL_FEATS[@]}" "$OPTARG"
      feats=( $OPTARG )
      ;;
    m)
      argcheck_all_a_choice $opt "${ALL_MODELS[@]}" "$OPTARG"
      models=( $OPTARG )
      ;;
  esac
done

if [ $# -ne $(($OPTIND - 1)) ]; then
  echo "Expected no positional arguments but found one: '${@:$OPTIND}'" 1>&2
  usage
fi

if ((check_jit)); then
  python -c 'import warnings; warnings.simplefilter("error"); import mconv'
fi

if ((only)) && [ $stage = 0 ]; then
  echo "The -x flag must be paired with the -s flag" 1>&2
  exit 1
fi

# prep the dataset
if [ $stage -le 1 ]; then
  if [ ! -f "$data/local/.complete" ]; then
    echo "Beginning stage 1"
    if [ -z "$timit" ]; then
      echo "timit directory unset, but needed for this command (use -i)" 1>&2
      exit 1
    fi
    argcheck_is_writable d "$data"
    argcheck_is_readable i "$timit"
    python prep/timit.py "$data" preamble "$timit"
    python prep/timit.py "$data" init_phn --lm
    touch "$data/local/.complete"
    echo "Finished stage 1"
  else
    echo "$data/.complete exists already. Skipping stage 1."
  fi
  ((only)) && exit 0
fi


if [ $stage -le 2 ]; then
  for (( i=$OFFSET; i < ${#feats[@]}; i += $STRIDE )); do
    feat="${feats[$i]}"
    if [ ! -f "$data/$feat/.complete" ]; then
      argcheck_is_rw d "$data"
      echo "Beginning stage 2 with feats $feat"
      python prep/timit.py "$data" torch_dir phn48 "$feat" \
        --computer-json "conf/feats/$feat.json" \
        --seed 0
      touch "$data/$feat/.complete"
      echo "Finished stage 2 with feats $feat"
    else
      echo \
        "$data/$feat/.complete already exists." \
        "Skipping stage 2 with feats $feat"
    fi
  done
  ((only)) && exit 0
fi

argcheck_is_readable d "$data"
mkdir -p "$exp" 2> /dev/null
argcheck_is_rw o "$exp"

if ((do_hyperparam)); then
  source scripts/hyperparam.sh
elif [ "$stage" -le 15 ]; then
  if ((only)); then
    echo "Stage $stage is a part of hyperparameter selection. If you want " \
      "to run that, include the flag -q. Otherwise skip to stage 16." 1>&2
    exit 1
  fi
  echo \
    "Skipping hyperparameter stages $stage to 16. If you wanted to do" \
    "these, rerun with the -q flag."
fi

# for model in "${models[@]}"; do
#   for feat in "${feats[@]}"; do
#     mconf="conf/model/$model-$feat.ini"
#     if [ ! -f "$mconf" ]; then
#       echo "could not find '$mconf'" \
#         "(did you finish the hyperparameter search?)" 1>&2
#       exit 1
#     fi
#   done
# done

if [ $stage -le 16 ]; then
  for (( i=OFFSET; i < ${#models[@]} * ${#feats[@]} * seeds; i += $STRIDE )); do
    model="${models[((i / (${#feats[@]} * seeds) ))]}"
    ii=$((i % (${#feats[@]} * seeds) ))
    feat="${feats[((ii / seeds))]}"
    seed="$((ii % seeds))"
    mconf="conf/model/$model-$feat.ini"
    model_dir="$exp/$model-$feat/$seed"
    mkdir -p "$model_dir"
    if [ ! -f "$model_dir/model.pt" ]; then
      echo "Beginning training of $model-$feat with seed $seed"
      for ((n=1; n < max_retries; n++)); do
        echo "Attempt $n/$max_retries of training"
        python asr.py ${ckpt_dir+--ckpt-dir "$ckpt_dir/$model-$feat/$seed"} \
          --device "$device" \
          --read-ini "$mconf" \
          train \
            "$data/$feat/train" \
            "$data/$feat/dev" \
            "$model_dir" \
            --seed $seed && echo "Run $n/$max_retries succeeded!" && break
          echo "Run $n/$max_retries failed"
      done
      [ $n -eq $max_retries ] && exit 1
      echo "Done training of $model-$feat with seed $seed"
    else
      echo "'$model_dir/model.pt' already exists; skipping training"
    fi
  done
  ((only)) && exit 1
fi

if ((only)); then
  echo "Stage $stage does not exist. The -x flag must be paired " \
    "with an existing stage" 1>&2
  exit 1
fi

exit 0
