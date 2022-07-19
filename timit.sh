#! /usr/bin/env bash

set -e

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

argcheck_list() {
  arg="$1"
  cmd="$2"
  read -ra a <<<"$3"
  shift 3
  for x in "${a[@]}"; do
    $cmd $arg "$1" "$@"
  done
}

argcheck_nat() {
  if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "$0: '-$1' argument '$2' is not a natural number" 1>&2
    usage
  fi
}

argcheck_rdir() {
  if ! [ -d "$2" ]; then
    echo "$0: '-$1' argument '$2' is not a readable directory." 1>&2
    usage
  fi
}

argcheck_writable() {
  if ! [ -w "$2" ]; then
    echo "$0: '-$1' argument '$2' is not writable." 1>&2
    usage
  fi
}

argcheck_choices() {
  if ! [[ " $3 " =~ " $2 " ]]; then
    echo "$0: '-$1' argument '$2' not one of '$3'." 1>&2
    usage
  fi
}

check_config() {
  if [[ " ${INVALIDS[*]} " =~ " $1_$2 " ]] || \
     [[ " ${INVALIDS[*]} " =~ " $2_$3 " ]] || \
     [[ " ${INVALIDS[*]} " =~ " $1_$3 " ]] ; then
    return 1
  fi
  mkdir -p "$confdir"
  yml="$confdir/$1_$2_$3.yaml"
  combine-yaml-files \
    --nested --quiet \
    conf/proto/{base,model_$1,estimator_$2,lm_$3}.yaml "$yml"
}

# constants
ALL_FEATS=( fbank-81-10ms sigbank-41-2ms )
ALL_MODELS=( lcorr mcorr )
INVALIDS=( )

# variables
stage=1
timit=
data=data/timit
exp=exp/timit
seeds=20
offset=1
device=cuda
feats=( "${ALL_FEATS[@]}" )
models=( "${ALL_MODELS[@]}" )
beam_widths=( 1 2 4 8 16 32 )
only=0
do_hyperparam=0
check_jit=1

while getopts "qxhws:i:d:o:b:n:k:c:m:e:l:" opt; do
  case $opt in
    s)
      argcheck_nat $opt "$OPTARG"
      stage=$OPTARG
      ;;
    k)
      argcheck_nat $opt "$OPTARG"
      offset=$OPTARG
      ;;
    i)
      argcheck_rdir $opt "$OPTARG"
      timit="$OPTARG"
      ;;
    d)
      argcheck_writable $opt "$OPTARG"
      data="$OPTARG"
      ;;
    o)
      argcheck_writable $opt "$OPTARG"
      exp="$OPTARG"
      ;;
    b)
      argcheck_list $opt argcheck_nat "$OPTARG"
      beam_widths=( $OPTARG )
      ;;
    n)
      argcheck_nat $opt "$OPTARG"
      seeds=$OPTARG
      ;;
    c)
      device="$OPTARG"
      ;;
    f)
      argcheck_list $opt argcheck_choices "$OPTARG" "${ALL_FEATS[*]}"
      feats=( $OPTARG )
      ;;
    m)
      argcheck_list $opt argcheck_choices "$OPTARG" "${ALL_MODELS[*]}"
      models=( $OPTARG )
      ;;
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
  esac
done

if [ $# -ne $(($OPTIND - 1)) ]; then
  echo "Expected no positional arguments but found one: '${@:$OPTIND}'" 1>&2
  usage
fi

if ((check_jit)); then
  python -c 'import warnings; warnings.simplefilter("error"); import mconv'
fi

# prep the dataset
if [ $stage -le 1 ]; then
  if [ ! -f "$data/.complete" ]; then
    if [ -z "$timit" ]; then
      echo "timit directory unset, but needed for this command (use -i)" 1>&2
      exit 1
    fi
    python prep/timit.py "$data" preamble "$timit"
    python prep/timit.py "$data" init_phn --lm
    touch "$data/.complete"
  fi
  for feat in "${feats[@]}"; do
    if [ ! -f "$data/$feat/.complete" ]; then 
      python prep/timit.py "$data" torch_dir phn48 "$feat" \
        --computer-json "conf/feats/$feat.json" \
        --seed 0
      touch "$data/$feat/.complete"
    fi
    ((only)) && exit 0
  done
fi

((do_hyperparam)) && source scripts/hyperparam.sh

for model in "${models[@]}"; do
  for feat in "${feats[@]}"; do
    mconf="conf/model/$model-$feat.ini"
    if [ ! -f "$mconf" ]; then
      echo "could not find '$mconf'" \
        "(did you finish the hyperparameter search?)" 1>&2
      exit 1
    fi
  done
done

exit 0
