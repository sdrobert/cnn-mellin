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
beam_widths=( 1 2 4 8 16 32 64 128 )
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
  for (( i=OFFSET; i < ${#models[@]} * ${#feats[@]} * seeds; i += STRIDE )); do
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

if [ $stage -le 17 ]; then
  for (( i = OFFSET; i < ${#models[@]} * ${#feats[@]} * seeds; i += STRIDE )); do
    model="${models[((i / (${#feats[@]} * seeds) ))]}"
    ii=$((i % (${#feats[@]} * seeds) ))
    feat="${feats[((ii / seeds))]}"
    seed="$((ii % seeds))"
    mconf="conf/model/$model-$feat.ini"
    mname="$model-$feat"
    mdir="$exp/$mname/$seed"
    mpth="$mdir/model.pt"
    if [ ! -f "$mpth" ]; then
      echo "Cannot decode $mname with seed $seed: '$mpth' does not exist" \
        "(did you finish stage 16?)" 1>&2
      exit 1
    fi
    for part in dev test; do
      hdir="$mdir/hyp/$part"
      if [ "$part" = dev ]; then
        ddir="$data/$feat/dev"
        active_widths=( "${beam_widths[@]}" )
      else
        ddir="$data/$feat/test"
        active_widths=( "$(awk '
$1 ~ /^best/ {a=gensub(/.*\/dev\.hyp\.([^.]*).*$/, "\\1", 1, $3); print a}
' "$exp/$mname/results.dev.$seed.txt")" )
      fi
      for beam_width in "${active_widths[@]}"; do
        beam_width="$(printf '%02d' $((10#$beam_width + 0)))"
        bdir="$hdir/$beam_width"
        mkdir -p "$bdir"
        if [ ! -f "$bdir/.complete" ]; then
          echo "Beginning stage 5 - decoding $part using $mname with seed" \
            "$seed and beam width $beam_width"
          python asr.py \
            --device "$device" \
            --read-ini "$mconf" \
            decode \
              "$mpth" "$ddir" "$bdir" \
              --beam-width "$beam_width"
          touch "$bdir/.complete"
          echo "Ending stage 5 - decoding $part using $mname with seed" \
            "$seed and beam width $beam_width"
        else
          echo "'$bdir/.complete' exists. Skipping decoding $part using" \
            "$mname with seed $seed and beam width $beam_width"
        fi
        if [ ! -f "$mdir/$part.hyp.$beam_width.trn" ]; then
          echo "Beginning stage 5 - gathering hyps for $part using $mname" \
            "with $seed and beam with $beam_width"
          torch-token-data-dir-to-trn \
            "$bdir" "$data/ext/id2token.txt" \
            "$mdir/$part.hyp.$beam_width.utrn"
          python prep/timit.py "$data" filter \
            "$mdir/$part.hyp.$beam_width."{u,}trn
          echo "Ending stage 5 - gathering hyps for $part using $mname" \
            "with seed $seed and beam with $beam_width"
        fi
      done
      active_files=( "$mdir/$part.hyp."*.trn )
      if [ ${#active_files[@]} -ne ${#active_widths[@]} ]; then
        echo "The number of evaluated beam widths does not equal the number" \
          "of hypothesis files for partition '$part' in '$mdir'. This could" \
          "mean you changed the -b parameter after running once or you reran" \
          "experiments with different parameters and the partition is" \
          "'test'. Delete all hyp files in '$amdir' and try running this step"\
          "again" 1>&2
        exit 1
      fi
      [ -f "$exp/$mname/results.$part.$seed.txt" ] || \
        python prep/error-rates-from-trn.py \
          "$data/$feat/ext/$part.ref.trn" "$mdir/$part.hyp."*.trn \
          --suppress-warning > "$exp/$mname/results.$part.$seed.txt"
    done
  done
  ((only)) && exit 1
fi

if ((only)); then
  echo "Stage $stage does not exist. The -x flag must be paired " \
    "with an existing stage" 1>&2
  exit 1
fi

# compute descriptives for all the dependencies
echo "Phone Error Rates:"
for part in dev test; do
  for mdir in $(find "$exp" -maxdepth 1 -mindepth 1 -type d | sort ); do
    results=( $(find "$mdir" -name "results.$part.*.txt" -print) )
    if [ "${#results[@]}" -gt 0 ]; then
      echo -n "$part ${mdir##*/}: "
      awk '
BEGIN {n=0; s=0; min=1000; max=0}
$1 ~ /best/ {
  x=substr($NF, 1, length($NF) - 1) + 0;
  a[n++]=x; s+=x; if (x < min) min=x; if (x > max) max=x;
}
END {
  mean=s/n; med=a[int(n/2)];
  var=0; for (i=0;i<n;i++) var+=(a[i] - mean) * (a[i] - mean) / n; std=sqrt(var);
  printf "n=%d, mean=%.1f%%, std=%.1f%%, med=%.1f%%, min=%.1f%%, max=%.1f%%\n", n, mean, std, med, min, max;
}' "${results[@]}"
    fi
  done
done

exit 0

# dev lcorr-fbank-81-10ms: n=20, mean=15.5%, std=0.2%, med=15.8%, min=15.2%, max=15.9%
# dev lcorr-sigbank-41-2ms: n=20, mean=17.5%, std=0.2%, med=17.3%, min=17.1%, max=17.9%
# dev mcorr-fbank-81-10ms: n=20, mean=17.5%, std=0.2%, med=17.5%, min=17.2%, max=17.8%
# dev mcorr-sigbank-41-2ms: n=20, mean=17.3%, std=0.3%, med=17.2%, min=16.9%, max=18.0%
# test lcorr-fbank-81-10ms: n=20, mean=17.8%, std=0.3%, med=17.7%, min=17.2%, max=18.3%
# test lcorr-sigbank-41-2ms: n=20, mean=19.8%, std=0.4%, med=20.2%, min=18.8%, max=20.6%
# test mcorr-fbank-81-10ms: n=20, mean=19.7%, std=0.3%, med=19.5%, min=19.2%, max=20.4%
# test mcorr-sigbank-41-2ms: n=20, mean=19.5%, std=0.5%, med=18.6%, min=18.6%, max=20.2%