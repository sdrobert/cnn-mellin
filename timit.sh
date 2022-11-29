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
  -b 'A [B ...]'  The beam widths to test for decoding.
                  Value: '${beam_widths[*]}'
  -B 'A [B ...]'  LM mixing coefficients to test for decoding.
                  Value: '${lm_betas[*]}'
  -n N            Number of repeated trials to perform.
                  Value: '$seeds'
  -k N            Offset (inclusive) of the seed to start from.
                  Value: '$offset'
  -c DEVICE       Device to run experiments on.
                  Value: '$device'
  -f 'A [B ...]'  Feature configurations to experiment with.
                  Value: '${feats[*]}'
  -m 'A [B ...]'  Model configurations to experiment with.
                  Value: '${models[*]}'
  -H              Perform hyperparameter optimization step (must begin on or
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
lm_betas=( 0.0 0.2 0.5 1 )
only=0
do_hyperparam=0
check_jit=1
quiet=0

while getopts "Hqxhws:k:i:d:o:O:b:B:n:c:f:m:" opt; do
  case $opt in
    q)
      ((quiet+=1)) || true
      ;;
    H)
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
    B)
      argcheck_all_nnfloat $opt "$OPTARG"
      lm_betas=( $OPTARG )
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

beam_widths=( $(printf "%02d " "${beam_widths[@]}") )
lm_betas=( $(printf "%3.2f " "${lm_betas[@]}") )

if ((only)) && [ $stage = 0 ]; then
  echo "The -x flag must be paired with the -s flag" 1>&2
  exit 1
fi

quiet_flag=
if [ $quiet -ne 0 ]; then
  quiet_flag="--quiet"
fi

qecho () {
  if ((quiet<2)); then
    echo "$@"
  fi
}

# prep the dataset
if [ $stage -le 1 ]; then
  if [ ! -f "$data/local/.complete" ]; then
    qecho "Beginning stage 1"
    if [ -z "$timit" ]; then
      echo "timit directory unset, but needed for this command (use -i)" 1>&2
      exit 1
    fi
    argcheck_is_writable d "$data"
    argcheck_is_readable i "$timit"
    python prep/timit.py "$data" preamble "$timit"
    python prep/timit.py "$data" init_phn --lm
    touch "$data/local/.complete"
    qecho "Finished stage 1"
  else
    qecho "$data/.complete exists already. Skipping stage 1."
  fi
  ((only)) && exit 0
fi


if [ $stage -le 2 ]; then
  for (( i=$OFFSET; i < ${#feats[@]}; i += $STRIDE )); do
    feat="${feats[$i]}"
    if [ ! -f "$data/$feat/.complete" ]; then
      argcheck_is_rw d "$data"
      qecho "Beginning stage 2 with feats $feat"
      python prep/timit.py "$data" torch_dir phn48 "$feat" \
        --computer-json "conf/feats/$feat.json" \
        --seed 0
      python prep/arpa-lm-to-state-dict.py \
        "$data/$feat/ext/"{lm.arpa.gz,token2id.txt,lm.pt} \
        --remove-eos --save-sos --save-vocab-size
      touch "$data/$feat/.complete"
      qecho "Finished stage 2 with feats $feat"
    else
      qecho \
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
      "to run that, include the flag -H. Otherwise skip to stage 16." 1>&2
    exit 1
  fi
  qecho \
    "Skipping hyperparameter stages $stage to 16. If you wanted to do" \
    "these, rerun with the -H flag."
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
    mdir="$exp/$model-$feat/$seed"
    mkdir -p "$mdir"
    if [ ! -f "$mdir/model.pt" ]; then
      qecho "Beginning training of $model-$feat with seed $seed"
      for ((n=1; n < max_retries; n++)); do
        qecho "Attempt $n/$max_retries of training"
        python asr.py ${ckpt_dir+--ckpt-dir "$ckpt_dir/$model-$feat/$seed"} \
          --device "$device" $quiet_flag \
          --read-ini "$mconf" \
          train \
            "$data/$feat/train" \
            "$data/$feat/dev" \
            "$mdir" \
            --seed $seed && echo "Run $n/$max_retries succeeded!" && break
        echo "Run $n/$max_retries failed" 1>&2
      done
      [ $n -eq $max_retries ] && exit 1
      qecho "Done training of $model-$feat with seed $seed"
    else
      qecho "'$mdir/model.pt' already exists; skipping training"
    fi
  done
  ((only)) && exit 0
fi

if [ $stage -le 17 ]; then
  for (( i = OFFSET; i < ${#models[@]} * ${#feats[@]} * seeds * 2; i += STRIDE )); do
    model="${models[((i / (${#feats[@]} * seeds * 2) ))]}"
    ii=$((i % (${#feats[@]} * seeds * 2) ))
    feat="${feats[((ii / (seeds * 2) ))]}"
    ii=$((ii % (seeds * 2) ))
    $((ii / seeds)) && part=test || part=dev
    seed=$((ii % seeds))
    ddir="$data/$feat/$part"
    mconf="conf/model/$model-$feat.ini"
    mname="$model-$feat"
    mdir="$exp/$mname/$seed"
    mpth="$mdir/model.pt"
    if [ ! -f "$mpth" ]; then
      echo "Cannot compute logits of '$ddir' for $mname with seed" \
        "$seed: '$mpth' does not exist (did you finish stage 16?)" 1>&2
      exit 1
    fi
    ldir="$mdir/hyp/$part/logits"
    mkdir -p "$ldir"
    if [ ! -f "$ldir/.complete" ]; then
      qecho "Beginning stage 17 - Computing logits of '$ddir' for $mname with" \
        "seed $seed..."
      python asr.py \
        --device "$device" $quiet_flag \
        --read-ini "$mconf" \
        logits "$mpth" "$ddir" "$ldir"
      touch "$ldir/.complete"
      qecho "Ending stage 17 - Computed logits of '$ddir' for $mname with" \
        "seed $seed"
    else
      qecho "'$ldir/.complete' already exists, skipping logits computation"
    fi
  done
  ((only)) && exit 0
fi

if [ $stage -le 18 ]; then
  for (( i = OFFSET; i < ${#models[@]} * ${#feats[@]} * seeds * 2 * ${#beam_widths[@]} * ${#lm_betas[@]}; i += STRIDE )); do
    model="${models[((i / (${#feats[@]} * seeds * 2 * ${#beam_widths[@]} * ${#lm_betas[@]}) ))]}"
    ii=$((i % (${#feats[@]} * seeds * 2 * ${#beam_widths[@]} * ${#lm_betas[@]}) ))
    feat="${feats[((ii / (seeds * 2 * ${#beam_widths[@]} * ${#lm_betas[@]}) ))]}"
    ii=$((ii % (seeds * 2 * ${#beam_widths[@]} * ${#lm_betas[@]}) ))
    ((ii / (seeds * ${#beam_widths[@]} * ${#lm_betas[@]}) )) && part=test || part=dev
    ii=$((ii % (seeds * ${#beam_widths[@]} * ${#lm_betas[@]}) ))
    beam_width="${beam_widths[((ii / (seeds * ${#lm_betas[@]}) ))]}"
    ii=$((ii % (seeds * ${#lm_betas[@]}) ))
    lm_beta="${lm_betas[((ii / seeds))]}"
    seed=$((ii % seeds))
    mname="$model-$feat"
    cname="$mname w/ beam width $beam_width, lm_beta $lm_beta, and seed $seed"
    mdir="$exp/$mname/$seed"
    hdir="$mdir/hyp/$part"
    ldir="$hdir/logits"
    if [ ! -f "$ldir/.complete" ]; then
      echo "Cannot decode $cname: '$ldir/.complete' does not exist" \
        "(did you finish stage 17?)" 1>&2
      exit 1
    fi
    bdir="$hdir/$beam_width-$lm_beta"
    mkdir -p "$bdir"
    tfile="$mdir/$part.hyp.$beam_width-$lm_beta.trn" 
    if [ ! -f "$bdir/.complete" ]; then
      qecho "Stage 18 - Decoding '$data/$feat/$part' with $cname..."
      python asr.py \
        --device "$device" $quiet_flag \
        decode \
          "$ldir" "$bdir" \
          --beam-width "$beam_width" \
          --lm-beta "$lm_beta" \
          --lm-pt "$data/$feat/ext/lm.pt"
      rm -f "$tfile" || true  # make sure there isn't a mismatch
      touch "$bdir/.complete"
      qecho "Stage 18 - Decoded '$data/$feat/$part' with $cname"
    else
      qecho "'$bdir/.complete' exists. Skipping decoding '$data/$feat/$part'" \
        "with $cname"
    fi
    if [ ! -f "$tfile" ]; then
      qecho "Beginning stage 18 - gathering hyps for '$tfile' with $cname"
      torch-token-data-dir-to-trn \
        "$bdir" "$data/$feat/ext/id2token.txt" \
        "$mdir/$part.hyp.$beam_width-$lm_beta.utrn"
      python prep/timit.py "$data" filter \
        "$mdir/$part.hyp.$beam_width-$lm_beta."{u,}trn
      qecho "Ending stage 18 - gathered hyps for '$tfile' with $cname"
    else
      qecho "'$tfile' already exists, skipping gathering hyps"
    fi
  done
  ((only)) && exit 0
fi

if [ $stage -le 19 ]; then
  for (( i=OFFSET; i < ${#models[@]} * ${#feats[@]} * seeds; i += STRIDE )); do
    model="${models[((i / (${#feats[@]} * seeds) ))]}"
    ii=$((i % (${#feats[@]} * seeds) ))
    feat="${feats[((ii / seeds))]}"
    seed="$((ii % seeds))"
    mname="$model-$feat"
    mdir="$exp/$mname/$seed"
    for part in 'dev' 'test' ; do
      rfile="$exp/$mname/results.$part.$seed.txt"
      if [ ! -f "$rfile" ]; then
        if [ "$part" = dev ]; then
          active_widths=( "${beam_widths[@]}" )
          active_betas=( "${lm_betas[@]}" )
        else
          active_widths=( "$(awk '
$1 ~ /^best/ {a=gensub(/.*\/dev\.hyp\.([^-]*).*$/, "\\1", 1, $3); print a}
' "$exp/$mname/results.dev.$seed.txt")" )
          active_betas=( "$(awk '
$1 ~ /^best/ {a=gensub(/.*\/dev\.hyp\.[^-]*-(.*)[.]trn.*$/, "\\1", 1, $3); print a}
' "$exp/$mname/results.dev.$seed.txt")" )
        fi
        in_files=( )
        for j in "${!active_widths[@]}"; do
          for k in "${!active_betas[@]}"; do
            in_files+=( "$mdir/$part.hyp.${active_widths[j]}-${active_betas[k]}.trn" )
          done
        done
        if [ "${#in_files[@]}" -eq 0 ]; then
          echo "Stage 20 - Cannot compute error rates for '$rfile': either" \
            "active beam widths (${active_widths[*]}) or active lm betas" \
            "(${active_betas[*]}) is empty" 2>&1
          exit 1
        fi
        qecho "Stage 20 - Computing error rates for '$rfile'..."
        v=0
        python prep/error-rates-from-trn.py \
          "$data/$feat/ext/$part.ref.trn" "${in_files[@]}" \
          --suppress-warning > "$rfile" || v=1
        if ((v)); then
          rm "$rfile"
          exit 1
        fi
        qecho "Stage 20 - Computed error rates for '$rfile'"
      else
        qecho "'$rfile' exists - skipping computing those error rates"
      fi
    done
  done
  ((only)) && exit 0
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

### My results

# lm


# no lm (beta=0.0)
# dev lcorr-fbank-81-10ms: n=20, mean=15.5%, std=0.2%, med=15.8%, min=15.2%, max=15.9%
# dev lcorr-sigbank-41-2ms: n=20, mean=17.5%, std=0.2%, med=17.3%, min=17.1%, max=17.9%
# dev mcorr-fbank-81-10ms: n=20, mean=17.5%, std=0.2%, med=17.5%, min=17.2%, max=17.8%
# dev mcorr-sigbank-41-2ms: n=20, mean=17.3%, std=0.3%, med=17.2%, min=16.9%, max=18.0%
# test lcorr-fbank-81-10ms: n=20, mean=17.8%, std=0.3%, med=17.7%, min=17.2%, max=18.3%
# test lcorr-sigbank-41-2ms: n=20, mean=19.8%, std=0.4%, med=20.2%, min=18.8%, max=20.6%
# test mcorr-fbank-81-10ms: n=20, mean=19.7%, std=0.3%, med=19.5%, min=19.2%, max=20.4%
# test mcorr-sigbank-41-2ms: n=20, mean=19.5%, std=0.5%, med=18.6%, min=18.6%, max=20.2%