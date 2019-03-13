#! /usr/bin/env bash

[ -f path.sh ] && . path.sh

echo "$0 $*"

num_deltas=0
recipe_dir=../s5
train_ali_dir=exp/dnn4_pretrain-dbn_dnn_ali
dev_ali_dir=exp/dnn4_pretrain-dbn_dnn_ali_dev
gmm_mdl=exp/dnn4_pretrain-dbn_dnn/final.mdl
HCLG=exp/tri3/graph/HCLG.fst
words=exp/tri3/graph/words.txt
help_message="Construct torch data dir from feats and alignments

Usage: $0 [options] <kaldi-feat-dir> <torch-data-dir>
e.g.: $0 --num-deltas 2 ../s5/data data/torch/mfcc

Combine features and alignments into a torch data dir, for use in training
a neural acoustic model. This assumes that the cnn_mellin module and all of
its core dependencies have been installed in the active python environment.
The first argument should contain subfolders train/, dev/, and test/ with
their own feats.scp files. The second is the output folder

Options:
--num-deltas INT      : The number of deltas to add to the features before
                        writing (deft: ${num_deltas})
--recipe-dir PATH     : Sets the relative root of the --train-ali,--dev-ali,
                        --gmm-mdl, and --HCLG options (deft: ${recipe_dir})
--train-ali-dir PATH  : The directory of training alignments, absolute or
                        relative to --recipe-dir. Contains files ali.*.gz
                        (deft: ${train_ali_dir})
--dev-ali-dir PATH    : The directory of development alignments, absolute or
                        relative to --recipe-dir. Contains files ali.*.gz
                        (deft: ${dev_ali_dir})
--gmm-mdl PATH        : The path to a GMM-based topology file (final.mdl),
                        absolute or relative to --recipe-dir (deft: ${gmm_mdl})
--HCLG PATH           : The path to a decoding weighted FST file (HCLG.fst),
                        absolute or relative to --recipe-dir (deft: ${HCLG})
--words PATH          : The path to the word-to-id dictionary (words.txt),
                        absolute or relative to --recipe-dir (deft: ${words})
"

. parse_options.sh

if [ $# != 2 ]; then
  echo "${help_message}" | grep "Usage" 1>&2
  echo "$0 --help for more info" 1>&2
  exit 1
fi

kaldi_data_dir="$1"
torch_data_dir="$2"

set -e

tmp="$(mktemp -d)"
trap "rm -rf '${tmp}'" EXIT

## sanitation & verification

for x in train dev test ; do
  if [ ! -d "${kaldi_data_dir}/$x" ]; then
    echo -e "Directory '${kaldi_data_dir}/$x' does not exist"
    exit 1
  fi
done
kaldi_data_dir="$(cd "${kaldi_data_dir}"; pwd -P)"

all_recipe_files_absolute=true
for f in "${train_ali_dir}" "${dev_ali_dir}" "${gmm_mdl}" "${HCLG}" "${words}"; do
  if [[ "$f" = /* ]]; then
    if [ ! -e "$f" ]; then
      echo -e "File '$f' does not exist"
      exit 1
    fi
  else
    all_recipe_files_absolute=false
  fi
done

if ! $all_recipe_files_absolute ; then
  if [ ! -d "${recipe_dir}" ]; then
    echo -e "\
One or more files are relative to --recipe-dir, but '${recipe_dir}' does not
exist"
    exit 1
  fi
  for f in "${train_ali_dir}" "${dev_ali_dir}" "${gmm_mdl}" "${HCLG}"; do
    if ! [[ "$f" = /* ]] && [ ! -e "${recipe_dir}/$f" ]; then
      echo -e "File '${recipe_dir}/$f' does not exist"
      exit 1
    fi
  done
  recipe_dir="$(cd "${recipe_dir}"; pwd -P)"
  train_ali_dir="$(cd "${recipe_dir}"; cd "${train_ali_dir}"; pwd -P)"
  dev_ali_dir="$(cd "${recipe_dir}"; cd "${dev_ali_dir}"; pwd -P)"
  gmm_mdl="$(cd "${recipe_dir}"; cd "$(dirname "${gmm_mdl}")"; pwd -P)/$(basename "${gmm_mdl}")"
  HCLG="$(cd "${recipe_dir}"; cd "$(dirname "${HCLG}")"; pwd -P)/$(basename "${HCLG}")"
  words="$(cd "${recipe_dir}"; cd "$(dirname "${words}")"; pwd -P)/$(basename "${words}")"
fi

for d in "${train_ali_dir}" "${dev_ali_dir}"; do
  if [ -f "$d/num_jobs" ]; then
    num_ali="$(cat "$d/num_jobs")"
    expected_files=( $(seq 1 "${num_ali}" | awk '{printf("'"${d}"'/ali.%d.gz\n",$1)}' | sort) )
    actual_files=( "${d}"/ali.*.gz )
    if [ "${#expected_files[@]}" -gt "${#actual_files[@]}" ]; then
      num_files="${#expected_files[@]}"
    else
      num_files="${#actual_files[@]}"
    fi
    for idx in $(seq 1 ${num_files}); do
      expected_file="${expected_files[idx-1]}"
      actual_file="${actual_files[idx-1]}"
      if [ -z "${actual_file}" ] ; then
        echo "\
The file '${expected_file}' was expected to exist (as per num_jobs), but does
not" 1>&2
        exit 1
      elif ! [ "${expected_file}" = "${actual_file}" ]; then
        echo "\
An additional alignment file '${actual_file}' was found that does not coincide
with the expected number of alignment files (listed in num_jobs)" 1>&2
        exit 1
      fi
    done
  fi
done

copy-int-vector 'ark:gunzip -c "'"${train_ali_dir}"'/ali."*.gz "'"${dev_ali_dir}"'/ali."*.gz |' ark,t:- | awk '{print $1,NF-1" "}' | sort > "${tmp}/ali_lens.txt"
feat-to-len 'scp:cat "'"${kaldi_data_dir}"'/train/feats.scp" "'"${kaldi_data_dir}"'/dev/feats.scp" |' "ark,t:| sort > ${tmp}/feat_lens.txt"
if ! cmp -s "${tmp}/feat_lens.txt" "${tmp}/ali_lens.txt" ; then
  echo "Feature and alignment lengths differ. Here's an example: " 1>&2
  echo "feats                                        alignments" 1>&2
  diff --suppress-common-lines -W 80 -y "${tmp}/feat_lens.txt" "${tmp}/ali_lens.txt" | head -n 10 1>&2
  exit 1
fi

## creating in temp

for x in train dev test ; do
  mkdir -p "${tmp}/torch/$x/feats"
  feats_rxspecifier="scp:${kaldi_data_dir}/$x/feats.scp"
  if [ -f "${kaldi_data_dir}/$x/cmvn.scp" ]; then
    if [ -f "${kaldi_data_dir}/$x/utt2spk" ]; then
      cmvn_extra_flags="--utt2spk='ark:${kaldi_data_dir}/$x/utt2spk'"
    else
      cmvn_extra_flags=""
    fi
    feats_rxspecifier="ark:apply-cmvn ${cmvn_extra_flags} 'scp:${kaldi_data_dir}/$x/cmvn.scp' '${feats_rxspecifier}' ark:- |"
  fi
  if [ "${num_deltas}" -gt 0 ]; then
    feats_rxspecifier="ark:add-deltas --order=${num_deltas} '${feats_rxspecifier}' ark:-"
  fi
  write-table-to-torch-dir "${feats_rxspecifier}" "${tmp}/torch/$x/feats"
done

get-torch-spect-data-dir-info --strict "${tmp}/torch/test" /dev/null

for x in train dev ; do
  mkdir -p "${tmp}/torch/$x/ali"
  ali_dir="$(eval "echo \${${x}_ali_dir}")"
  write-table-to-torch-dir \
    -i iv -o long \
    "ark:ali-to-pdf '${gmm_mdl}' 'ark:gunzip -c "${ali_dir}"/ali.*.gz |' ark:- |" \
    "${tmp}/torch/$x/ali"
  if [ "$x" = "train" ]; then
    get-torch-spect-data-dir-info \
      --strict "${tmp}/torch/$x" "${tmp}/info.ark"
  else
    get-torch-spect-data-dir-info --strict "${tmp}/torch/$x" /dev/null
  fi
done

# this ensures that we don't mess up everything in this directory if we re-run
# the recipe in the other directory (whoops)
cp "${gmm_mdl}" "${tmp}/torch/final.mdl"
cp "${HCLG}" "${tmp}/torch/HCLG.fst"
cp "${words}" "${tmp}/torch/words.txt"

target_dim=$(hmm-info "${gmm_mdl}" | awk '/pdfs/ {print $4}')
freq_dim=$(awk '$1 == "num_filts" {print $2}' "${tmp}/info.ark")
target-count-info-to-tensor \
  --num-targets ${target_dim} \
  "${tmp}/info.ark" inv_weight "${tmp}/torch/weights.pt"
target-count-info-to-tensor \
  --num-targets ${target_dim} \
  "${tmp}/info.ark" log_prior "${tmp}/torch/log_prior.pt"

# at this point, we have to make the target directory so we know its location
# on disk (and get rid of final forward slash, if there)
mkdir -p "${torch_data_dir}"
torch_data_dir="$(cd "${torch_data_dir}"; pwd -P)"

cat << EOF > "${tmp}/torch/variables"
target_dim=${target_dim}
freq_dim=${freq_dim}
HCLG='${torch_data_dir}/HCLG.fst'
gmm_mdl='${torch_data_dir}/final.mdl'
words='${torch_data_dir}/words.txt'
weight_file='${torch_data_dir}/weights.pt'
log_prior='${torch_data_dir}/log_prior.pt'
train_data='${torch_data_dir}/train'
dev_data='${torch_data_dir}/dev'
test_data='${torch_data_dir}/test'
dev_ref='${kaldi_data_dir}/dev'
test_ref='${kaldi_data_dir}/test'
EOF

# copying
rsync -r "${tmp}/torch/" "${torch_data_dir}"
