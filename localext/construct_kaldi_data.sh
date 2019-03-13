#! /usr/bin/env bash

[ -f path.sh ] && . path.sh
echo "$0 $*"

pybank_json=
cmvn_style=utt
log_dir=exp/logs/make_kaldi_data
help_message="Generate kaldi data dir from another kaldi data dir

Usage: $0 [options] <orig-data-dir> <new-data-dir> [<new-raw-dir>]
e.g. 1: $0 ../s5/data data/kaldi/fbank
e.g. 2: $0 --pybank-conf conf/gbank_41.json ../s5/data data/kaldi/gbank

<new-data-dir> is the front-facing feature directory, including feats.scp.
<new-raw-dir> is where the actual features will be dumped. Defaults to
<new-data-dir>/raw.

This script makes it simple to use a recipe's collected corpus info, such
as transcripts and wave files, to create a new, compatible data dir with
different features. By default, this script creates an f-bank using Kaldi's
implementation, but if --pybank-json is set, it will the python package
pydrobert-speech to generate features according to some configuration

Options
--pybank-json PATH                : If set, use pydrobert-speech's feature
                                    generation, configured with this config
                                    file (json). Otherwise, use kaldi fbanks
--cmvn-style {speaker,utt,none}   : If 'speaker', perform Cepstral Mean
                                    Variance normalization at the speaker level
                                    (speaker dependent). If 'utt', perform
                                    CMVN at the utterance level (speaker
                                    independent). If 'none', do not perform
                                    CMVN (deft: ${cmvn_style})
--log-dir DIR                     : Where to write logs of creation process
"

. parse_options.sh

if [ $# != 2 ] && [ $# != 3 ]; then
  echo "${help_message}" | grep "Usage" 1>&2
  echo "$0 --help for more info" 1>&2
  exit 1
fi

src_dir="$1"
dest_dir="$2"
raw_dir="${3:-$2/raw}"

set -e

mkdir -p "${dest_dir}" "${raw_dir}" "${log_dir}"
for x in train dev test; do
  if [ ! -d "${src_dir}/$x" ]; then
    echo "Expected '${src_dir}/$x' to exist!" 1>&2
    exit 1
  fi
  mkdir -p "${dest_dir}/$x" "${log_dir}/$x"
  cp "${src_dir}/$x/"{glm,stm,text,wav.scp} "${dest_dir}/$x/"
  if [ "${cmvn_style}" = "speaker" ]; then
    n_utts="$(cat "${src_dir}/$x/utt2spk" | wc -l)"
    n_spks="$(cat "${src_dir}/$x/spk2utt" | wc -l)"
    if [ "${n_utts}" -le "${n_spks}" ]; then
      echo "\
Specified that speaker-dependent CMVN desired, but n source dir
'${src_dir}/$x' the number of utterances in utt2spk (${n_utts}) is less than
or equal to the number of speakers (${n_spks}). Unless something weird is
going on, '${src_dir}/$x' is likely speaker-independent. Will continue, though
" 1>&2
      sleep 10
    fi
    cp "${src_dir}/$x/"{spk2utt,utt2spk} "${dest_dir}/$x/"
  else
    paste <(cut -d' ' -f 1 "${src_dir}/$x/utt2spk") \
          <(cut -d' ' -f 1 "${src_dir}/$x/utt2spk") > "${dest_dir}/$x/utt2spk"
    utils/utt2spk_to_spk2utt.pl "${dest_dir}/$x/utt2spk" \
      > "${dest_dir}/$x/spk2utt"
  fi
  if [ -z "${pybank_json}" ]; then
    steps/make_fbank.sh "${dest_dir}/$x" "${log_dir}/$x" "${raw_dir}"
  else
    stepsext/make_pybank.sh \
      --pybank-json "${pybank_json}" \
      "${dest_dir}/$x" "${log_dir}/$x" "${raw_dir}"
  fi
  if [ "${cmvn}" != "none" ]; then
    steps/compute_cmvn_stats.sh "${dest_dir}/$x" "${log_dir}/$x" "${raw_dir}"
  fi
  utils/validate_data_dir.sh "${dest_dir}/$x"
done
