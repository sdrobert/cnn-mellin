#! /usr/bin/env bash

# This is specific to TIMIT-style utterance IDs:
#  <utt_id> = <speaker_id>_<rec_id>

[ -f path.sh ] && . path.sh
echo "$0 $*"

help_message="Split TIMIT torch dir with subset ids in files

Usage: $0 <data-dir> <subset_file_1> [<subset_file_2 [...]]
e.g: $0 data/torch/fbank/train exp/my_config/1/{p1.txt,p2.txt,p3.txt}

This implementation is specific to TIMIT-style torch data directories, where
utterances have ids
  <utt_id> = <speaker_id>_<rec_id>

This script will try to balance each partition with an equal number of
utterances, subject to the constraint that no speaker occurs in more than
one partition.
"

. utils/parse_options.sh

if [ $# -lt 2 ]; then
  echo "${help_message}" | grep "Usage" 1>&2
  echo "$0 --help for more info" 1>&2
  exit 1
fi

data_dir="$1"
shift
N=$(echo "$# - 1" | bc)

set -e

# get-torch-spect-data-dir-info --strict "${data_dir}" /dev/null

tmp="$(mktemp -d)"
trap "rm -rf '${tmp}'" EXIT

# find all the speakers
speakers=( $(find "${data_dir}/feats" -maxdepth 1 -name '*_*.pt' -exec basename "{}" \; | cut -d'_' -f 1 | sort -u) )
M=$(echo "${#speakers[@]} - 1" | bc)
for m in $(seq 0 $M); do
  find "${data_dir}/feats" -maxdepth 1 -name "${speakers[m]}_"'*.pt' -exec basename "{}"  \; | cut -d'.' -f 1 > "${tmp}/s${m}.txt"
done

for n in $(seq 0 $N); do
  > "${tmp}/p${n}.txt"
done

function partitions_by_smallest() {
  find "${tmp}" -name 'p*.txt' -exec wc -l {} \; | \
    sort --reverse | cut -d' ' -f 2-
  return 0
}

function speakers_by_smallest() {
  find "${tmp}" -name 's*.txt' -exec wc -l {} \; | \
    sort --reverse | cut -d' ' -f 2-
  return 0
}

cur_speakers=( $(speakers_by_smallest) )
while [ "${#cur_speakers[@]}" -gt 0 ]; do
  cur_partitions=( $(partitions_by_smallest) )
  for n in $(seq 0 $N); do
    cat "${cur_speakers[n]}" 2> /dev/null >> "${cur_partitions[n]}" || true
  done
  cur_speakers=( "${cur_speakers[@]:N+1}" )
done

for n in $(seq 0 $N); do
  sort "${tmp}/p${n}.txt" > "$1"
  shift
done
