#! /usr/bin/env bash

set -e

timit_dir="${1:-"~/Databases/TIMIT"}"
ws=cnn-mellin  # workspace name
env_type=dev  # core or dev
ncpu=1  # maximum number of CPU nodes in the GPU cluster
ngpu=1  # maximum number of GPU nodes in the GPU cluster
do_hyperparam=1  # do hyperparameter search (optional, requires dev)

run_stage() {
  local stage="$(printf "%02d" "$1")"
  if [ -f "exp/timit/completed_stages/$stage" ]; then
    echo "'exp/timit/completed_stages/$stage' exists. Skipping stage $1"
    return 0
  fi
  shift
  local set_string="--set inputs.stage=$stage display_name=timit-stage-$stage"
  while [ $# -gt 0 ]; do
    case "$1" in
      -n)
        shift;
        set_string="${set_string} resources.instance_count=$1";
        shift;;
      -gpu)
        set_string="${set_string} compute=azureml:cnn-mellin-gpu";
        shift;;
    esac
  done
  az ml job create -w $ws -f scripts/azureml/timit-stage-n.yaml --stream $set_string
  touch "exp/timit/completed_stages/$stage"
}

mkdir -p exp/timit/completed_stages

if [ ! -f "exp/timit/completed_stages/00" ]; then
  az ml workspace create -n $ws
  az ml environment create -w $ws -f scripts/azureml/create-environment-$env_type.yaml
  az ml compute create -w $ws -f scripts/azureml/create-cluster-cpu.yaml --set max_instances=$ncpu
  az ml compute create -w $ws -f scripts/azureml/create-cluster-gpu.yaml --set max_instances=$ngpu
  az ml data create -w $ws -n timit-ldc --type uri_folder --path "$timit_dir"
  touch exp/timit/completed_stages/00
else
  echo "'exp/timit/completed_stages/00' exists. Skipping Azure setup"
fi

if [ ! -f "exp/timit/completed_stages/01_02" ]; then
  az ml job create -w $ws -f scripts/azureml/timit-stages-1-and-2.yaml --stream
  touch exp/timit/completed_stages/01_02
else
  echo "'exp/timit/completed_stages/01_02' exists. Skipping stages 1 and 2"
fi

if ((do_hyperparam)); then

  [ -f ./db_creds.sh ] && source db_creds.sh
  if [ -z "$db_url" ]; then
    echo \
      "variable 'db_url' unset. Please create a file db_creds.sh in the" \
      "folder containing timit.sh and set the variable there" 2>&1
    exit 1
  fi

  if [ ! -f "exp/timit/completed_stages/create_db_secret" ]; then
    sub_id="$(az ad signed-in-user show --query id -o tsv)"
    vault_name="$(az ml workspace show -n $ws --query "key_vault" -o tsv | awk -F / '{print $NF}')"
    az keyvault set-policy -n "${vault_name}" --object-id "${sub_id}" --secret-permissions all
    az keyvault secret set --vault-name "${vault_name}" -n db-url --value="$db_url"
    touch exp/timit/completed_stages/create_db_secret
  fi

  for s in {3..13..2}; do
    run_stage $s
    run_stage $((s + 1)) -gpu -n $ngpu
  done
fi