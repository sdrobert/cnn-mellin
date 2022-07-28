#! /usr/bin/env bash

set -e

timit_dir="${1:-"~/Databases/TIMIT"}"
ws=cnn-mellin
env_type=dev  # core or dev
ncpu=1
ngpu=20
run_tests=0
do_hyperparam=1

run_stage() {
  local set_string="--set inputs.stage=$1 display_name=timit-stage-$s"
  shift
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
}

# az ml workspace create -n $ws
# az ml environment create -w $ws -f scripts/azureml/create-environment-$env_type.yaml
# az ml compute create -w $ws -f scripts/azureml/create-cluster-cpu.yaml --set max_instances=$ncpu
# az ml compute create -w $ws -f scripts/azureml/create-cluster-gpu.yaml --set max_instances=$ngpu

if ((run_tests)); then
  # test the cluster, if desired
  if [ $env_type != "dev" ]; then
    echo "env_type must be 'dev' to run tests" 1>&2
    exit 1
  fi
  az ml job create -w $ws -f scripts/azureml/run-pytest.yaml --stream
fi

# az ml data create -w $ws -n timit-ldc --type uri_folder --path "$timit_dir"

# az ml job create -w $ws -f scripts/azureml/timit-stages-1-and-2.yaml --stream

if ((do_hyperparam)); then

  [ -f ./db_creds.sh ] && source db_creds.sh
  if [ -z "$db_url" ]; then
    echo \
      "variable 'db_url' unset. Please create a file db_creds.sh in the" \
      "folder containing timit.sh and set the variable there" 2>&1
    exit 1
  fi

  # sub_id="$(az ad signed-in-user show --query id -o tsv)"
  # vault_name="$(az ml workspace show -n $ws --query "key_vault" -o tsv | awk -F / '{print $NF}')"
  # az keyvault set-policy -n "${vault_name}" --object-id "${sub_id}" --secret-permissions all
  # az keyvault secret set --vault-name "${vault_name}" -n db-url --value="$db_url"

  for s in {3..14..2}; do
    run_stage $s
    run_stage $((s + 1)) -gpu -n $ngpu
  done
fi