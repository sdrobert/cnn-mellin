# cnn-mellin

## Recipe (Bash)

This recipe can be ported to Windows relatively easily.

``` sh
# environment variables used anywhere at all. Put up front for easy
# manipulation

# database variables are probably server-specific.

# postgresql
# db_uname=optuna
# db_server=localhost
# db_name=optim
# db_url=postgresql+pg8000://${db_uname}@${db_server}/${db_name}

db_name=sdrobert_db1
deft_file="$(cd ~; pwd -P)/.my.cnf"
db_url="mysql+pymysql:///${db_name}?read_default_file=${deft_file}"

model_types=( mcorr lcorr )
# Only one representation is necessary for training a model and each
# representation produces models that will not (in general) perform well on
# other representations.
# - fbank-81-10ms:   80+1 Mel-scaled triangular overlapping filters (+energy)
#                    computed with STFTs with a 25ms frame and 10ms frame shift
# - sigbank-41-10ms: 40+1 Mel-scaled Gabor filters (+energy) computed with
#                    short integration every 2ms
# - raw:             Raw audio
# All representations assume 16kHz samples.
feature_types=( fbank-81-10ms sigbank-41-2ms raw )
gpu_mem_limit_sm="$(python -c 'print(6 * (1024 ** 3))')"
num_trials_sm=128
top_k_md=10
gpu_mem_limit_md="$(python -c 'print(9 * (1024 ** 3))')"
top_k_lg=5
gpu_mem_limit_lg="$(python -c 'print(12 * (1024 ** 3))')"
optim_command="sbatch --gres=gpu:t4:1 scripts/cnn_mellin.slrm"
timit_prep_cmd="sbatch --gres=gpu:0 --time=03:00:00 scripts/timit_prep.slrm"
timit_dir="$(cd ~/Databases/TIMIT; pwd -P)"

mkdir -p exp/logs

# Replicate the conda environment + install this
conda env create -f environment.yaml
conda activate cnn-mellin
pip install .

# STEP 1: timit feature/database prep. Consult the script for more info
$timit_prep_cmd "$timit_dir" "${feature_types[@]}"

# STEP 2 (optional): hyperparameter search. Consult below section for more
# info

```

### Hyperparameter optimization recipe

Continues from above at Step 2. We rely on PostgreSQL since the built-in
backend, SQLite, deadlocks for parallel reads with the heartbeat. If you don't
do parallel reads of the database file, you can probably go with SQLite.

``` python
mkdir -p exp/logs

# setting up the database 
# postgresql
# initdb -D exp/${db_name}
# pg_ctl -D exp/${db_name} start
# createuser ${db_uname}
# dropdb ${db_name}  # clear existing results
# createdb --owner=${db_uname} ${db_name}

# mysql
mysql -u ${db_uname} -h ${db_server} -e "create database if not exists ${db_name}"
mysql -u ${db_uname} -h ${db_server} -e "drop database ${db_name}"

# STEP 2.1: create optuna experiments for small model selection.
# We consider any combination of model parameters for about 30 epochs using an
# agressive memory limit.

for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    cnn-mellin \
      --read-ini conf/optim/model-${model_type}-sm.ini \
      --device cuda \
      optim \
        --study-name model-${model_type}-sm-${feature_type} \
        "$db_url" \
        init \
          data/timit/${feature_type}/train \
          --blacklist 'training.*' 'data.*' 'model.convolutional_mellin' \
          --num-data-workers 4 \
          --mem-limit-bytes $gpu_mem_limit_sm \
          --num-trials $num_trials_sm \
          --pruner none
  done
done

# STEP 2.2: run a random hyperparameter search for a while
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    if [[ "$(optuna trials --study-name model-${model_type}-sm-${feature_type} --storage $db_url -f yaml 2> /dev/null | grep -e 'state: COMPLETE' -e 'state: PRUNED' | wc -l)" -ge ${num_trials_sm} ]]; then
      echo "Already done ${num_trials_sm} trials for model-${model_type}-sm-${feature_type}"
      continue
    fi
    $optim_command \
      optim \
        --study-name model-${model_type}-sm-${feature_type} \
        "$db_url" \
        run \
          --sampler random
  done
done

# STEP 2.3: determine the model parameters that optuna rates as most important.
# The rest will take on the setting with the lowest median error rate
# independent on the other settings. Then use those parameters to construct
# new optuna experiments with larger models.

mkdir -p exp/conf
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
      cnn-mellin \
        optim \
          --study-name model-${model_type}-sm-${feature_type} \
          "$db_url" \
          important \
            --top-k=${top_k_lg} \
            exp/conf/model-${model_type}-lg-${feature_type}.params
      cnn-mellin \
        optim \
          --study-name model-${model_type}-sm-${feature_type} \
          "$db_url" \
          best \
            --independent \
            exp/conf/model-${model_type}-lg-${feature_type}.ini
      cnn-mellin \
        --read-ini exp/conf/model-${model_type}-lg-${feature_type}.ini \
        --device cuda \
        optim \
          --study-name model-${model_type}-lg-${feature_type} \
          "$db_url" \
          init \
            data/timit/${feature_type}/train \
            --whitelist $(cat exp/conf/model-${model_type}-lg-${feature_type}.params) \
            --num-data-workers 4 \
            --mem-limit-bytes $gpu_mem_limit_lg
  done
done

# STEP 2.4: Optimize again, this time only with the important model parameters
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    optuna
    $optim_command \
      optim \
        --study-name model-${model_type}-lg-${feature_type} \
        "$db_url" \
        run \
          --sampler tpe
  done
done

# STEP 2.5: Create optuna experiments to determine the most important training
# parameters. We'll use the best small models from above and use the same
# GPU limit
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    cnn-mellin \
      optim \
        --study-name model-${model_type}-sm-${feature_type} \
        "$db_url" \
        best \
          - \
      | sed 's/\(max_.*_mask\)[ ]*=.*/\1 = 10000/g' \
      > exp/conf/train-${model_type}-sm-${feature_type}.ini
    cnn-mellin \
      --read-ini exp/conf/train-${model_type}-sm-${feature_type}.ini \
      --device cuda \
      optim \
        --study-name train-${model_type}-sm-${feature_type} \
        "$db_url" \
        init \
          data/timit/${feature_type}/train \
          --blacklist 'model.*' 'training.max_.*_mask' 'training.num_epochs' 'training.early_.*' \
          --num-data-workers 4 \
          --mem-limit-bytes $gpu_mem_limit_sm
  done
done

# STEP 2.6: run a random hyperparameter search for a while
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    $optim_command \
      optim \
        --study-name train-${model_type}-sm-${feature_type} \
        "$db_url" \
        run \
          --sampler random
  done
done

# STEP 2.7: the importance of hyperparameters from the train-*-sm-* searches
# is too spread out among hyperparameters to go straight into train-*-lg-*.
# Instead, we do a set of train-*-md-* on the top 10.
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
      cnn-mellin \
        optim \
          --study-name train-${model_type}-sm-${feature_type} \
          "$db_url" \
          important \
            --top-k=${top_k_md} \
            exp/conf/train-${model_type}-md-${feature_type}.params
      cnn-mellin \
        optim \
          --study-name train-${model_type}-sm-${feature_type} \
          "$db_url" \
          best \
            --independent \
            exp/conf/train-${model_type}-md-${feature_type}.ini
      cnn-mellin \
        --read-ini exp/conf/train-${model_type}-md-${feature_type}.ini \
        --device cuda \
        optim \
          --study-name train-${model_type}-md-${feature_type} \
          "$db_url" \
          init \
            data/timit/${feature_type}/train \
            --whitelist $(cat exp/conf/train-${model_type}-md-${feature_type}.params) \
            --num-data-workers 4 \
            --mem-limit-bytes $gpu_mem_limit_md
  done
done

# STEP 2.8
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    $optim_command \
      optim \
        --study-name train-${model_type}-md-${feature_type} \
        "$db_url" \
        run \
          --sampler random
  done
done

```