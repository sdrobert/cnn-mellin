# cnn-mellin

## Recipe (Bash)

This recipe can be ported to Windows relatively easily.

``` sh
# Replicate the conda environment + install this
conda env create -f environment.yaml
conda activate cnn-mellin
pip install .

# STEP 1.1: initial dataset prep
# Assuming TIMIT with fbank features. Easily swap for WSJ, GGWS. See
# https://github.com/sdrobert/pytorch-database-prep
python prep/timit.py data/timit preamble /path/to/timit
python prep/timit.py data/timit init_phn --lm

# STEP 1.2: construct feature representations
# Only one representation is necessary for training a model and each
# representation produces models that will not (in general) perform well on
# other representations.
# - fbank-81-10ms:   80+1 Mel-scaled triangular overlapping filters (+energy)
#                    computed with STFTs with a 25ms frame and 10ms frame shift
# - sigbank-41-10ms: 40+1 Mel-scaled Gabor filters (+energy) computed with
#                    short integration every 2ms
# - raw:             Raw audio
# All representations assume 16kHz samples.
feature_types=( fbank-81-10ms sigbank-41-2ms )
for feature_type in "${feature_types[@]}"; do
  python prep/timit.py data/timit torch_dir phn48 $feature_type \
    --computer-json conf/feats/$feature_type.json
done
python prep/timit.py data/timit torch_dir phn48 raw --raw
feature_types+=( raw )

model_types=( mcorr lcorr )

# STEP 2 (optional): hyperparameter optimization (see subsection below)
```

### Hyperparameter optimization recipe

Continues from above at Step 2. We rely on PostgreSQL since the built-in
backend, SQLite, deadlocks for parallel reads with the heartbeat. If you don't
do parallel reads of the database file, you can probably go with SQLite.

``` python
# setting up the database
db_url=postgresql+pg8000://optuna@localhost/optim
mkdir -p exp/logs
initdb -D exp/optim
pg_ctl -D exp/optim -l exp/logs/optim.log start
createuser optuna
# dropdb optim  # clear existing results
createdb --owner=optuna optim


# Note we're using slurm and sbatch to spawn jobs. You can just as easily
# call cnn-mellin directly, but you'll want to configure a way of a) running
# them in parallel and b) some stopping criterion.
# 
# STEP 2.1: create optuna experiments for small model selection.
# We consider any combination of model parameters for about 30 epochs using an
# agressive memory limit.
gpu_mem_limit_sm="$(python -c 'print(6 * (1024 ** 3))')"
num_trials_sm=128
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
    sbatch -p t4v1 --qos normal scripts/cnn_mellin.slrm \
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
top_k_lg=5
gpu_mem_limit_lg="$(python -c 'print(12 * (1024 ** 3))')"
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
    sbatch -p t4v1 --qos normal scripts/cnn_mellin.slrm \
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
    sbatch -p t4v1 --qos normal scripts/cnn_mellin.slrm \
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
top_k_md=10
gpu_mem_limit_md="$(python -c 'print(9 * (1024 ** 3))')"
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
    sbatch -p t4v1 --qos normal scripts/cnn_mellin.slrm \
      optim \
        --study-name train-${model_type}-md-${feature_type} \
        "$db_url" \
        run \
          --sampler random
  done
done

```