# cnn-mellin

## Recipe (Bash)

This recipe can be ported to Windows relatively easily. If you plan on running
more than one Optuna instance simultaneously, you might want to change the
backend from SQLite to something else if you're using an NFS file system
([it doesn't have file locks](https://www.sqlite.org/faq.html#q5)).

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

Continues from above at Step 2

``` python
# Note we're using slurm and sbatch to spawn jobs. You can just as easily
# call cnn-mellin directly, but you'll want to configure a way of a) running
# them in parallel and b) some stopping criterion.
# 
# STEP 2.1: create optuna experiments for small model selection.
# We consider any combination of model parameters for about 30 epochs using an
# agressive 
mkdir -p exp/logs
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    cnn-mellin \
      --read-ini conf/optim/model-${model_type}-sm.ini \
      --device cuda \
      optim \
        --study-name model-${model_type}-sm-${feature_type} \
        sqlite:///exp/optim.db \
        init \
          data/timit/${feature_type}/train \
          --blacklist 'training.*' 'data.*' 'model.convolutional_mellin' \
          --num-data-workers 4 \
          --mem-limit-bytes "$(python -c 'print(6 * (1024 ** 3))')"
  done
done

# STEP 2.2: run a random hyperparameter search for a while
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    # sbatch -p t4v1 --qos normal --time 48:00:00 scripts/cnn_mellin.slrm
    ./scripts/cnn_mellin.slrm \
      optim \
        --study-name model-${model_type}-sm-${feature_type} \
        sqlite:///exp/optim.db \
        run \
          --sampler random
  done
done

# STEP 2.4: write best back to file to get ready for training optimization
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    cnn-mellin \
      optim \
        --study-name model-${model_type}-${feature_type} \
        sqlite:///exp/optim.db \
        best \
          conf/optim/train-${model_type}-${feature_type}.ini
  done
done

# STEP 2.5: create optuna experiments for training optimization
# We don't optimize the learning rate b/c it's adaptive w/ adam
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    cnn-mellin \
      --read-ini conf/optim/train-${model_type}-${feature_type}.ini \
      --device cuda \
      optim \
        --study-name train-${model_type}-${feature_type} \
        sqlite:///exp/optim.db \
        init \
          data/timit/${feature_type}/train \
          --num-data-workers 4 \
          --whitelist \
            'training.dropout_prob' \
            'training.convolutional_dropout_2d' \
            'training.max_.+' \
            'training.noise_eps' \
            'training.num_(time|freq)_.*'
  done
done

# STEP 2.6: tpe
for model_type in "${model_types[@]}"; do
  for feature_type in "${feature_types[@]}"; do
    sbatch -p t4v1 --qos normal --time 24:00:00 scripts/cnn_mellin.slrm  \
      optim \
        --study-name train-${model_type}-${feature_type} \
        sqlite:///exp/optim.db \
        run \
          --sampler tpe
  done
done
```