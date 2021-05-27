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
feature_names=( fbank-81-10ms sigbank-41-2ms )
for feature_name in "${feature_names[@]}"; do
  python prep/timit.py data/timit torch_dir phn48 $feature_name \
    --computer-json conf/feats/$feature_name.json
done
python prep/timit.py data/timit torch_dir phn48 raw --raw
feature_names+=( raw )

# STEP 2 (optional): hyperparameter optimization
# Note we're using slurm and sbatch to spawn jobs. You can just as easily
# call cnn-mellin directly, but you'll want to configure a way of a) running
# them in parallel and b) some stopping criterion.
# 
# STEP 2.1: create optuna experiments for model selection
model_types=( mcorr lcorr )
mkdir -p exp/logs
for model_type in "${model_types[@]}"; do
  for feature_name in "${feature_names[@]}"; do
    cnn-mellin \
      --read-ini conf/optim/model-${model_type}-${feature_name}.ini \
      --device cuda \
      optim \
        --study-name model-${model_type}-${feature_name} \
        sqlite:///exp/optim.db \
        init \
          data/timit/${feature_name}/train \
          --blacklist 'training.*' 'data.*' 'model.convolutional_mellin' \
          --num-data-workers 4
  done
done

# STEP 2.2: run a random hyperparameter search for a while
for model_type in "${model_types[@]}"; do
  for feature_name in "${feature_names[@]}"; do
    sbatch -p t4v1 --qos normal --time 10:00:00 scripts/cnn_mellin.slrm  \
      --read-ini conf/optim/model-${model_type}-${feature_name}.ini \
      optim \
        --study-name model-${model_type}-${feature_name} \
        sqlite:///exp/optim.db \
        run \
          --sampler random
  done
done

```