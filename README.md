# cnn-mellin

## Instructions

``` sh
# Replicate the conda environment + install this
conda env create -f environment.yaml
conda activate cnn-mellin
pip install .

# Assuming TIMIT with fbank features. Easily swap for WSJ, GGWS and/or raw. See
# https://github.com/sdrobert/pytorch-database-prep
python prep/timit.py data preamble /path/to/timit
python prep/timit.py data init_phn
python prep/timit.py data torch_dir

mkdir -p exp/logs
cnn-mellin \
  --read-ini conf/model_lcorr_optim.ini \
  --device cuda \
  optim \
    --study-name lcorr-model-fbank \
    sqlite:///exp/optim.db \
    init \
      data/train \
      --blacklist 'training.*' 'data.*' 'model.convolutional_mellin'

cnn-mellin \
  --device cuda \
  optim \
    --study-name lcorr-model-fbank \
    sqlite:///exp/optim.db \
    run \
      --sampler random \
  > >(tee -a exp/logs/lcorr-model-fbank-random.out) \
  2> >(tee -a exp/logs/lcorr-model-fbank-random.err >&2)

cnn-mellin \
  --device cuda \
  optim \
    --study-name lcorr-model-fbank \
    sqlite:///exp/optim.db \
    run \
      --sampler tpe \
  > >(tee -a exp/logs/lcorr-model-fbank-tpe.out) \
  2> >(tee -a exp/logs/lcorr-model-fbank-tpe.err >&2)

cnn-mellin \
  --read-ini conf/model_mcorr_optim.ini \
  --device cuda \
  optim \
    --study-name mcorr-model-fbank \
    sqlite:///exp/optim.db \
    init \
      data/train \
      --blacklist 'training.*' 'data.*' 'model.convolutional_mellin'

cnn-mellin \
  --device cuda \
  optim \
    --study-name mcorr-model-fbank \
    sqlite:///exp/optim.db \
    run \
      --sampler random \
  > >(tee -a exp/logs/mcorr-model-fbank-random.out) \
  2> >(tee -a exp/logs/mcorr-model-fbank-random.err >&2)

cnn-mellin \
  --device cuda \
  optim \
    --study-name mcorr-model-fbank \
    sqlite:///exp/optim.db \
    run \
      --sampler tpe \
  > >(tee -a exp/logs/mcorr-model-fbank-tpe.out) \
  2> >(tee -a exp/logs/mcorr-model-fbank-tpe.err >&2)
```