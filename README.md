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


```