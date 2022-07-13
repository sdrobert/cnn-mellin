# cnn-mellin

## Recipe (Bash)

This recipe can be ported to Windows relatively easily.

``` sh
# environment variables used anywhere at all. Put up front for easy
# manipulation

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
feature_types=( fbank-81-10ms sigbank-41-2ms )
timit_prep_cmd="sbatch --gres=gpu:0 --time=03:00:00 scripts/timit_prep.slrm"
timit_dir="$(cd ~/Databases/TIMIT; pwd -P)"

mkdir -p exp/logs

# Replicate the conda environment + install this
conda env create -f environment.yaml
conda activate cnn-mellin
pip install .

# STEP 1: timit feature/database prep. Consult the script for more info
$timit_prep_cmd "$timit_dir" "${feature_types[@]}"

# STEP 2 (optional): hyperparameter search. Consult script for more info
scripts/hyperparam.sh
```
