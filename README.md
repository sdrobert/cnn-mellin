# cnn-mellin

## Installation

To match the environment which this package was developed with, use the command

``` sh
conda env create -f environment.yaml
```

Though no further steps are necessary to run the code, the code will run more
quickly if the C++/CUDA on-the-fly extensions can be built. Merely set the
`CUDA_HOME` environment variable to that of CUDA v11.3. If a different version
of CUDA is necessary, change the version of `cudatoolkit` in
`environment.yaml`.

## Recipe

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


## The Mellin C++/CUDA library

The Mellin library is header-only, meaning it doesn't need compilation by
itself. The files already written in `ext` should suffice.
However, if you wish to: a) test the C++ interface; b) change the default
algorithm of the interface; or c) benchmark the various algorithms, you can use
CMake to compile the project. We assume the conda environment has been created
and `CUDA_HOME` set as above.

``` sh
CUDACXX="${CUDA_HOME}/bin/nvcc" \
cmake -B build c -G Ninja \
  "-DCMAKE_INSTALL_PREFIX=ext" \
  "-DCUDAToolkit_ROOT=${CUDA_HOME}" \
  "-DCMAKE_BUILD_TYPE=Release" \
  "-DMELLIN_BUILD_CUDA=ON" \
  "-DCMAKE_PROJECT_VERSION=$(python -c 'from setuptools_scm import get_version; print(get_version().split(".dev")[0])')" \
  "-DMELLIN_SKIP_PERFORMANCE_BUILDS=OFF"
cmake --build build --target install --config Release
```

The above installs the library to `ext/{include,lib}`. For the purposes of the
python scripts, the `lib` subdirectory can be safely deleted. The Mellin
library is combined with the Torch wrapper files `ext/torch*` and compiled
on-the-fly with
[JIT](https://pytorch.org/tutorials/advanced/cpp_extension.html#jit-compiling-extensions)
when available.
