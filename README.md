# cnn-mellin

## Running on a dedicated instance with Bash

These instructions are applicable in the usual case where you have access to
some dedicated machine (such as your PC or a cloud service you have to spin up
or down). We also assume you can run a Bash script, which usually means Linux.

To run the main steps (training + decoding), you need a Conda environment
with some "core" libraries installed:

``` sh
CONDA_OVERRIDE_CUDA=11.3 conda env create -f conf/python/environment-core.yaml
```

Though no further steps are necessary to run the code, the code will run *much*
more quickly if the C++/CUDA on-the-fly extensions can be built. Merely set the
`CUDA_HOME` environment variable to tell PyTorch where to look, e.g.

``` sh
export CUDA_HOME="$(cd "$(which nvcc)/../.."; pwd -P)"
```

Instructions for how to install the toolkit on Linux can be found
[here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
You'll want to install the 11.3 toolkit.

If your CUDA driver doesn't support version 11.3, you want to use a later
version, or whatever, you'll have to modify two things at least. First, you'll
have to change the version of `cudatoolkit` in the `environment-core.yaml`
file. Second, set the `CONDA_OVERRIDE_CUDA` flag to the appropriate (minor)
version. Note that PyTorch is only compiled for some CUDA versions per release;
you might not get the combination you're looking for. Try:

``` sh
conda search -c pytorch pytorch
```

If you want the full environment used for development -- needed only if you
plan on doing anything in the [Advanced section](#advanced) -- you should
configure the environment with `environment-dev.yaml` instead.

Assuming the installation went well, running the TIMIT recipe from start to
finish is straightforward:

``` sh
conda activate cnn-mellin
./timit.sh -i /path/to/ldc/timit
```

Where `/path/to/ldc/timit` is the path which you downloaded
[TIMIT](https://github.com/sdrobert/pytorch-database-prep/wiki/The-TIMIT-Corpus)
to from the LDC. We can't do this for you because of its license/paywall. Once
the recipe is completed, the script spits out descriptive statistics about the
various experimental conditions' error rates. The script keeps track of what's
been done so far, so you can kill/resume the script with little consequence.

The two major downsides to this approach are: a) you must use Bash; and b) each
trial (`feature type x model type x seed`) is run in serial, which can take a
long time.

## Recipe details

The script `timit.sh` wraps calls to `python prep/timit.py` and `python asr.py`
with some additional glue to keep track of where everything is being saved, the
stages completed, and the matrix of trial. It is loosely based off the `run.sh`
script of [Kaldi
recipes](https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/run.sh).
Call

``` sh
./timit.sh -h
```

to list the options. Options like `-n`, `-k`, `-f`, and `-m` control which
trials of the `feature x model x seed` matrix are run, avoiding spurious
computation if you're not interested in descriptives or certain model
combinations.

Like Kaldi, the recipe is organized into a series of commands. Larger recipes
such as
[librispeech](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/run.sh)
organize blocks of commands into stages to allow for easy resumption (though
you have to modify the value of the `stage` variable inside `run.sh` manually
to do so). `timit.sh` exposes that functionality via the option `-s`. For
example,

``` sh
./timit.sh -s 4
```

runs the recipe from stage 4. A stage usually has the pseudocode

``` text
for each matrix combination {
   if "done" file for combination does not exist {
      call either asr.py or prep/timit.py with combination;
      create "done" file for combination;
   }
}
```

where the "done" file is some kind of file which only exists upon the
completion of a specific combination of the appropriate matrix. It is usually
just an empty file. If you know that the stage's python call has finished
without errors

## Running via a scheduling system

### Slurm

### Azure ML

1. Create an Azure account and a resource group you want to perform experiments
   in. By default the recipe uses 100 low-priority NC-series cores and only
   4 dedicated CPU cores. You might have to request quota increases.
2. [Install the
   CLI](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public).
3. Configure the CLI with the desired subscription and resource groups as
   defaults:

   ``` sh
   az account set --subscription "XX"
   az config set "defaults.group=YY"
   ```

   This will allow the recipe to submit jobs and create resources without
   specifying the environment explicitly.
2. Set up the environment. Run

   ``` sh
   az ml environment create --file scripts/azureml/create-environment-core.yaml
   ```

   This creates an environment with all the packages in
   `environment-core.yaml`.
3. Set up the compute clusters. Run

   ``` sh
   az ml compute create --file scripts/azureml/create-cluster-cpu.yaml
   az ml compute create --file scripts/azureml/create-cluster-gpu.yaml
   ```

  This creates one dedicated, CPU-only cluster with a single node responsible
  for doing the steps between any major work and a low-priority, GPU cluster
  for doing the rest.
4. (Optional) Double-check you've got the configuration working by running
   pytest jobs. Run

   ``` sh
   az ml job create --file scripts/azureml/run-pytest.yaml
   ```

   The GPU component might be pre-empted - don't worry about that. Just make
   sure there aren't any other errors.

5. Set up the TIMIT dataset. TIMIT is licensed by the LDC so we can't do this
   step automatically.

   ``` sh
   az ml data create --name timit-ldc --type uri_folder --path /uri/to/timit
   ```

## Advanced

### The Mellin C++/CUDA library

The Mellin library is header-only, meaning it doesn't need compilation by
itself. The files already written in `ext` should suffice. However, if you wish
to: a) test the C++ interface; b) change the default algorithm of the
interface; or c) benchmark the various algorithms, you can use CMake to compile
the project. We assume the dev environment has been created as opposed to the
core environment (see
[Installation](#running-on-a-dedicated-instance-with-bash)).

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

### Hyperparameter search

#### Azure

Like in the other cases, you'll need access to additional packages listed in
`environment-dev.yaml`. Those packages should be installed in the environment:

``` sh
az ml environment create --file scripts/azureml/create-environment-dev.yaml
```

You've got two ways of handling `db_url` in Azure: the easy, less secure way,
and the harder, more secure way.

The easy way is to update `scripts/azureml/timit_wrapper.sh` and hard-code
your URL in there. Of course, then the URL (including the password) is just
sitting in the cloud in plain text.

The second way is to add the URL as a secret to the workspace's Key Vault.
First, you have to give permission for whatever account's logged in to the CLI
to muck with secrets. Follow the instructions on [this
page](https://go.microsoft.com/fwlink/?linkid=2125287) (note: you only need to
grant permissions for secrets). If you don't know the name of the associated
Key Vault, find it with the command

``` sh
az ml workspace show  --query "key_vault" -o tsv
```

Then you can navigate to the "Secrets" tab and add `db-url` as a secret
(note the `-` instead of the `_`). Alternatively, you can add it via the
CLI:

``` sh
vault_name="$(az ml workspace show --query "key_vault" -o tsv | awk  -F '/' '{print $NF}')"
az keyvault secret set --vault-name "${vault_name}" -n db-url --value "$(source db_creds.sh; echo $db_url)"
```

## License

Apache 2.0