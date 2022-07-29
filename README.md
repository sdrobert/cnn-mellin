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

The `timit.sh` recipe was designed so that the user may run a single stage and
exit. This is convenient for job schedulers like
[Slurm](https://slurm.schedmd.com/documentation.html) or [Azure Machine
Learning](https://azure.microsoft.com/en-ca/services/machine-learning/), both
of which run tasks on a remote instance with a pre-defined environment
asynchronously. The idea is to write a script which runs on a dedicated
instance that tells the scheduler to run a step, waits for the step to finish,
and then moves on to the next. Parallelization occurs within-stage accross the
configuration matrix. Consult the start of `./timit.sh` for more info on how
this is done.

### Slurm

1. Set up the Conda environment as above.
2. Run the script

   ``` sh
   ./scripts/slurm/timit.sh /path/to/ldc/timit \
      'sbatch options for cpu tasks' \
      'sbatch options for gpu tasks'
   ```

   where options include things like setting the partition, QOS, and a
   maximum time if necesary. Consult `man sbatch` for more info. Do not muck
   with the options for parallelization such as `--array` or `--num-tasks`
   as those are handled by the script. Also, `--gres=gpu:1` is automatically
   added for gpu tasks.

### Azure ML

1. Create an Azure account and a resource group you want to perform experiments
   in. By default the recipe uses 120 low-priority NC-series cores and only 4
   dedicated CPU cores. You might have to request quota increases. You can
   safely decrease the number of low-priority cores to 6 (the size of a single
   NC6 instance) by decreasing the value of `ngpu` in
   `scripts/azureml/timit.sh`, though an N-fold decrease will take roughly N
   times as long. Any more than 120 cores won't be used.
2. [Install the
   CLI](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public).
   *Note:* you don't need a local installation of the Conda environment for
   Azure since the environment is installed in the cloud.
3. Configure the CLI with the desired subscription and resource groups as
   defaults:

   ``` sh
   az account set --subscription "XX"
   az config set "defaults.group=YY"
   ```

   This will allow the recipe to submit jobs and create resources without
   specifying the environment explicitly.
4. If you have all the quotas necessary, you can just run

   ``` sh
   ./scripts/azureml/timit.sh /path/to/ldc/timit
   ```

   and finish successfully. This creates a workspace called `cnn-mellin` which
   you should delete when you're done with.

## Advanced

### The Mellin C++/CUDA library

The Mellin library is header-only, meaning it doesn't need compilation by
itself. The files already written in `ext` should suffice. However, if you wish
to: a) test the C++ interface; b) change the default algorithm of the
interface; or c) benchmark the various algorithms, you can use CMake to compile
the project. We assume the dev environment has been created as opposed to the
core environment.

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

The hyperparameter search has already been performed with the best resulting
architecture configurations stored in `conf/models`. If you want to redo the
search, the first thing you'll need is access to some Relational DataBase (RDB)
to store results in. Unless you plan on performing the search serially (*not*
recommended), you will need a proper RDB - [not
SQLite](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html?highlight=deadlock).

Once you've configured the RDB and started it, you'll need to define the URL
used by SQLAlchemy to access the RDB. Follow the instructions
[here](https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls).
Different RDBs used different drivers. We've put `PostgreSQL + psycopg2`
in the dev environment - you can update these as needed. Once you've figured
out your URL, create the file `db_creds.sh` in the root of this project and
fill the variable `db_url` with the url:

``` sh
# db_creds.sh
db_url='my:database@url/here'
```

Speaking of the dev environment: that will need to be installed instead. If you
are using Azure, set `env_type=dev` in `scripts/azureml/timit.sh`.

You will also need to keep the recipe from skipping over the steps related to
the hyperparameter search. For dedicated running, just add the `-q` flag to the
call to `./timit.sh`. If you're running on a scheduler, you'll need to update
the flag `do_hyperparam=1` in the appropriate recipe
`scripts/{azureml,slurm}/timit.sh`.

## License

Apache 2.0