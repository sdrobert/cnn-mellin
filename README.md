# cnn-mellin

## Installation

To match the environment which this package was developed with, use the command

``` sh
CONDA_OVERRIDE_CUDA=11.3 conda env create -f environment.yaml
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
have to change the version of `cudatoolkit` in the `environment.yaml` file.
Second, set the `CONDA_OVERRIDE_CUDA` flag to the appropriate (minor) version.
Note that PyTorch is only compiled for some CUDA versions per release; you
might not get the combination you're looking for. Try:

``` sh
conda search -c pytorch pytorch
```

## The Mellin C++/CUDA library

The Mellin library is header-only, meaning it doesn't need compilation by
itself. The files already written in `ext` should suffice. However, if you wish
to: a) test the C++ interface; b) change the default algorithm of the
interface; or c) benchmark the various algorithms, you can use CMake to compile
the project. We assume the conda environment has been created and `CUDA_HOME`
set as above.

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

## Azure

Be careful that there aren't any extra files lying around in this directory or
they'll end up getting copied over to the server each time a job is run.

### TIMIT

1. Perform initial setup of the [Azure ML
   workspace](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources).
   You'll also need to [install the
   CLI](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public).
   **Note** every CLI command (`az`) in each subsequent step should include
   the subscription name, resource group, and workspace you've created:

   ``` sh
   az ml ... \
      --subscription my-subscription-name \
      --resource-group my-resource-group \
      --workspace-name my-workspace-name \
      ...
   ```

   we exclude them from below for brevity's sake.
2. Set up the environment. Run

   ``` sh
   az ml environment create --file conf/azureml/create-environment.yaml
   ```

   This creates an environment with all the packages in `environment.yaml`.
3. Set up the compute clusters. Run

   ``` sh
   az ml compute create --file conf/azureml/create-cluster-cpu.yaml
   az ml compute create --file conf/azureml/create-cluster-gpu.yaml
   ```

  This creates one dedicated, CPU-only cluster with a single node responsible
  for doing the steps between any major work and a low-priority, GPU cluster
  for doing the rest.
4. (Optional) Double-check you've got the configuration working by running
   pytest jobs. Run

   ``` sh
   az ml job create --file conf/azureml/run-pytest.yaml
   ```

5. Set up the TIMIT dataset. TIMIT is licensed by the LDC so we can't do this
   step automatically.

   ``` sh
   az ml data create --name timit-ldc --type uri_folder --path /uri/to/timit
   ```

