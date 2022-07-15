cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()
project(Cdrobert_Mellin_Cuda LANGUAGES CXX CUDA)

### Configuration
option(
  MELLIN_CUDA_SERIAL
  "Perform all operations with block 1 thread 1 (good for debugging)"
  OFF
)
set(
  MCONV1D_CUDA_ALGORITHM_VERSION 1 CACHE STRING
  "Version of algorithm to use for 1D mellin convolutions (CUDA)")
set(
  MCORR1D_CUDA_ALGORITHM_VERSION 1 CACHE STRING
  "Version of algorithm to use for 1D mellin correlations (CUDA)")
set(
  MCONVLCONV_CUDA_ALGORITHM_VERSION 1 CACHE STRING
  "Version of algorithm to use for mellin-linear convolutions (CUDA)")
set(
  SND2COL_CUDA_ALGORITHM_VERSION 1 CACHE STRING
  "Version of algorithm to use for snd2col (CUDA)")
set(
    COL2SND_CUDA_ALGORITHM_VERSION 1 CACHE STRING
    "Version of algorithm to use for col2snd (CUDA)")
set(
  SPEC2COL_CUDA_ALGORITHM_VERSION 1 CACHE STRING
  "Version of the algorithm to use for spec2col (CUDA)"
)
set(
  COL2SPEC_CUDA_ALGORITHM_VERSION 1 CACHE STRING
  "Version of the algorithm to use for col2spec (CUDA)"
)    
set(
  MCORRLCORR_CUDA_ALGORITHM_VERSION 1 CACHE STRING
  "Version of algorithm to use for mellin-linear correlations (CUDA)")
set(
  MELLIN_MAX_CUDA_THREADS_PER_BLOCK 1024 CACHE STRING
  "Maximum number of threads to spawn in a mellin CUDA kernel"
)

if (MELLIN_CUDA_SERIAL)
  set(MELLIN_CUDA_SERIAL true)
else (MELLIN_CUDA_SERIAL)
  set(MELLIN_CUDA_SERIAL false)
endif (MELLIN_CUDA_SERIAL)
configure_file(
  config/cdrobert/mellin/config_cuda.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cdrobert/mellin/config_cuda.h
  @ONLY
)

### Library
add_library(mellin_cuda INTERFACE
  include/cdrobert/mellin/mconv_cuda.cuh
)
add_library(Cdrobert::mellin_cuda ALIAS mellin_cuda)
target_compile_features(mellin_cuda
  INTERFACE
    cuda_std_11
)
target_include_directories(mellin_cuda
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

### Installation
install(
  TARGETS mellin_cuda
  EXPORT Cdrobert_MellinTargets
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
