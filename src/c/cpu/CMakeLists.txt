cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(Cdrobert_Mellin_Cpu LANGUAGES CXX)

### Configuration
set(
  MCONV1D_ALGORITHM_VERSION 2 CACHE STRING
  "Version of algorithm to use for 1D mellin convolutions (CPU)")
set(
  MCORR1D_ALGORITHM_VERSION 3 CACHE STRING
  "Version of algorithm to use for 1D mellin correlations (CPU)")
set(
  MCONVLCONV_ALGORITHM_VERSION 1 CACHE STRING
  "Version of algorithm to use for mellin-linear convolutions (CPU)")
set(
  MCORRLCORR_ALGORITHM_VERSION 1 CACHE STRING
  "Version of algorithm to use for mellin-linear correlations (CPU)")
set(
  SND2COL_ALGORITHM_VERSION 1 CACHE STRING
  "Version of the algorithm to use for snd2col (CPU)"
)
set(
  COL2SND_ALGORITHM_VERSION 1 CACHE STRING
  "Version of the algorithm to use for col2snd (CPU)"
)
set(
  SPEC2COL_ALGORITHM_VERSION 1 CACHE STRING
  "Version of the algorithm to use for spec2col (CPU)"
)
set(
  COL2SPEC_ALGORITHM_VERSION 1 CACHE STRING
  "Version of the algorithm to use for col2spec (CPU)"
)

## Generate configuration header
configure_file(
  config/cdrobert/mellin/config_cpu.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/cdrobert/mellin/config_cpu.h
  @ONLY
)

### Library
add_library(mellin_cpu INTERFACE
  include/cdrobert/mellin/mconv.h
)
add_library(Cdrobert::mellin_cpu ALIAS mellin_cpu)
target_compile_features(mellin_cpu
  INTERFACE
    cxx_alias_templates
    cxx_variadic_macros
    cxx_lambda_init_captures
    cxx_right_angle_brackets
)
target_include_directories(mellin_cpu
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(mellin_cpu
  PROPERTIES
  OUTPUT_NAME mellin
)
## Use openmp if we've got it. If we don't, we don't.
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(mellin_cpu INTERFACE OpenMP::OpenMP_CXX)
endif()

### Installation
install(
  TARGETS mellin_cpu
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
