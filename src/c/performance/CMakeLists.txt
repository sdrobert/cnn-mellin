cmake_minimum_required(VERSION 3.19)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()
project(Cdrobert_Mellin_Performance LANGUAGES CXX CUDA VERSION 0.0.0)

set(BLA_VENDOR Intel10_64_dyn)
find_package(BLAS)
if (BLAS_FOUND)
  find_path(MKL_INCLUDE_DIR mkl.h)
endif()

#add_executable(mconv_performance mconv_performance.cpp)
#target_link_libraries(mconv_performance Cdrobert::mellin_cpu)
#add_executable(mconv_cuda_performance mconv_cuda_performance.cu)
#target_link_libraries(mconv_cuda_performance Cdrobert::mellin_cuda)

set_source_files_properties(OptionParser.cpp PROPERTIES LANGUAGE CUDA)
add_library(OptionParser
  OptionParser.cpp
)
target_include_directories(OptionParser
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if (TARGET Cdrobert::mellin_cuda)
  add_executable(mconv_cuda_profiling mconv_cuda_profiling.cu)
  target_link_libraries(mconv_cuda_profiling
    PRIVATE
      Cdrobert::mellin
      OptionParser
  )
  target_compile_features(mconv_cuda_profiling PRIVATE cuda_std_11)
  find_package(CUDAToolkit)
  target_link_libraries(mconv_cuda_profiling PRIVATE CUDA::cublas)
endif()

add_executable(mconv_profiling mconv_profiling.cpp)
target_link_libraries(mconv_profiling PRIVATE Cdrobert::mellin_cpu OptionParser)
if (MKL_INCLUDE_DIR)
  target_include_directories(mconv_profiling PRIVATE ${MKL_INCLUDE_DIR})
  target_link_libraries(mconv_profiling PRIVATE BLAS::BLAS)
  target_compile_options(mconv_profiling PRIVATE -DHAVE_MKL)
endif()