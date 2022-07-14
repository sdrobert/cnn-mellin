cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

option(FULL_TEST
  "Perform full test suite (false does not check transpose, different batch sizes)"
  OFF)

# WARNING(sdrobert): cmake has to be re-run any time these test files change!
foreach(conv_type mconv1d lconv1d mconvlconv mcorr1d lcorr1d mcorrlcorr snd2col col2snd lin2col col2lin spec2col col2spec)
  file(READ "test_${conv_type}_buffers.h" "${conv_type}_file")
  string(REGEX REPLACE
    ".*kNumTests *= *([0-9]+).*" "\\1"
    "num_${conv_type}_tests" "${${conv_type}_file}")
endforeach(conv_type)
math(EXPR
  num_mconvlconv_tests
  "${num_mconv1d_tests} + ${num_lconv1d_tests} + ${num_mconvlconv_tests}"
)
math(EXPR
  num_mcorrlcorr_tests
  "${num_mcorr1d_tests} + ${num_lcorr1d_tests} + ${num_mcorrlcorr_tests}"
)
math(EXPR
  num_spec2col_tests
  "${num_snd2col_tests} + ${num_lin2col_tests} + ${num_spec2col_tests}"
)
math(EXPR
  num_col2spec_tests
  "${num_col2snd_tests} + ${num_col2lin_tests} + ${num_col2spec_tests}"
)


if (FULL_TEST)
  set(batch_sizes 1 2 10)
  set(transpose_fs T F)
  set(transpose_gs T F)
  set(transpose_hs T F)
  set(extra_hs -1 0 10)
  set(c_ins 1 128)
  set(c_outs 1 256)
  set(nfs 3 10)
  set(ngs 10 300)
  set(nhs 10 40)
  set(ss 1 2)
  set(ds 1 3)
  set(ps 0 5)
  set(us 1 7)
else (FULL_TEST)
  set(batch_sizes 2)
  set(transpose_fs F)
  set(transpose_gs F)
  set(transpose_hs F)
  set(extra_hs 10)
  set(nfs 3)
  set(ngs 10)
  set(nhs 10)
  set(c_ins 10)
  set(c_outs 10)
  set(ss 1)
  set(ds 1)
  set(ps 0)
  set(us 1)
endif (FULL_TEST)

add_executable(test_mconv1d_cpu test_mconv1d.cc)
target_link_libraries(test_mconv1d_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_mconv1d_cpu PRIVATE cxx_std_11)
add_executable(test_mcorr1d_cpu test_mcorr1d.cc)
target_link_libraries(test_mcorr1d_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_mcorr1d_cpu PRIVATE cxx_std_11)
add_executable(test_mconvlconv_cpu test_mconvlconv.cc)
target_link_libraries(test_mconvlconv_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_mconvlconv_cpu PRIVATE cxx_std_11)
add_executable(test_mcorrlcorr_cpu test_mcorrlcorr.cc)
target_link_libraries(test_mcorrlcorr_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_mcorrlcorr_cpu PRIVATE cxx_std_11)
add_executable(test_snd2col_cpu test_snd2col.cc)
target_link_libraries(test_snd2col_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_snd2col_cpu PRIVATE cxx_std_11)
add_executable(test_col2snd_cpu test_col2snd.cc)
target_link_libraries(test_col2snd_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_col2snd_cpu PRIVATE cxx_std_11)
add_executable(test_spec2col_cpu test_spec2col.cc)
target_link_libraries(test_spec2col_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_spec2col_cpu PRIVATE cxx_std_11)
add_executable(test_col2spec_cpu test_col2spec.cc)
target_link_libraries(test_col2spec_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_col2spec_cpu PRIVATE cxx_std_11)
add_executable(test_batching_cpu test_batching.cc)
target_link_libraries(test_batching_cpu PRIVATE Cdrobert::mellin_cpu)
target_compile_features(test_batching_cpu PRIVATE cxx_std_11)

if (TARGET Cdrobert::mellin_cuda)
  enable_language(CUDA)
  # FIXME(sdrobert): this is a dumb hack until CMake supports linking CUDA
  # libraries with the C++ linker. Technically, these files and the interface
  # are pure C++. See
  # https://gitlab.kitware.com/cmake/cmake/issues/17816
  set_source_files_properties(
    test_mconv1d_cuda.cc test_mcorr1d_cuda.cc test_mconvlconv_cuda.cc
    test_mcorrlcorr_cuda.cc test_snd2col_cuda.cc test_col2snd_cuda.cc
    test_spec2col_cuda.cc test_col2spec_cuda.cc test_batching_cuda.cc
    PROPERTIES LANGUAGE CUDA
  )
  add_executable(test_mconv1d_cuda test_mconv1d_cuda.cc)
  target_link_libraries(test_mconv1d_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_mconv1d_cuda PRIVATE cuda_std_11)
  add_executable(test_mcorr1d_cuda test_mcorr1d_cuda.cc)
  target_link_libraries(test_mcorr1d_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_mcorr1d_cuda PRIVATE cuda_std_11)
  add_executable(test_mconvlconv_cuda test_mconvlconv_cuda.cc)
  target_link_libraries(test_mconvlconv_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_mconvlconv_cuda PRIVATE cuda_std_11)
  add_executable(test_mcorrlcorr_cuda test_mcorrlcorr_cuda.cc)
  target_link_libraries(test_mcorrlcorr_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_mcorrlcorr_cuda PRIVATE cuda_std_11)
  add_executable(test_snd2col_cuda test_snd2col_cuda.cc)
  target_link_libraries(test_snd2col_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_snd2col_cuda PRIVATE cuda_std_11)
  add_executable(test_col2snd_cuda test_col2snd_cuda.cc)
  target_link_libraries(test_col2snd_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_col2snd_cuda PRIVATE cuda_std_11)
  add_executable(test_spec2col_cuda test_spec2col_cuda.cc)
  target_link_libraries(test_spec2col_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_spec2col_cuda PRIVATE cuda_std_11)
  add_executable(test_col2spec_cuda test_col2spec_cuda.cc)
  target_link_libraries(test_col2spec_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_col2spec_cuda PRIVATE cuda_std_11)
  add_executable(test_batching_cuda test_batching_cuda.cc)
  target_link_libraries(test_batching_cuda PRIVATE Cdrobert::mellin_cuda Cdrobert::mellin_cpu)
  target_compile_features(test_batching_cuda PRIVATE cuda_std_11)

  set (TEST_DEVICES Cpu Cuda)
else()
  set (TEST_DEVICES Cpu)
endif()

enable_testing()
foreach(device ${TEST_DEVICES})
  string(TOLOWER "${device}" x)
  foreach(conv_type MConv1D MCorr1D MConvLConv MCorrLCorr Snd2Col Col2Snd Spec2Col Col2Spec)
    string(TOLOWER "${conv_type}" y)
    math(EXPR N "${num_${y}_tests} - 1")
      foreach(batch_size ${batch_sizes})
        foreach(transpose_f ${transpose_fs})
          foreach(transpose_g ${transpose_gs})
            foreach(transpose_h ${transpose_hs})

              if (NOT (${conv_type} STREQUAL Snd2Col OR ${conv_type} STREQUAL Col2Snd OR ${conv_type} STREQUAL Spec2Col OR ${conv_type} STREQUAL Col2Spec))
                foreach(c_in ${c_ins})
                  foreach(c_out ${c_outs})
                    foreach(nf ${nfs})
                      foreach(ng ${ngs})
                        foreach(nh ${nhs})
                          foreach(s ${ss})
                            foreach(d ${ds})
                              foreach(p ${ps})
                                foreach(u ${us})

                                  if (${conv_type} STREQUAL MConv1D OR ${conv_type} STREQUAL MCorr1D)
                                    add_test(
                                      NAME
                                        Cdrobert.Mellin.${device}.${conv_type}_batching.${c_out},${c_in},${batch_size},${nf},${ng},${nh},${s},${d},${p},${u},${transpose_f},${transpose_g},${transpose_h}
                                      COMMAND
                                        $<TARGET_FILE:test_batching_${x}>
                                          ${c_out} ${c_in} ${batch_size} ${nf} ${ng} ${nh} ${s} ${d} ${p} ${u}
                                          ${transpose_f} ${transpose_g} ${transpose_h} $<IF:$<STREQUAL:${conv_type},MConv1D>,t,f>
                                    )
                                  else()
                                    
                                    add_test(
                                      NAME
                                        Cdrobert.Mellin.${device}.${conv_type}_batching.${c_out},${c_in},${batch_size},${nf},${ng},${nh},${s},${d},${p},${u},${transpose_f},${transpose_g},${transpose_h}
                                      COMMAND
                                        $<TARGET_FILE:test_batching_${x}>
                                          ${c_out} ${c_in} ${batch_size} ${nf} ${ng} ${nh} ${s} ${d} ${p} ${u}
                                          ${transpose_f} ${transpose_g} ${transpose_h} $<IF:$<STREQUAL:${conv_type},MConvLConv>,t,f>
                                          ${nf} ${ng} ${nh} ${s} ${d} ${p} ${u}
                                    )
              
                                  endif()

                                endforeach(u)
                              endforeach(p)
                            endforeach(d)
                          endforeach(s)
                        endforeach(nh)
                      endforeach(ng)
                    endforeach(nf)
                  endforeach(c_out)
                endforeach(c_in)
              endif()

              foreach(extra_h ${extra_hs})
                foreach (test_num RANGE 0 ${N})
                  add_test(
                      NAME
                        Cdrobert.Mellin.${device}.${conv_type}.${test_num}_${batch_size},${transpose_f},${transpose_g},${transpose_h},${extra_h}
                      COMMAND
                        $<TARGET_FILE:test_${y}_${x}>
                          ${test_num} ${batch_size}
                          ${transpose_f} ${transpose_g} ${transpose_h}
                          ${extra_h}
                      )
                endforeach (test_num)
              endforeach(extra_h)

            endforeach(transpose_h)
          endforeach(transpose_g)
        endforeach(transpose_f)
      endforeach(batch_size)
  endforeach(conv_type)
endforeach(device)
