// Copyright 2021 Sean Robertson
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cpu.h"
#include "test_utils_cuda.h"
#include "test_col2snd_buffers.h"
#include "cdrobert/mellin/mconv.h"
#include "cdrobert/mellin/mconv_cuda.cuh"

int main(int argc, const char* argv[]) {
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    return 1;
  }

  int test_idx = 0, batch = 0, extra_h = 0;
  bool transposed_f = false, transposed_g = false, transposed_h = false;
  if (ParseArgs(argc, argv,
                &test_idx, &batch,
                &transposed_f, &transposed_g, &transposed_h,
                &extra_h)) {
    return 1;
  }
  if (test_idx >= col2sndb::kNumTests || test_idx < 0) {
    std::cerr << "Invalid test idx: " << test_idx << std::endl;
    return 1;
  }

  int nf = col2sndb::kNF[test_idx], ng = col2sndb::kNG[test_idx],
      c_in = col2sndb::kCIn[test_idx],
      s = col2sndb::kS[test_idx], d = col2sndb::kD[test_idx],
      p = col2sndb::kP[test_idx], u = col2sndb::kU[test_idx];
  int nh = cdrobert::mellin::MCorrSupportSize(ng, s, d, p, u);
  double *g, *h, g_exp[10000];
  // N.B. extra_h doesn't make sense in the negative direction as it will
  // change the values of g (fewer terms to accumulate)
  extra_h = std::max(extra_h, 0);
  CheckCuda(cudaMallocManaged(&g, sizeof(double) * batch * c_in * ng));  // device
  CheckCuda(cudaMalloc(&h, sizeof(double) * batch * c_in * nf * (nh + extra_h))); // unif
  CopyHCuda(col2sndb::kH[test_idx], batch, c_in, nf, nh, 0, extra_h,
           transposed_h, h);
  std::fill_n(g, batch * c_in * ng, -12345.);

  // do g_exp copy while cuda is running the kernel
  CopyGOrHCpu(col2sndb::kG[test_idx], batch, c_in, 1, ng, 0, 0, transposed_g, g_exp);

  cdrobert::mellin::Col2SndCuda(
    h,
    c_in, batch, nf, ng, nh + extra_h, s, d, p, u,
    transposed_g, transposed_h, g
  );

  cudaDeviceSynchronize();
  CheckCuda(cudaGetLastError());

  CheckCuda(cudaFree(h));

  std::vector<int> shape {batch, c_in, ng};
  if (transposed_g) shape = {c_in, batch, ng};
  if (AllClose(g_exp, g, shape, 1e-10)) {
    std::cerr << "Failed test: " << test_idx << std::endl;
    CheckCuda(cudaFree(g));
    return 1;
  }
  CheckCuda(cudaFree(g));
  return 0;
}