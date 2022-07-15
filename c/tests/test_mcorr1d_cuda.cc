// Copyright 2018 Sean Robertson
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cuda.h"
#include "test_mcorr1d_buffers.h"
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
  if (test_idx >= mcorr1db::kNumTests || test_idx < 0) {
    std::cerr << "Invalid test idx: " << test_idx << std::endl;
    return 1;
  }

  int nf = mcorr1db::kNF[test_idx], ng = mcorr1db::kNG[test_idx],
      c_out = mcorr1db::kCOut[test_idx], c_in = mcorr1db::kCIn[test_idx],
      s = mcorr1db::kS[test_idx], d = mcorr1db::kD[test_idx],
      p = mcorr1db::kP[test_idx], u = mcorr1db::kU[test_idx];
  int nh = cdrobert::mellin::MCorrSupportSize(ng, s, d, p, u);
  double *f, *g, *h, h_exp[10000];
  CheckCuda(cudaMalloc(&f, sizeof(double) * c_out * c_in * nf));  // device
  CheckCuda(cudaMalloc(&g, sizeof(double) * batch * c_in * ng));  // device
  CheckCuda(cudaMallocManaged(&h, sizeof(double) * batch * c_out * (nh + extra_h))); // unif
  CheckCuda(cudaMemset(h, 0., sizeof(double) * batch * c_out * (nh + extra_h)));
  CopyFCuda(mcorr1db::kF[test_idx], c_in, c_out, nf, transposed_f, f);
  CopyGCuda(mcorr1db::kG[test_idx], batch, c_in, ng, transposed_g, g);
  // do h_exp copy while cuda is running the kernel

  cdrobert::mellin::MCorr1DCuda(
    f, g, c_out, c_in, batch, nf, ng, nh + extra_h, s, d, p, u,
    transposed_f, transposed_g, transposed_h, h
  );
  CopyHCpu(mcorr1db::kH[test_idx], batch, c_out, 1, nh, 0, extra_h,
           transposed_h, h_exp);

  cudaDeviceSynchronize();
  CheckCuda(cudaGetLastError());

  CheckCuda(cudaFree(f));
  CheckCuda(cudaFree(g));

  std::vector<int> shape {batch, c_out, nh + extra_h};
  if (transposed_h) shape = {c_out, batch, nh + extra_h};
  if (AllClose(h_exp, h, shape, 1e-10)) {
    std::cerr << "Failed test: " << test_idx << std::endl;
    CheckCuda(cudaFree(h));
    return 1;
  }
  CheckCuda(cudaFree(h));
  return 0;
}
