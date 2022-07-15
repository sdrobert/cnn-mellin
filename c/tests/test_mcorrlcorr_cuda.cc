// Copyright 2018 Sean Robertson
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cuda.h"
#include "test_mcorr1d_buffers.h"
#include "test_lcorr1d_buffers.h"
#include "test_mcorrlcorr_buffers.h"
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
  if (
      test_idx >= (
        mcorr1db::kNumTests + lcorr1db::kNumTests + mcorrlcorrb::kNumTests
      ) || test_idx < 0) {
    std::cerr << "Invalid test idx: " << test_idx << std::endl;
    return 1;
  }

  int nfx, nfy, ngx, ngy, c_out, c_in, sx, sy, dx, dy, px, py, ux, uy;
  const char *name;
  const double *F, *G, *H;
  if (test_idx < mcorr1db::kNumTests) {
    name = "mcorr1db";
    nfx = mcorr1db::kNF[test_idx]; nfy = 1;
    ngx = mcorr1db::kNG[test_idx]; ngy = 1;
    c_out = mcorr1db::kCOut[test_idx]; c_in = mcorr1db::kCIn[test_idx];
    sx = mcorr1db::kS[test_idx]; sy = 1;
    dx = mcorr1db::kD[test_idx]; dy = 1;
    px = mcorr1db::kP[test_idx]; py = 0;
    ux = mcorr1db::kU[test_idx]; uy = 1;
    F = mcorr1db::kF[test_idx];
    G = mcorr1db::kG[test_idx];
    H = mcorr1db::kH[test_idx];
  } else if (test_idx < mcorr1db::kNumTests + lcorr1db::kNumTests) {
    name = "lcorr1db";
    test_idx -= mcorr1db::kNumTests;
    nfx = 1; nfy = lcorr1db::kNF[test_idx];
    ngx = 1; ngy = lcorr1db::kNG[test_idx];
    c_out = lcorr1db::kCOut[test_idx]; c_in = lcorr1db::kCIn[test_idx];
    sx = 1; sy = lcorr1db::kS[test_idx];
    dx = 1; dy = lcorr1db::kD[test_idx];
    px = 0; py = lcorr1db::kP[test_idx];
    ux = 1; uy = lcorr1db::kU[test_idx];
    F = lcorr1db::kF[test_idx];
    G = lcorr1db::kG[test_idx];
    H = lcorr1db::kH[test_idx];
  } else {
    name = "mcorrlcorrb";
    test_idx -= mcorr1db::kNumTests + lcorr1db::kNumTests;
    nfx = mcorrlcorrb::kNFX[test_idx]; nfy = mcorrlcorrb::kNFY[test_idx];
    ngx = mcorrlcorrb::kNGX[test_idx]; ngy = mcorrlcorrb::kNGY[test_idx];
    c_out = mcorrlcorrb::kCOut[test_idx]; c_in = mcorrlcorrb::kCIn[test_idx];
    sx = mcorrlcorrb::kSX[test_idx]; sy = mcorrlcorrb::kSY[test_idx];
    dx = mcorrlcorrb::kDX[test_idx]; dy = mcorrlcorrb::kDY[test_idx];
    px = mcorrlcorrb::kPX[test_idx]; py = mcorrlcorrb::kPY[test_idx];
    ux = mcorrlcorrb::kUX[test_idx]; uy = mcorrlcorrb::kUY[test_idx];
    F = mcorrlcorrb::kF[test_idx];
    G = mcorrlcorrb::kG[test_idx];
    H = mcorrlcorrb::kH[test_idx];
  }

  int nhx = cdrobert::mellin::MCorrSupportSize(ngx, sx, dx, px, ux);
  int nhy = cdrobert::mellin::LCorrSupportSize(ngy, sy, py, uy);
  double *f, *g, *h, h_exp[10000];
  CheckCuda(cudaMalloc(&f, sizeof(double) * c_out * c_in * nfx * nfy));
  CheckCuda(cudaMalloc(&g, sizeof(double) * batch * c_in * ngx * ngy));
  CheckCuda(cudaMallocManaged(&h, sizeof(double) * batch * c_out * (nhx + extra_h) * (nhy + extra_h)));
  CheckCuda(cudaMemset(h, 0., sizeof(double) * batch * c_out * (nhx + extra_h) * (nhy + extra_h)));
  CopyFCuda(F, c_in, c_out, nfx * nfy, transposed_f, f);
  CopyGCuda(G, batch, c_in, ngx * ngy, transposed_g, g);
  // do h_exp copy while cuda is running the kernel

  cdrobert::mellin::MCorrLCorrCuda(
    f, g, c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx + extra_h, nhy + extra_h,
    sx, sy, dx, dy, px, py, ux, uy,
    transposed_f, transposed_g, transposed_h, h
  );
  CopyHCpu(H, batch, c_out, nhx, nhy, extra_h, extra_h, transposed_h, h_exp);

  cudaDeviceSynchronize();
  CheckCuda(cudaGetLastError());

  CheckCuda(cudaFree(f));
  CheckCuda(cudaFree(g));

  std::vector<int> shape {batch, c_out, nhx + extra_h, nhy + extra_h};
  if (transposed_h) shape = {c_out, batch, nhx + extra_h, nhy + extra_h};
  if (AllClose(h_exp, h, shape, 1e-10)) {
    std::cerr << "Failed test: " << test_idx << " (" << name << ")"
              << std::endl;
    CheckCuda(cudaFree(h));
    return 1;
  }
  CheckCuda(cudaFree(h));
  return 0;
}
