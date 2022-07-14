// Copyright 2018 Sean Robertson
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cuda.h"
#include "test_mconv1d_buffers.h"
#include "test_lconv1d_buffers.h"
#include "test_mconvlconv_buffers.h"
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
        mconv1db::kNumTests + lconv1db::kNumTests + mconvlconvb::kNumTests
      ) || test_idx < 0) {
    std::cerr << "Invalid test idx: " << test_idx << std::endl;
    return 1;
  }

  int nfx, nfy, ngx, ngy, c_out, c_in, sx, sy, dx, dy, px, py, ux, uy;
  const char *name;
  const double *F, *G, *H;
  if (test_idx < mconv1db::kNumTests) {
    name = "mconv1db";
    nfx = mconv1db::kNF[test_idx]; nfy = 1;
    ngx = mconv1db::kNG[test_idx]; ngy = 1;
    c_out = mconv1db::kCOut[test_idx]; c_in = mconv1db::kCIn[test_idx];
    sx = mconv1db::kS[test_idx]; sy = 1;
    dx = mconv1db::kD[test_idx]; dy = 1;
    px = mconv1db::kP[test_idx]; py = 0;
    ux = mconv1db::kU[test_idx]; uy = 1;
    F = mconv1db::kF[test_idx];
    G = mconv1db::kG[test_idx];
    H = mconv1db::kH[test_idx];
  } else if (test_idx < mconv1db::kNumTests + lconv1db::kNumTests) {
    name = "lconv1db";
    test_idx -= mconv1db::kNumTests;
    nfx = 1; nfy = lconv1db::kNF[test_idx];
    ngx = 1; ngy = lconv1db::kNG[test_idx];
    c_out = lconv1db::kCOut[test_idx]; c_in = lconv1db::kCIn[test_idx];
    sx = 1; sy = lconv1db::kS[test_idx];
    dx = 1; dy = lconv1db::kD[test_idx];
    px = 0; py = lconv1db::kP[test_idx];
    ux = 1; uy = lconv1db::kU[test_idx];
    F = lconv1db::kF[test_idx];
    G = lconv1db::kG[test_idx];
    H = lconv1db::kH[test_idx];
  } else {
    name = "mconvlconvb";
    test_idx -= mconv1db::kNumTests + lconv1db::kNumTests;
    nfx = mconvlconvb::kNFX[test_idx]; nfy = mconvlconvb::kNFY[test_idx];
    ngx = mconvlconvb::kNGX[test_idx]; ngy = mconvlconvb::kNGY[test_idx];
    c_out = mconvlconvb::kCOut[test_idx]; c_in = mconvlconvb::kCIn[test_idx];
    sx = mconvlconvb::kSX[test_idx]; sy = mconvlconvb::kSY[test_idx];
    dx = mconvlconvb::kDX[test_idx]; dy = mconvlconvb::kDY[test_idx];
    px = mconvlconvb::kPX[test_idx]; py = mconvlconvb::kPY[test_idx];
    ux = mconvlconvb::kUX[test_idx]; uy = mconvlconvb::kUX[test_idx];
    F = mconvlconvb::kF[test_idx];
    G = mconvlconvb::kG[test_idx];
    H = mconvlconvb::kH[test_idx];
  }

  int nhx = cdrobert::mellin::MConvSupportSize(nfx, ngx, sx, dx, px, ux);
  int nhy = cdrobert::mellin::LConvSupportSize(nfy, ngy, sy, dy, py, uy);
  double *f, *g, *h, h_exp[10000];
  CheckCuda(cudaMalloc(&f, sizeof(double) * c_out * c_in * nfx * nfy));
  CheckCuda(cudaMalloc(&g, sizeof(double) * batch * c_in * ngx * ngy));
  CheckCuda(cudaMallocManaged(&h, sizeof(double) * batch * c_out * (nhx + extra_h) * (nhy + extra_h)));
  CheckCuda(cudaMemset(h, 0., sizeof(double) * batch * c_out * (nhx + extra_h) * (nhy + extra_h)));
  CopyFCuda(F, c_in, c_out, nfx * nfy, transposed_f, f);
  CopyGCuda(G, batch, c_in, ngx * ngy, transposed_g, g);
  // do h_exp copy while cuda is running the kernel

  cdrobert::mellin::MConvLConvCuda(
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
