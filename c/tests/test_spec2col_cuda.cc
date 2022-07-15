// Copyright 2021 Sean Robertson
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cpu.h"
#include "test_utils_cuda.h"
#include "test_snd2col_buffers.h"
#include "test_lin2col_buffers.h"
#include "test_spec2col_buffers.h"
#include "cdrobert/mellin/mconv.h"
#include "cdrobert/mellin/mconv_cuda.cuh"


int main(int argc, const char* argv[]) {

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
        snd2colb::kNumTests + lin2colb::kNumTests + spec2colb::kNumTests
      ) || test_idx < 0) {
    std::cerr << "Invalid test idx: " << test_idx << std::endl;
    return 1;
  }


  int nfx, nfy, ngx, ngy, c_in, sx, sy, dx, dy, px, py, ux, uy;
  const char *name;
  const double *G, *H;
  if (test_idx < snd2colb::kNumTests) {
    name = "snd2colb";
    nfx = snd2colb::kNF[test_idx]; nfy = 1;
    ngx = snd2colb::kNG[test_idx]; ngy = 1;
    c_in = snd2colb::kCIn[test_idx];
    sx = snd2colb::kS[test_idx]; sy = 1;
    dx = snd2colb::kD[test_idx]; dy = 1;
    px = snd2colb::kP[test_idx]; py = 0;
    ux = snd2colb::kU[test_idx]; uy = 1;
    G = snd2colb::kG[test_idx];
    H = snd2colb::kH[test_idx];
  } else if (test_idx < snd2colb::kNumTests + lin2colb::kNumTests) {
    name = "lin2colb";
    test_idx -= snd2colb::kNumTests;
    nfx = 1; nfy = lin2colb::kNF[test_idx];
    ngx = 1; ngy = lin2colb::kNG[test_idx];
    c_in = lin2colb::kCIn[test_idx];
    sx = 1; sy = lin2colb::kS[test_idx];
    dx = 1; dy = lin2colb::kD[test_idx];
    px = 0; py = lin2colb::kP[test_idx];
    ux = 1; uy = lin2colb::kU[test_idx];
    G = lin2colb::kG[test_idx];
    H = lin2colb::kH[test_idx];
  } else {
    name = "spec2colb";
    test_idx -= snd2colb::kNumTests + lin2colb::kNumTests;
    nfx = spec2colb::kNFX[test_idx]; nfy = spec2colb::kNFY[test_idx];
    ngx = spec2colb::kNGX[test_idx]; ngy = spec2colb::kNGY[test_idx];
    c_in = spec2colb::kCIn[test_idx];
    sx = spec2colb::kSX[test_idx]; sy = spec2colb::kSY[test_idx];
    dx = spec2colb::kDX[test_idx]; dy = spec2colb::kDY[test_idx];
    px = spec2colb::kPX[test_idx]; py = spec2colb::kPY[test_idx];
    ux = spec2colb::kUX[test_idx]; uy = spec2colb::kUY[test_idx];
    G = spec2colb::kG[test_idx];
    H = spec2colb::kH[test_idx];
  }

  int nhx = cdrobert::mellin::MCorrSupportSize(ngx, sx, dx, px, ux);
  int nhy = cdrobert::mellin::LCorrSupportSize(ngy, sy, py, uy);
  double *g, *h, h_exp[10000];
  CheckCuda(cudaMalloc(&g, sizeof(double) * batch * c_in * ngx * ngy));  // device
  CheckCuda(cudaMallocManaged(&h, sizeof(double) * batch * c_in * nfx * (nhx + extra_h))); // unif
  CopyGCuda(G, batch, c_in, ngx * ngy, transposed_g, g);
  // FIXME(sdrobert): we're only using extra_h on the second-last dimension
  // because I can't be arsed to update the copy code for the extra dim.
  std::fill_n(h, batch * c_in * nfx * nfy * (nhx + extra_h) * nhy, -12345.);

  cdrobert::mellin::Spec2ColCuda(
    g,
    c_in, batch, nfx, nfy, ngx, ngy, nhx + extra_h, nhy,
    sx, sy, dx, dy, px, py, ux, uy,
    transposed_g, transposed_h, h
  );

  CopyGOrHCpu(H, batch, c_in, nfx * nfy, nhx * nhy, 0, extra_h * nhy, transposed_h, h_exp);

  cudaDeviceSynchronize();
  CheckCuda(cudaGetLastError());

  CheckCuda(cudaFree(g));

  std::vector<int> shape {batch, c_in, nfx, nfy, nhx + extra_h, nhy};
  if (transposed_h) shape = {c_in, batch, nfx, nfy, nhx + extra_h, nhy};
  if (AllClose(h_exp, h, shape, 1e-10)) {
    std::cerr << "Failed test: " << test_idx << " (" << name << ")"
              << std::endl;
    return 1;
  }
  return 0;
}
