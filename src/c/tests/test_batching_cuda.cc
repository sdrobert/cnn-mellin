// Copyright 2021 Sean Robertson
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include "test_utils.h"
#include "test_utils_cuda.h"
#include "cdrobert/mellin/mconv_cuda.cuh"


int main(int argc, const char* argv[]) {

  CheckCuda(cudaGetLastError());

  if ((argc != 15) && (argc != 22))
  {
    std::cerr << "Invalid number of arguments: " << argc << std::endl;
    return 1;
  }

  int c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy;
  bool transposed_f, transposed_g, transposed_h, conv;

  if (ParseNumber(argv[1], &c_out)) return 1;
  if (ParseNumber(argv[2], &c_in)) return 1;
  if (ParseNumber(argv[3], &batch)) return 1;
  if (ParseNumber(argv[4], &nfx)) return 1;
  if (ParseNumber(argv[5], &ngx)) return 1;
  if (ParseNumber(argv[6], &nhx)) return 1;
  if (ParseNumber(argv[7], &sx)) return 1;
  if (ParseNumber(argv[8], &dx)) return 1;
  if (ParseNumber(argv[9], &px)) return 1;
  if (ParseNumber(argv[10], &ux)) return 1;
  if (ParseBoolean(argv[11], &transposed_f)) return 1;
  if (ParseBoolean(argv[12], &transposed_g)) return 1;
  if (ParseBoolean(argv[13], &transposed_h)) return 1;
  if (ParseBoolean(argv[14], &conv)) return 1;
  std::cout << "c_out=" << c_out
          << ", c_in=" << c_in
          << ", batch=" << batch
          << ", nfx=" << nfx
          << ", ngx=" << ngx
          << ", nhx=" << nhx
          << ", sx=" << sx
          << ", dx=" << dx
          << ", px=" << px
          << ", ux=" << ux
          << ", transposed_f=" << transposed_f
          << ", transposed_g=" << transposed_g
          << ", transposed_h=" << transposed_h
          << ", conv=" << conv;
  
  if (argc == 22) {
    if (ParseNumber(argv[15], &nfy)) return 1;
    if (ParseNumber(argv[16], &ngy)) return 1;
    if (ParseNumber(argv[17], &nhy)) return 1;
    if (ParseNumber(argv[18], &sy)) return 1;
    if (ParseNumber(argv[19], &dy)) return 1;
    if (ParseNumber(argv[20], &py)) return 1;
    if (ParseNumber(argv[21], &uy)) return 1;

    std::cout << ", nfy=" << nfy
              << ", ngy=" << ngy
              << ", nhy=" << nhy
              << ", sy=" << sy
              << ", dy=" << dy
              << ", py=" << py
              << ", uy=" << uy;
  } else {
    nfy = ngy = nhy = sy = dy = py = uy = 1;
  }

  std::cout << std::endl;

  std::vector<float> f_(c_out * c_in * nfx * nfy),
                     g1_(c_in * ngx * ngy);
  
  if (GenerateRandom(c_out * c_in * nfx * nfy, f_.data(), 0)) return 1;
  if (GenerateRandom(c_in * ngx * ngy, g1_.data(), 1)) return 1;
  if (std::all_of(f_.begin(), f_.end(), [](float x) { return x == 0.0f; })) {
    std::cerr << "Warning! With this configuration, f are all zero" << std::endl;
  }
  if (std::all_of(g1_.begin(), g1_.end(), [](float x) { return x == 0.0f; })) {
    std::cerr << "Warning! With this configuration, g are all zero" << std::endl;
  }

  float *f, *g1, *exp_h1, *gN, *exp_hN, *act_hN;
  CheckCuda(cudaMalloc(&f, sizeof(float) * c_out * c_in * nfx * nfy));  // device
  CheckCuda(cudaMalloc(&g1, sizeof(float) * c_in * ngx * ngy));  // device
  CheckCuda(cudaMalloc(&gN, sizeof(float) * batch * c_in * ngx * ngy)); // device
  CheckCuda(cudaMallocManaged(&exp_h1, sizeof(float) * c_out * nhx * nhy)); // device
  CheckCuda(cudaMemset(exp_h1, 0., sizeof(float) * c_out * nhx * nhy));
  CheckCuda(cudaMallocManaged(&exp_hN, sizeof(float) * c_out * batch * nhx * nhy)); // unif
  CheckCuda(cudaMallocManaged(&act_hN, sizeof(float) * c_out * batch * nhx * nhy)); // unif
  CheckCuda(cudaMemset(act_hN, 0., sizeof(float) * c_out * batch * nhx * nhy));

  CopyFCuda(f_.data(), c_in, c_out, nfx * nfy, transposed_f, f);
  CopyGCuda(g1_.data(), 1, c_in, ngx * ngy, transposed_g, g1);

  if (argc == 15) {
    if (conv) {
      cdrobert::mellin::MConv1DCuda(f, g1,
                                c_out, c_in, 1, nfx, ngx, nhx, sx, dx, px, ux,
                                transposed_f, transposed_g, transposed_h,
                                exp_h1);
    } else {
      cdrobert::mellin::MCorr1DCuda(f, g1,
                                c_out, c_in, 1, nfx, ngx, nhx, sx, dx, px, ux,
                                transposed_f, transposed_g, transposed_h,
                                exp_h1);
    }
  } else {
    if (conv) {
      cdrobert::mellin::MConvLConvCuda(f, g1,
                                   c_out, c_in, 1, nfx, nfy, ngx, ngy, nhx,
                                   nhy, sx, sy, dx, dy, px, py, ux, uy,
                                   transposed_f, transposed_g, transposed_h,
                                   exp_h1);
    } else {
      cdrobert::mellin::MCorrLCorrCuda(f, g1,
                                   c_out, c_in, 1, nfx, nfy, ngx, ngy, nhx,
                                   nhy, sx, sy, dx, dy, px, py, ux, uy,
                                   transposed_f, transposed_g, transposed_h,
                                   exp_h1);
    }
  }

  cudaDeviceSynchronize();
  CheckCuda(cudaGetLastError());

  if (std::all_of(exp_h1, exp_h1 + (c_out * nhx * nhy), [](float x) { return x == 0.0f; })) {
    std::cerr << "Warning! With this configuration, expected values are all zero" << std::endl;
  }

  CopyGCuda(g1, batch, c_in, ngx * ngy, transposed_g, gN, cudaMemcpyDeviceToDevice);
  CopyHCpu(exp_h1, batch, c_out, nhx, nhy, 0, 0, transposed_h, exp_hN);

  if (argc == 15) {
    if (conv) {
      cdrobert::mellin::MConv1DCuda(f, gN,
                                c_out, c_in, batch, nfx, ngx, nhx, sx, dx, px,
                                ux, transposed_f, transposed_g, transposed_h,
                                act_hN);
    } else {
      cdrobert::mellin::MCorr1DCuda(f, gN,
                                c_out, c_in, batch, nfx, ngx, nhx, sx, dx, px,
                                ux, transposed_f, transposed_g, transposed_h,
                                act_hN);
    }
  } else {
    if (conv) {
      cdrobert::mellin::MConvLConvCuda(f, gN,
                                   c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx,
                                   nhy, sx, sy, dx, dy, px, py, ux, uy,
                                   transposed_f, transposed_g, transposed_h,
                                   act_hN);
    } else {
      cdrobert::mellin::MCorrLCorrCuda(f, gN,
                                   c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx,
                                   nhy, sx, sy, dx, dy, px, py, ux, uy,
                                   transposed_f, transposed_g, transposed_h,
                                   act_hN);
    }
  }
  
  cudaDeviceSynchronize();
  CheckCuda(cudaGetLastError());

  CheckCuda(cudaFree(f));
  CheckCuda(cudaFree(g1));
  CheckCuda(cudaFree(gN));
  CheckCuda(cudaFree(exp_h1));

  std::vector<int> shape;
  if (transposed_h) {
    shape = {c_out, batch, nhx};
  } else {
    shape = {batch, c_out, nhx};
  }
  if (argc == 22) shape.emplace_back(nhy);
  int ret = AllClose(exp_hN, act_hN, shape, 1e-8f);

  CheckCuda(cudaFree(exp_hN));
  CheckCuda(cudaFree(act_hN));
  return ret;
}
