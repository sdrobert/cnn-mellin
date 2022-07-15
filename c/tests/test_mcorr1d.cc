// Copyright 2018 Sean Robertson
#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cpu.h"
#include "test_mcorr1d_buffers.h"
#include "cdrobert/mellin/mconv.h"

int main(int argc, const char* argv[]) {

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
  double f[10000], g[10000], h_exp[10000];
  CopyFCpu(mcorr1db::kF[test_idx], c_in, c_out, nf, transposed_f, f);
  CopyGOrHCpu(mcorr1db::kG[test_idx], batch, c_in, 1, ng, 0, 0, transposed_g, g);
  CopyGOrHCpu(mcorr1db::kH[test_idx], batch, c_out, 1, nh, 0, extra_h, transposed_h, h_exp);
  double h[10000] = {0.};

  cdrobert::mellin::MCorr1D(
    f, g, c_out, c_in, batch, nf, ng, nh + extra_h, s, d, p, u,
    transposed_f, transposed_g, transposed_h, h);

  std::vector<int> shape {batch, c_out, nh + extra_h};
  if (transposed_h) shape = {c_out, batch, nh - 1};
  if (AllClose(h_exp, h, shape, 1e-10)) {
    std::cerr << "Failed test: " << test_idx << std::endl;
    return 1;
  }
  return 0;
}
