// Copyright 2021 Sean Robertson
#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cpu.h"
#include "test_snd2col_buffers.h"
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
  if (test_idx >= snd2colb::kNumTests || test_idx < 0) {
    std::cerr << "Invalid test idx: " << test_idx << std::endl;
    return 1;
  }

  int nf = snd2colb::kNF[test_idx], ng = snd2colb::kNG[test_idx],
      c_in = snd2colb::kCIn[test_idx],
      s = snd2colb::kS[test_idx], d = snd2colb::kD[test_idx],
      p = snd2colb::kP[test_idx], u = snd2colb::kU[test_idx];
  int nh = cdrobert::mellin::MCorrSupportSize(ng, s, d, p, u);
  double g[10000], h_exp[10000];
  // CopyFCpu(snd2colb::kF[test_idx], c_in, c_out, nf, transposed_f, f);
  CopyGOrHCpu(snd2colb::kG[test_idx], batch, c_in, 1, ng, 0, 0, transposed_g, g);
  CopyGOrHCpu(snd2colb::kH[test_idx], batch, c_in, nf, nh, 0, extra_h, transposed_h, h_exp);
  double h[10000];
  std::fill_n(h, batch * c_in * nf * (nh + extra_h), -12345.);

  cdrobert::mellin::Snd2Col(
    g,
    c_in, batch, nf, ng, nh + extra_h, s, d, p, u,
    transposed_g, transposed_h, h
  );
  // std::cout << "nf: " << nf << " ng: " << ng << " nh:" << nh << " s: " << s << " d: " << d << " p: " << p << " u: " << u << " extra_h: " << extra_h << std::endl;

  std::vector<int> shape {batch, c_in, nf, nh + extra_h};
  if (transposed_h) shape = {c_in, batch, nf, nh + extra_h};
  if (AllClose(h_exp, h, shape, 1e-10)) {
    std::cerr << "Failed test: " << test_idx << std::endl;
    return 1;
  }
  return 0;
}
