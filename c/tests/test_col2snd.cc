// Copyright 2021 Sean Robertson
#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cpu.h"
#include "test_col2snd_buffers.h"
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
  if (test_idx >= col2sndb::kNumTests || test_idx < 0) {
    std::cerr << "Invalid test idx: " << test_idx << std::endl;
    return 1;
  }

  int nf = col2sndb::kNF[test_idx], ng = col2sndb::kNG[test_idx],
      c_in = col2sndb::kCIn[test_idx],
      s = col2sndb::kS[test_idx], d = col2sndb::kD[test_idx],
      p = col2sndb::kP[test_idx], u = col2sndb::kU[test_idx];
  int nh = cdrobert::mellin::MCorrSupportSize(ng, s, d, p, u);
  double g_exp[10000], h[10000];
  // N.B. extra_h doesn't make sense in the negative direction as it will
  // change the values of g (fewer terms to accumulate)
  extra_h = std::max(extra_h, 0);
  CopyGOrHCpu(col2sndb::kG[test_idx], batch, c_in, 1, ng, 0, 0, transposed_g, g_exp);
  CopyGOrHCpu(col2sndb::kH[test_idx], batch, c_in, nf, nh, 0, extra_h, transposed_h, h);
  double g[10000];
  std::fill_n(g, batch * c_in * ng, -12345.);

  cdrobert::mellin::Col2Snd(
    h,
    c_in, batch, nf, ng, nh + extra_h, s, d, p, u,
    transposed_g, transposed_h, g
  );
  // std::cout << "nf: " << nf << " ng: " << ng << " nh:" << nh << " s: " << s << " d: " << d << " p: " << p << " u: " << u << " extra_h: " << extra_h << std::endl;

  std::vector<int> shape {batch, c_in, ng};
  if (transposed_g) shape = {c_in, batch, ng};
  if (AllClose(g_exp, g, shape, 1e-10)) {
    std::cerr << "Failed test: " << test_idx << std::endl;
    return 1;
  }
  return 0;
}
