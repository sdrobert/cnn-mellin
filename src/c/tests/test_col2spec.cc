// Copyright 2021 Sean Robertson
#include <iostream>
#include <vector>

#include "test_utils.h"
#include "test_utils_cpu.h"
#include "test_col2snd_buffers.h"
#include "test_col2lin_buffers.h"
#include "test_col2spec_buffers.h"
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
  if (
      test_idx >= (
        col2sndb::kNumTests + col2linb::kNumTests + col2specb::kNumTests
      ) || test_idx < 0) {
    std::cerr << "Invalid test idx: " << test_idx << std::endl;
    return 1;
  }


  int nfx, nfy, ngx, ngy, c_in, sx, sy, dx, dy, px, py, ux, uy;
  const char *name;
  const double *G, *H;
  if (test_idx < col2sndb::kNumTests) {
    name = "col2sndb";
    nfx = col2sndb::kNF[test_idx]; nfy = 1;
    ngx = col2sndb::kNG[test_idx]; ngy = 1;
    c_in = col2sndb::kCIn[test_idx];
    sx = col2sndb::kS[test_idx]; sy = 1;
    dx = col2sndb::kD[test_idx]; dy = 1;
    px = col2sndb::kP[test_idx]; py = 0;
    ux = col2sndb::kU[test_idx]; uy = 1;
    G = col2sndb::kG[test_idx];
    H = col2sndb::kH[test_idx];
  } else if (test_idx < col2sndb::kNumTests + col2linb::kNumTests) {
    name = "col2linb";
    test_idx -= col2sndb::kNumTests;
    nfx = 1; nfy = col2linb::kNF[test_idx];
    ngx = 1; ngy = col2linb::kNG[test_idx];
    c_in = col2linb::kCIn[test_idx];
    sx = 1; sy = col2linb::kS[test_idx];
    dx = 1; dy = col2linb::kD[test_idx];
    px = 0; py = col2linb::kP[test_idx];
    ux = 1; uy = col2linb::kU[test_idx];
    G = col2linb::kG[test_idx];
    H = col2linb::kH[test_idx];
  } else {
    name = "col2specb";
    test_idx -= col2sndb::kNumTests + col2linb::kNumTests;
    nfx = col2specb::kNFX[test_idx]; nfy = col2specb::kNFY[test_idx];
    ngx = col2specb::kNGX[test_idx]; ngy = col2specb::kNGY[test_idx];
    c_in = col2specb::kCIn[test_idx];
    sx = col2specb::kSX[test_idx]; sy = col2specb::kSY[test_idx];
    dx = col2specb::kDX[test_idx]; dy = col2specb::kDY[test_idx];
    px = col2specb::kPX[test_idx]; py = col2specb::kPY[test_idx];
    ux = col2specb::kUX[test_idx]; uy = col2specb::kUY[test_idx];
    G = col2specb::kG[test_idx];
    H = col2specb::kH[test_idx];
  }

  int nhx = cdrobert::mellin::MCorrSupportSize(ngx, sx, dx, px, ux);
  int nhy = cdrobert::mellin::LCorrSupportSize(ngy, sy, py, uy);
  double g_exp[10000], h[10000];
  // N.B. extra_h doesn't make sense in the negative direction as it will
  // change the values of g (fewer terms to accumulate)
  extra_h = std::max(extra_h, 0);
  CopyGOrHCpu(G, batch, c_in, ngx, ngy, 0, 0, transposed_g, g_exp);
  // FIXME(sdrobert): we're only using extra_h on the second-last dimension
  // because I can't be arsed to update the copy code for the extra dim.
  CopyGOrHCpu(H, batch, c_in, nfx * nfy, nhx * nhy, 0, extra_h * nhy, transposed_h, h);
  double g[10000];
  std::fill_n(g, batch * c_in * ngx * ngy, -12345.);

  cdrobert::mellin::Col2Spec(
    h,
    c_in, batch, nfx, nfy, ngx, ngy, nhx + extra_h, nhy,
    sx, sy, dx, dy, px, py, ux, uy,
    transposed_g, transposed_h, g
  );
  // std::cout << "nf: " << nf << " ng: " << ng << " nh:" << nh << " s: " << s << " d: " << d << " p: " << p << " u: " << u << " extra_h: " << extra_h << std::endl;

  std::vector<int> shape {batch, c_in, ngx, ngy};
  if (transposed_g) shape = {c_in, batch, ngx, ngy};
  if (AllClose(g_exp, g, shape, 1e-10)) {
    std::cerr << "Failed test: " << test_idx << " (" << name << ")"
              << std::endl;
    return 1;
  }
  return 0;
}
