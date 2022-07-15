#include <iostream>
#include <vector>
#include "test_utils.h"
#include "test_utils_cpu.h"
#include "cdrobert/mellin/mconv.h"


int main(int argc, const char* argv[]) {
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

  std::vector<float> f(c_out * c_in * nfx * nfy),
                     g1(c_in * ngx * ngy),
                     gN(batch * c_in * ngx * ngy),
                     exp_h1(c_out * nhx * nhy),
                     exp_hN(batch * c_out * nhx * nhy),
                     act_hN(batch * c_out * nhx * nhy);
  
  if (GenerateRandom(c_out * c_in * nfx * nfy, f.data(), 0)) return 1;
  if (GenerateRandom(c_in * ngx * ngy, g1.data(), 1)) return 1;
  if (std::all_of(f.begin(), f.end(), [](float x) { return x == 0.0f; })) {
    std::cerr << "Warning! With this configuration, f are all zero" << std::endl;
  }
  if (std::all_of(g1.begin(), g1.end(), [](float x) { return x == 0.0f; })) {
    std::cerr << "Warning! With this configuration, g are all zero" << std::endl;
  }

  if (argc == 15) {
    if (conv) {
      cdrobert::mellin::MConv1D(f.data(), g1.data(),
                                c_out, c_in, 1, nfx, ngx, nhx, sx, dx, px, ux,
                                transposed_f, transposed_g, transposed_h,
                                exp_h1.data());
    } else {
      cdrobert::mellin::MCorr1D(f.data(), g1.data(),
                                c_out, c_in, 1, nfx, ngx, nhx, sx, dx, px, ux,
                                transposed_f, transposed_g, transposed_h,
                                exp_h1.data());
    }
  } else {
    if (conv) {
      cdrobert::mellin::MConvLConv(f.data(), g1.data(),
                                   c_out, c_in, 1, nfx, nfy, ngx, ngy, nhx,
                                   nhy, sx, sy, dx, dy, px, py, ux, uy,
                                   transposed_f, transposed_g, transposed_h,
                                   exp_h1.data());
    } else {
      cdrobert::mellin::MCorrLCorr(f.data(), g1.data(),
                                   c_out, c_in, 1, nfx, nfy, ngx, ngy, nhx,
                                   nhy, sx, sy, dx, dy, px, py, ux, uy,
                                   transposed_f, transposed_g, transposed_h,
                                   exp_h1.data());
    }
  }

  if (std::all_of(exp_h1.begin(), exp_h1.end(), [](float x) { return x == 0.0f; })) {
    std::cerr << "Warning! With this configuration, expected values are all zero" << std::endl;
  }

  CopyGOrHCpu(g1.data(), batch, c_in, ngx, ngy, 0, 0, transposed_g, gN.data());
  CopyGOrHCpu(exp_h1.data(), batch, c_out, nhx, nhy, 0, 0, transposed_h, exp_hN.data());

  if (argc == 15) {
    if (conv) {
      cdrobert::mellin::MConv1D(f.data(), gN.data(),
                                c_out, c_in, batch, nfx, ngx, nhx, sx, dx, px,
                                ux, transposed_f, transposed_g, transposed_h,
                                act_hN.data());
    } else {
      cdrobert::mellin::MCorr1D(f.data(), gN.data(),
                                c_out, c_in, batch, nfx, ngx, nhx, sx, dx, px,
                                ux, transposed_f, transposed_g, transposed_h,
                                act_hN.data());
    }
  } else {
    if (conv) {
      cdrobert::mellin::MConvLConv(f.data(), gN.data(),
                                   c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx,
                                   nhy, sx, sy, dx, dy, px, py, ux, uy,
                                   transposed_f, transposed_g, transposed_h,
                                   act_hN.data());
    } else {
      cdrobert::mellin::MCorrLCorr(f.data(), gN.data(),
                                   c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx,
                                   nhy, sx, sy, dx, dy, px, py, ux, uy,
                                   transposed_f, transposed_g, transposed_h,
                                   act_hN.data());
    }
  }

  std::vector<int> shape;
  if (transposed_h) {
    shape = {c_out, batch, nhx};
  } else {
    shape = {batch, c_out, nhx};
  }
  if (argc == 22) shape.emplace_back(nhy);
  return AllClose(exp_hN.data(), act_hN.data(), shape, 1e-8f);
}
