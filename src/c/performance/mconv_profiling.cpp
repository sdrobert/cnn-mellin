// Copyright 2019 Sean Robertson
#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include "OptionParser.h"
#include "blas_cpu.h"
#include "cdrobert/mellin/mconv.h"

using namespace cdrobert::mellin;
using namespace cdrobert::mellin::detail;

// matrix-multiplication-style kernels
template <typename T, int V>
void MCorr1DMM(
  const T *f, const T *g,
  ssize_t c_out, ssize_t c_in, ssize_t batch,
  ssize_t nf, ssize_t ng, ssize_t nh,
  ssize_t s, ssize_t d, ssize_t p, ssize_t u,
  bool transposed_f, bool transposed_g, bool transposed_h,
  T *h, snd2col_algo<V>)
{
  std::vector<T> f_;
  if (transposed_f) {
    f_.resize(c_out * c_in * nf);
    TransposeTensor(f, c_out, c_in, nf, f_.data());
    f = f_.data();
  }
  std::vector<T> gg(batch * c_in * nf * nh);
  Snd2Col<T>(
    g, c_in, batch, nf, ng, nh, s, d, p, u, transposed_g, false,
    gg.data(), snd2col_algo<V>());
  BatchedMatrixMultiplication<T>(
    f, gg.data(),
    batch /* p */, c_out /* m */, nh /* n */, c_in * nf /* k */,
    h
  );
  if (transposed_h)
    TransposeTensor(h, batch, c_out, nh);
}

template <typename T, int V>
void MCorr1DMMT(
  const T *f, T *g,
  ssize_t c_out, ssize_t c_in, ssize_t batch,
  ssize_t nf, ssize_t ng, ssize_t nh,
  ssize_t s, ssize_t d, ssize_t p, ssize_t u,
  bool transposed_f, bool transposed_g, bool transposed_h,
  const T *h, col2snd_algo<V>)
{  // the "transpose" op
  std::vector<T> f_(c_out * c_in * nf), h_;
  if (transposed_f) {
    // c_in, c_out, nf -> c_in, nf, c_out
    TransposeMatrixBatched(f, c_in, c_out, nf, f_.data());
  } else {
    // c_out, c_in, nf -> c_in, nf, c_out
    TransposeMatrix(f, c_out, c_in * nf, f_.data());
  }
  f = f_.data();
  if (transposed_h) {
    h_.resize(batch * c_out * nh);
    TransposeTensor(h, c_out, batch, nh, h_.data());
    h = h_.data();
  }
  std::vector<T> gg(batch * c_in * nf * nh, 0);
  BatchedMatrixMultiplication<T>(
    f, h,
    batch /* p */,  c_in * nf /* m */, nh /* n */, c_out /* k */,
    gg.data()
  );
  Col2Snd<T>(
    gg.data(), c_in, batch, nf, ng, nh, s, d, p, u, transposed_g, false,
    g, col2snd_algo<V>());
}

template<typename T, int V>
void MCorrLCorrMM(
  const T *f, const T *g,
  ssize_t c_out, ssize_t c_in, ssize_t batch,
  ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nhx, ssize_t nhy,
  ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
  ssize_t ux, ssize_t uy,
  bool transposed_f, bool transposed_g, bool transposed_h,
  T *h, spec2col_algo<V>)
{
  std::vector<T> f_;
  if (transposed_f) {
    f_.resize(c_out * c_in * nfx * nfy);
    TransposeTensor(f, c_out, c_in, nfx * nfy, f_.data());
    f = f_.data();
  }
  std::vector<T> gg(batch * c_in * nfx * nfy * nhx * nhy);
  Spec2Col<T>(
    g, c_in, batch, nfx, nfy, ngx, ngy, nhx, ngy, sx, sy, dx, dy,
    px, py, ux, uy, transposed_g, false, gg.data(), spec2col_algo<V>());
  BatchedMatrixMultiplication<T>(
    f, gg.data(),
    batch /* p */, c_out /* m */, nhx * nhy /* n */, c_in * nfx * nfy /* k */,
    h
  );
  if (transposed_h)
    TransposeTensor(h, batch, c_out, nhx * nhy);
}

template<typename T, int V>
void MCorrLCorrMMT(
  const T *f, T *g,
  ssize_t c_out, ssize_t c_in, ssize_t batch,
  ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nhx, ssize_t nhy,
  ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
  ssize_t ux, ssize_t uy,
  bool transposed_f, bool transposed_g, bool transposed_h,
  const T *h, col2spec_algo<V>)
{
  std::vector<T> f_(c_out * c_in * nfx * nfy), h_;
  if (transposed_f) {
    // c_in, c_out, nfx, nfy -> c_in, nfx, nfy, c_out
    TransposeMatrixBatched(f, c_in, c_out, nfx * nfy, f_.data());
  } else {
    // c_out, c_in, nfx, nfy -> c_in, nfx, nfy, c_out
    TransposeMatrix(f, c_out, c_in * nfx * nfy, f_.data());
  }
  f = f_.data();
  if (transposed_h) {
    h_.resize(batch * c_out * nhx * nhy);
    TransposeTensor(h, c_out, batch, nhx * nhy, h_.data());
    h = h_.data();
  }
  std::vector<T> gg(batch * c_in * nfx * nfy * nhx * nhy, 0);
  BatchedMatrixMultiplication<T>(
    f, h,
    batch /* p */,  c_in * nfx * nfy /* m */, nhx * nhy /* n */, c_out /* k */,
    gg.data()
  );
  Col2Spec<T>(
    gg.data(), c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px,
    py, ux, uy, transposed_g, false, g, col2spec_algo<V>());
}

#define TYPE_CONFIG_INNER(T, ...) \
  if (options["type"] == #T) { \
    using t = T; \
    __VA_ARGS__(); \
  }

#define TYPE_CONFIG_ALL(...) \
  TYPE_CONFIG_INNER(float, __VA_ARGS__) \
  else TYPE_CONFIG_INNER(double, __VA_ARGS__) \
  else TYPE_CONFIG_INNER(int, __VA_ARGS__)

#define _1D_ARGS \
    (t*) f.data(), (t*) g.data(), \
    c_out, c_in, batch, \
    nfx, ngx, nhx, sx, dx, px, ux, \
    transposed_f, transposed_g, transposed_h, \
    (t*) h.data()

#define _2D_ARGS \
    (t*) f.data(), (t*) g.data(), \
    c_out, c_in, batch, \
    nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy, \
    transposed_f, transposed_g, transposed_h, \
    (t*) h.data()


#define CALL_KERNEL_INNER(KERNEL_NAME, STRUCT_NAME, ALGO, DEFT_ALGO, ARGS) \
  if (kernel_name == #KERNEL_NAME && (algo == ALGO || (algo < 1 && ALGO == DEFT_ALGO))) {\
    TYPE_CONFIG_ALL(([&]{KERNEL_NAME<t>(ARGS, STRUCT_NAME<ALGO>());})) \
  }

#define CALL_KERNEL_ALL \
  CALL_KERNEL_INNER(MConv1D, mconv1d_algo, 1, kMConv1DAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MConv1D, mconv1d_algo, 2, kMConv1DAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MConv1D, mconv1d_algo, 3, kMConv1DAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MCorr1D, mcorr1d_algo, 1, kMCorr1DAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MCorr1D, mcorr1d_algo, 2, kMCorr1DAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MCorr1D, mcorr1d_algo, 3, kMCorr1DAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MCorr1DMM, snd2col_algo, 1, kSnd2ColAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MCorr1DMMT, col2snd_algo, 1, kCol2SndAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MConvLConv, mconvlconv_algo, 1, kMConvLConvAlgorithmVersion, _2D_ARGS) \
  else CALL_KERNEL_INNER(MCorrLCorr, mcorrlcorr_algo, 1, kMCorrLCorrAlgorithmVersion, _2D_ARGS) \
  else CALL_KERNEL_INNER(MCorrLCorrMM, spec2col_algo, 1, kSpec2ColAlgorithmVersion, _2D_ARGS) \
  else CALL_KERNEL_INNER(MCorrLCorrMMT, col2spec_algo, 1, kCol2SpecAlgorithmVersion, _2D_ARGS) \
  else throw std::runtime_error("Invalid algorithm for kernel");

int main(int argc, const char* argv[]) {
  auto parser = optparse::OptionParser().description("Profiling cpu kernels");
  parser.add_option("-k", "--kernel").choices({
    "MConv1D",
    "MCorr1D",
    "MCorr1DMM",
    "MCorr1DMMT",
    "MConvLConv",
    "MCorrLCorr",
    "MCorrLCorrMM",
    "MCorrLCorrMMT"
  }).help("What kernel to test").set_default("MConv1D");
  parser.add_option("-a", "--algorithm").type("int").set_default(-1)
        .help("Algorithm version");
  parser.add_option("-t", "--type").choices({
    "float",
    "double",
    "int",
  }).help("Data type to perform conv on").set_default("float");
  parser.add_option("-n", "--trials").type("int").set_default(1024)
        .help("The number of times the kernel should be run and recorded");
  parser.add_option("--burn-in").type("int").set_default(-1)
        .help(
          "The number of times the kernel should be run and NOT recorded. "
          "Default is the same number as the number of trials."
        );
  parser.add_option("--c-out").type("int").set_default(128)
        .help("Output channel dimension");
  parser.add_option("--c-in").type("int").set_default(128)
        .help("Input channel dimension");
  parser.add_option("--batch").type("int").set_default(-1)
        .help("Batch dimension. Default is 16 if 2D, 32 if 1D");
  parser.add_option("--nfx").type("int").set_default(3)
        .help("Filter x dimension");
  parser.add_option("--nfy").type("int").set_default(3)
        .help("Filter y dimension (ignored if 1D)");
  parser.add_option("--ngx").type("int").set_default(-1)
        .help("Signal x dimension. Default is 64 if 2D, 128 if 1D");
  parser.add_option("--ngy").type("int").set_default(64)
        .help("Signal y dimension (ignored if 1D)");
  parser.add_option("--sx").type("int").set_default(1);
  parser.add_option("--sy").type("int").set_default(1);
  parser.add_option("--dx").type("int").set_default(1);
  parser.add_option("--dy").type("int").set_default(1);
  parser.add_option("--px").type("int").set_default(-1);
  parser.add_option("--py").type("int").set_default(-1);
  parser.add_option("--ux").type("int").set_default(1);
  parser.add_option("--uy").type("int").set_default(1);
  parser.add_option("--rx").type("int").set_default(-1);
  parser.add_option("--ry").type("int").set_default(-1);
  parser.add_option("--transpose-f").action("store_true").set_default(0);
  parser.add_option("--transpose-g").action("store_true").set_default(0);
  parser.add_option("--transpose-h").action("store_true").set_default(0);

  optparse::Values options = parser.parse_args(argc, argv);

  const std::string kernel_name = options["kernel"];
  const bool is_1d = kernel_name.find("1D") != std::string::npos;

  int algo = (int) options.get("algorithm"),
      trials = (int) options.get("trials"),
      burn_in = (int) options.get("burn_in");
  ssize_t c_out = (ssize_t) options.get("c_out"),
          c_in = (ssize_t) options.get("c_in"),
          batch = (ssize_t) options.get("batch"),
          nfx = (ssize_t) options.get("nfx"),
          nfy = is_1d ? 1 : (ssize_t) options.get("nfy"),
          ngx = (ssize_t) options.get("ngx"),
          ngy = is_1d ? 1 : (ssize_t) options.get("ngy"),
          sx = (ssize_t) options.get("sx"),
          sy = is_1d ? 1 : (ssize_t) options.get("sy"),
          dx = (ssize_t) options.get("dx"),
          dy = is_1d ? 1 : (ssize_t) options.get("dy"),
          px = (ssize_t) options.get("px"),
          py = is_1d ? 0 : (ssize_t) options.get("py"),
          ux = (ssize_t) options.get("ux"),
          uy = is_1d ? 1 : (ssize_t) options.get("uy"),
          rx = (ssize_t) options.get("rx"),
          ry = is_1d ? 0 : (ssize_t) options.get("ry");

  bool transposed_f = (bool) options.get("transpose_f"),
       transposed_g = (bool) options.get("transpose_g"),
       transposed_h = (bool) options.get("transpose_h");

  if (burn_in < 0) burn_in = trials;
  if (batch < 0) batch = (ssize_t) (is_1d ? 32 : 16);
  if (ngx < 0) ngx = (ssize_t) (is_1d ? 128 : 64);
  if (px < 0) px = (nfx - 1) / 2;
  if (py < 0) py = (nfy - 1) / 2;
  ssize_t nhx, nhy;
  if (rx < 0) {
    nhx = ngx;
  } else {
    nhx = ((kernel_name.find("MCorr") == 0) ? MCorrValidSize(nfx, ngx, sx, dx, px, ux) : MConvValidSize(ngx, sx, dx, px, ux)) + rx;
  }
  if (rx < 0) {
    nhy = ngy;
  } else {
    nhy = ((kernel_name.find("LCorr", 5) == 5) ? LCorrValidSize(nfy, ngy, sy, dy, py, uy) : LConvValidSize(ngy, sy, py, uy)) + ry;
  }

  std::cout << "Configuration:\n"
            << "\tkernel: " << kernel_name << "\n"
            << "\talgorithm: " << ((algo < 1) ? "default" : std::to_string(algo)) << "\n"
            << "\ttype: " << options["type"] << "\n"
            << "\t(transposed_f, transposed_g, transposed_h) = "
            << transposed_f << ", " << transposed_g << ", " << transposed_h << "\n";
  if (is_1d) {
    std::cout << "\tf: (" << c_out << ", " << c_in << ", " << nfx << ")\n"
              << "\tg: (" << batch << ", " << c_in << ", " << ngx << ")\n"
              << "\th: (" << batch << ", " << c_out << ", " << nhx << ")\n"
              << "\t(s, d, p, u) = (" << sx << ", " << dx << ", " << px << ", " << ux << ")\n";
  } else {
    std::cout << "\tf: (" << c_out << ", " << c_in << ", " << nfx << ", " << nfy << ")\n"
              << "\tg: (" << batch << ", " << c_in << ", " << ngx << ", " << ngy << ")\n"
              << "\th: (" << batch << ", " << c_out << ", " << nhx << ", " << nhy << ")\n"
              << "\t(s, d, p, u) = ((" << sx << ", " << sy <<
                                "), (" << dx << ", " << dy <<
                                "), (" << px << ", " << py <<
                                "), (" << ux << ", " << uy << "))\n";
   }

  // for the sake of the transpose ops, we'll make g large enough to fit gg
  std::vector<double> f(c_out * c_in * nfx * nfy),
                      g(batch * c_in * std::max(nfx * nfy * nhx * nhy, ngx * ngy)), h(batch * c_out * nhx * nhy);

  for (int k = 0; k < burn_in; ++k) CALL_KERNEL_ALL;

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < trials; ++n) CALL_KERNEL_ALL;
  auto end = std::chrono::steady_clock::now();
  
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "total time: " << milliseconds.count()
            << "ms (average " << milliseconds.count() / trials << "ms)\n";

  return 0;
}
