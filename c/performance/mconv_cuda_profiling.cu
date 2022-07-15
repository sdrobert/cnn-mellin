// Copyright 2019 Sean Robertson
#include <vector>
#include <string>
#include <iostream>
#include <limits>

#include "cuda_profiler_api.h"
#include "OptionParser.h"
#include "cdrobert/mellin/mconv.h"
#include "cdrobert/mellin/config_cuda.h"
#include "cdrobert/mellin/mconv_cuda.cuh"
#include "blas_cuda.cuh"

using namespace cdrobert::mellin;
using namespace cdrobert::mellin::detail;

char *kBuf = nullptr;
ssize_t kBufSize = 0;
cublasHandle_t kHandle = 0;

cublasHandle_t GetHandle() {
  if (!kHandle && cublasCreate(&kHandle) != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Could not make cublas handle");
  return kHandle;
}

inline cudaError_t CheckCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(result));
  }
  return result;
}

template <typename T, typename I, int V>
void MCorr1DMMCuda(
  const T *f, const T *g,
  ssize_t c_out, ssize_t c_in, ssize_t batch,
  ssize_t nf, ssize_t ng, ssize_t nh,
  ssize_t s, ssize_t d, ssize_t p, ssize_t u,
  bool transposed_f, bool transposed_g, bool transposed_h,
  T *h, int threads, int blocks, int shared_memory, snd2col_cuda_algo<V>)
{
  ssize_t f_size = sizeof(T) * c_out * c_in * nf;
  ssize_t buf_size = f_size + sizeof(T) * batch * c_in * nf * nh;
  if (kBufSize < buf_size) {
    if (kBuf) CheckCuda(cudaFree(kBuf));
    CheckCuda(cudaMalloc(&kBuf, buf_size));
    kBufSize = buf_size;
  }
  T *gg = (T*) kBuf;
  if (transposed_f) {
    TransposeTensorCuda<T>(f, c_out, c_in, nf, gg);
    f = gg;
    gg += f_size;
  }
  Snd2ColCuda<T, I><<<blocks, threads, shared_memory>>>(
    g, c_in, batch, nf, ng, nh, s, d, p, u, transposed_g, false, gg,
    snd2col_cuda_algo<V>());
  BatchedMatrixMultiplicationCuda<T>(
    f, gg,
    batch, c_out, nh, c_in * nf,
    h
  );
}

template <typename T, typename I, int V>
void MCorr1DMMTCuda(
  const T *f, T *g,
  ssize_t c_out, ssize_t c_in, ssize_t batch,
  ssize_t nf, ssize_t ng, ssize_t nh,
  ssize_t s, ssize_t d, ssize_t p, ssize_t u,
  bool transposed_f, bool transposed_g, bool transposed_h,
  const T *h, int threads, int blocks, int shared_memory, col2snd_cuda_algo<V>)
{
  ssize_t f_size = sizeof(T) * c_out * c_in * nf,
          h_size = sizeof(T) * batch * c_out * nh,
          gg_size = sizeof(T) * batch * c_in * nf * nh;
  ssize_t buf_size = f_size + h_size + gg_size;
  if (kBufSize < buf_size) {
    if (kBuf) CheckCuda(cudaFree(kBuf));
    CheckCuda(cudaMalloc(&kBuf, buf_size));
    kBufSize = buf_size;
  }
  T *gg = (T*) kBuf;
  if (transposed_f) {
    TransposeMatrixBatchedCuda(f, c_in, c_out, nf, gg);
  } else {
    TransposeMatrixCuda(f, c_out, c_in * nf, gg);
  }
  f = gg;
  gg += f_size;
  if (transposed_h) {
    TransposeTensorCuda(h, c_out, batch, nh, gg);
    h = gg;
    gg += h_size;
  }
  cudaMemset(gg, 0, gg_size);
  BatchedMatrixMultiplicationCuda<T>(
    f, h,
    batch, c_in * nf, nh, c_out,
    gg
  );
  Col2SndCuda<T, I><<<blocks, threads, shared_memory>>>(
    gg, c_in, batch, nf, ng, nh, s, d, p, u, transposed_g, false,
    g, col2snd_cuda_algo<V>());
}

template<typename T, typename I, int V>
void MCorrLCorrMMCuda(
  const T *f, const T *g,
  ssize_t c_out, ssize_t c_in, ssize_t batch,
  ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nhx, ssize_t nhy,
  ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
  ssize_t ux, ssize_t uy,
  bool transposed_f, bool transposed_g, bool transposed_h,
  T *h, int threads, int blocks, int shared_memory, spec2col_cuda_algo<V>)
{
  ssize_t f_size = sizeof(T) * c_out * c_in * nfx * nfy;
  ssize_t buf_size = f_size + sizeof(T) * batch * c_in * nfx * nfy * nhx * nhy;
  if (kBufSize < buf_size) {
    if (kBuf) CheckCuda(cudaFree(kBuf));
    CheckCuda(cudaMalloc(&kBuf, buf_size));
    kBufSize = buf_size;
  }
  T *gg = (T*) kBuf;
  if (transposed_f) {
    TransposeTensorCuda<T>(f, c_out, c_in, nfx * nfy, gg);
    f = gg;
    gg += f_size;
  }
  Spec2ColCuda<T, I><<<blocks, threads, shared_memory>>>(
    g, c_in, batch, nfx, nfy, ngx, ngy, nhx, ngy, sx, sy, dx, dy,
    px, py, ux, uy, transposed_g, false, gg, spec2col_cuda_algo<V>());
  BatchedMatrixMultiplicationCuda<T>(
    f, gg,
    batch, c_out, nhx * nhy, c_in * nfx * nfy,
    h
  );
}

template <typename T, typename I, int V>
void MCorrLCorrMMTCuda(
  const T *f, T *g,
  ssize_t c_out, ssize_t c_in, ssize_t batch,
  ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nhx, ssize_t nhy,
  ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
  ssize_t ux, ssize_t uy,
  bool transposed_f, bool transposed_g, bool transposed_h,
  const T *h, int threads, int blocks, int shared_memory, col2spec_cuda_algo<V>)
{
  ssize_t f_size = sizeof(T) * c_out * c_in * nfx * nfy,
          h_size = sizeof(T) * batch * c_out * nhx * nhy,
          gg_size = sizeof(T) * batch * c_in * nfx * nfy * nhx * nhy;
  ssize_t buf_size = f_size + h_size + gg_size;
  if (kBufSize < buf_size) {
    if (kBuf) CheckCuda(cudaFree(kBuf));
    CheckCuda(cudaMalloc(&kBuf, buf_size));
    kBufSize = buf_size;
  }
  T *gg = (T*) kBuf;
  if (transposed_f) {
    TransposeMatrixBatchedCuda(f, c_in, c_out, nfx * nfy, gg);
  } else {
    TransposeMatrixCuda(f, c_out, c_in * nfx * nfy, gg);
  }
  f = gg;
  gg += f_size;
  if (transposed_h) {
    TransposeTensorCuda(h, c_out, batch, nhx * nhy, gg);
    h = gg;
    gg += h_size;
  }
  cudaMemset(gg, 0, gg_size);
  BatchedMatrixMultiplicationCuda<T>(
    f, h,
    batch, c_in * nfx * nfy, nhx * nhy, c_out,
    gg
  );
  Col2SpecCuda<T, I><<<blocks, threads, shared_memory>>>(
    gg, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px,
    py, ux, uy, transposed_g, false, g, col2spec_cuda_algo<V>());
}

#define IDX_CONFIG_INNER(I, ...) \
  if (options["index"] == #I) { \
    using i = I; \
    __VA_ARGS__(); \
  }

#define IDX_CONFIG_ALL(...) \
  IDX_CONFIG_INNER(int, __VA_ARGS__) \
  else IDX_CONFIG_INNER(ssize_t, __VA_ARGS__) \

#define TYPE_CONFIG_INNER(T, ...) \
  if (options["type"] == #T) { \
    using t = T; \
    IDX_CONFIG_ALL(__VA_ARGS__); \
  }

#define TYPE_CONFIG_ALL(...) \
  TYPE_CONFIG_INNER(float, __VA_ARGS__) \
  else TYPE_CONFIG_INNER(double, __VA_ARGS__) \
  else TYPE_CONFIG_INNER(int, __VA_ARGS__)

#define CALL_KERNEL_INNER(KERNEL_NAME, STRUCT_NAME, ALGO, DEFT_ALGO, ARGS) \
  if (kernel_name == #KERNEL_NAME && (algo == ALGO || (algo < 1 && ALGO == DEFT_ALGO))) {\
    if (threads < 1) threads = STRUCT_NAME<ALGO>::GetNumThreads(ARGS); \
    if (blocks < 1) blocks = STRUCT_NAME<ALGO>::GetNumBlocks(ARGS, threads); \
    TYPE_CONFIG_ALL(([&]{ \
      int shared_memory = STRUCT_NAME<ALGO>::GetDynamicSharedMemory<t>(ARGS, threads, blocks); \
      KERNEL_NAME ## Cuda<t, i><<<blocks, threads, shared_memory>>>(\
        (t*) f, (t*) g, ARGS, \
        transposed_f, transposed_g, transposed_h, \
        (t*) h, STRUCT_NAME<ALGO>() \
      ); \
    })) \
  }

#define CALL_FUNC_INNER(KERNEL_NAME, STRUCT_NAME, ALGO, DEFT_ALGO, ARGS) \
  if (kernel_name == #KERNEL_NAME && (algo == ALGO || (algo < 1 && ALGO == DEFT_ALGO))) {\
    if (threads < 1) threads = STRUCT_NAME<ALGO>::GetNumThreads(ARGS); \
    if (blocks < 1) blocks = STRUCT_NAME<ALGO>::GetNumBlocks(ARGS, threads); \
    TYPE_CONFIG_ALL(([&]{ \
      int shared_memory = STRUCT_NAME<ALGO>::GetDynamicSharedMemory<t>(ARGS, threads, blocks); \
      KERNEL_NAME ## Cuda<t, i, ALGO>( \
        (t*) f, (t*) g, ARGS, \
        transposed_f, transposed_g, transposed_h, \
        (t*) h, threads, blocks, shared_memory, STRUCT_NAME<ALGO>() \
      ); \
    })) \
  }

#define _1D_ARGS \
  c_out, c_in, batch, \
  nfx, ngx, nhx, sx, dx, px, ux

#define _2D_ARGS \
  c_out, c_in, batch, \
  nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy
  
#define CALL_KERNEL_ALL \
  CALL_KERNEL_INNER(MConv1D, mconv1d_cuda_algo, 1, kMConv1DCudaAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MConv1D, mconv1d_cuda_algo, 2, kMConv1DCudaAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MCorr1D, mcorr1d_cuda_algo, 1, kMCorr1DCudaAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MCorr1D, mcorr1d_cuda_algo, 2, kMCorr1DCudaAlgorithmVersion, _1D_ARGS) \
  else CALL_FUNC_INNER(MCorr1DMM, snd2col_cuda_algo, 1, kSnd2ColCudaAlgorithmVersion, _1D_ARGS) \
  else CALL_FUNC_INNER(MCorr1DMMT, col2snd_cuda_algo, 1, kCol2SndCudaAlgorithmVersion, _1D_ARGS) \
  else CALL_KERNEL_INNER(MConvLConv, mconvlconv_cuda_algo, 1, kMConvLConvCudaAlgorithmVersion, _2D_ARGS) \
  else CALL_KERNEL_INNER(MCorrLCorr, mcorrlcorr_cuda_algo, 1, kMCorrLCorrCudaAlgorithmVersion, _2D_ARGS) \
  else CALL_FUNC_INNER(MCorrLCorrMM, spec2col_cuda_algo, 1, kSpec2ColCudaAlgorithmVersion, _2D_ARGS) \
  else CALL_FUNC_INNER(MCorrLCorrMMT, col2spec_cuda_algo, 1, kCol2SpecCudaAlgorithmVersion, _2D_ARGS) \
  else throw std::runtime_error("Invalid algorithm for kernel");


int main(int argc, const char* argv[]) {
  auto parser = optparse::OptionParser().description("Profiling cuda kernels");
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
  parser.add_option("-i", "--index")
        .choices({"int", "ssize_t"})
        .help("Index width").set_default("int");
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
  parser.add_option("--threads").type("int").set_default(-1)
        .help("The number of threads per block. Default matches config max");
  parser.add_option("--blocks").type("int").set_default(-1)
        .help("The number of blocks. Default matches ceil(|h|/threads)");
  parser.add_option("--transpose-f").action("store_true").set_default(0);
  parser.add_option("--transpose-g").action("store_true").set_default(0);
  parser.add_option("--transpose-h").action("store_true").set_default(0);

  optparse::Values options = parser.parse_args(argc, argv);

  const std::string kernel_name = options["kernel"];
  const bool is_1d = kernel_name.find("1D") != std::string::npos;

  int algo = (int) options.get("algorithm"),
      trials = (int) options.get("trials"),
      burn_in = (int) options.get("burn_in"),
      threads = (int) options.get("threads"),
      blocks = (int) options.get("blocks");
  ssize_t c_out = (int) options.get("c_out"),
          c_in = (int) options.get("c_in"),
          batch = (int) options.get("batch"),
          nfx = (int) options.get("nfx"),
          nfy = is_1d ? 1 : (int) options.get("nfy"),
          ngx = (int) options.get("ngx"),
          ngy = is_1d ? 1 : (int) options.get("ngy"),
          sx = (int) options.get("sx"),
          sy = is_1d ? 1 : (int) options.get("sy"),
          dx = (int) options.get("dx"),
          dy = is_1d ? 1 : (int) options.get("dy"),
          px = (int) options.get("px"),
          py = is_1d ? 0 : (int) options.get("py"),
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
    << "\talgorithm: "
      << ((algo < 1) ? "default" : std::to_string(algo)) << "\n"
    << "\ttype: " << options["type"] << "\n"
    << "\tindex: " << options["index"] << "\n"
    << "\tblocks, threads = "
      << ((blocks < 1) ? "default" : std::to_string(blocks)) << ", "
      << ((threads < 1) ? "default" : std::to_string(threads)) << "\n"
    << "\ttransposed_f, transposed_g, transposed_h = "
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

  double *f, *g, *h;
  CheckCuda(cudaMalloc(&f, sizeof(double) * c_out * c_in * nfx * nfy));
  CheckCuda(cudaMalloc(&g, sizeof(double) * batch * c_in * ngx * ngy));
  CheckCuda(cudaMalloc(&h, sizeof(double) * batch * c_out * nhx * nhy));
  CheckCuda(cudaDeviceSynchronize());

  for (int k = 0; k < burn_in; ++k) CALL_KERNEL_ALL;

  cudaDeviceSynchronize();
  CheckCuda(cudaGetLastError());
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaDeviceSynchronize();

  cudaProfilerStart();
  cudaEventRecord(start);
  for (int n = 0; n < trials; ++n) CALL_KERNEL_ALL;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaProfilerStop();
  float milliseconds = 0.;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "total time: " << milliseconds
            << "ms (average " << milliseconds / trials << "ms)\n";
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  CheckCuda(cudaFree(f));
  CheckCuda(cudaFree(g));
  CheckCuda(cudaFree(h));
  if (kHandle) cublasDestroy(kHandle);
  if (kBuf) CheckCuda(cudaFree(kBuf));
  return 0;
}
