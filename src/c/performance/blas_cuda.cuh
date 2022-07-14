// Copyright 2021 Sean Robertson

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "cdrobert/mellin/config_cuda.h"

cublasHandle_t GetHandle();

template<typename T>
__global__ void _TransposeTensorCuda(
  const T* __restrict__ a,
  int n, int m, int z,
  T* __restrict__ b)
{
  if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
  const int i_start = blockDim.x * blockIdx.x + threadIdx.x;
  const int i_stride = blockDim.x * gridDim.x;
  for (int i = i_start; i < n * m * z; i += i_stride) {
    int iz = i % z, ii = i / z;
    int im = ii % m, in = ii / m;
    b[(im * n + in) * z + iz] = a[i];
  }
}

template<typename T>
void TransposeTensorCuda(
  const T* a /* n x m x z */,
  int n, int m, int z,
  T* b  /* m x n x z */)
{  // out-of-place
  const int threads = cdrobert::mellin::kMaxCudaThreadsPerBlock;
  const int blocks = (n * m * z + threads - 1) / threads;
  _TransposeTensorCuda<<<blocks, threads>>>(a, n, m, z, b);
}

template<typename T>
__global__ void _TransposeMatrixCuda(
  const T* __restrict__ a,
  int n, int m,
  T* __restrict__ b)
{
  if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
  const int i_start = blockDim.x * blockIdx.x + threadIdx.x;
  const int i_stride = blockDim.x * gridDim.x;
  for (int i = i_start; i < n * m; i += i_stride) {
    int in = i / m, im = i % m;
    b[im * n + in] = a[i];
  }
}

template<typename T>
void TransposeMatrixCuda(
  const T* a /* n x m */,
  int n, int m,
  T* b /* m x n */)
{
  const int threads = cdrobert::mellin::kMaxCudaThreadsPerBlock;
  const int blocks = (n * m + threads - 1) / threads;
  _TransposeMatrixCuda<<<blocks, threads>>>(a, n, m, b);
}


template<typename T>
__global__ void _TransposeMatrixBatchedCuda(
  const T* __restrict__ a,
  int p, int n, int m,
  T* __restrict__ b)
{
  if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
  const int i_start = blockDim.x * blockIdx.x + threadIdx.x;
  const int i_stride = blockDim.x * gridDim.x;
  for (int i = i_start; i < p * n * m; i += i_stride) {
    int im = i % m, ii = i / m;
    int in = ii % n, ip = ii / n;
    b[(ip * m + im) * m + in] = a[i];
  }
}

template<typename T>
void TransposeMatrixBatchedCuda(
  const T* a /* p x n x m */,
  int p, int n, int m,
  T* b /* p x m x n */)
{
  const int threads = cdrobert::mellin::kMaxCudaThreadsPerBlock;
  const int blocks = (p * n * m + threads - 1) / threads;
  _TransposeMatrixBatchedCuda<<<blocks, threads>>>(a, p, n, m, b);
}

template<typename T>
__global__ void _BatchedMatrixMultiplicationCuda(
  const T* __restrict__ a /* m x k */, const T* b /* p x k x n */,
  int p, int m, int n, int k,
  T* __restrict__ c /* p x m x n */)
{
  if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
  const int i_start = blockDim.x * blockIdx.x + threadIdx.x;
  const int i_stride = blockDim.x * gridDim.x;
  for (int i = i_start; i < p * n * m; i += i_stride) {
    int im = i % m, ii = i / m;
    int in = ii % n, ip = ii / n;
    cdrobert::mellin::detail::cuda_acc_t<T> v = 0;
    for (int ik = 0; ik < k; ++ik)
      v += a[im * k + ik] * b[(ip * k + ik) * n + in];
    c[i] += v;
  }
}

template<typename T>
void BatchedMatrixMultiplicationCuda(
  const T* a /* m x k */, const T* b /* p x k x n */,
  int p, int m, int n, int k,
  T* c /* p x m x n */)
{
  const int threads = cdrobert::mellin::kMaxCudaThreadsPerBlock;
  const int blocks = (p * n * m + threads - 1) / threads;
  _BatchedMatrixMultiplicationCuda<<<blocks, threads>>>(a, b, p, m, n, k, c);
}

// template specialization

template<>
void TransposeMatrixCuda(
  const float* a /* n x m */,
  int n, int m,
  float* b /* m x n */)
{
  float alpha = 1, beta = 0;
  cublasSgeam(
    GetHandle(),
    CUBLAS_OP_T, CUBLAS_OP_N,
    n, m,
    &alpha, a, m,
    &beta, a, n,
    b, m
  );
}

template<>
void TransposeMatrixCuda(
  const double* a /* n x m */,
  int n, int m,
  double* b /* m x n */)
{
  double alpha = 1, beta = 0;
  cublasDgeam(
    GetHandle(),
    CUBLAS_OP_T, CUBLAS_OP_N,
    n, m,
    &alpha, a, m,
    &beta, a, n,
    b, m
  );
}


template<>
void BatchedMatrixMultiplicationCuda(
  const float* a, const float* b,
  int p, int m, int n, int k,
  float* c)
{
  float alpha = 1.0, beta = 0.0;
  cublasSgemmStridedBatched(
    GetHandle(),
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,
    &alpha,
    b, n, n * k,
    a, k, 0,
    &beta,
    c, n, m * n,
    p
  );
}

template<>
void BatchedMatrixMultiplicationCuda(
  const double* a, const double* b,
  int p, int m, int n, int k,
  double* c)
{
  double alpha = 1.0, beta = 0.0;
  cublasDgemmStridedBatched(
    GetHandle(),
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,
    &alpha,
    b, n, n * k,
    a, k, 0,
    &beta,
    c, n, m * n,
    p
  );
}