// Copyright 2021 Sean Robertson

#pragma once

#include <cstring>

#include "cdrobert/mellin/mconv.h"

template<typename T>
void TransposeTensor(
  const T* a /* n x m x z */,
  int n, int m, int z,
  T* b  /* m x n x z */
) {  // out-of-place
  #pragma omp parallel for collapse(2) firstprivate(a, b)
  for (int in = 0; in < n; ++in)
    for (int im = 0; im < m; ++im)
      std::memcpy(b + (im * n + in) * z, a + (in * m + im) * z, z * sizeof(T));
}

template<typename T>
void TransposeTensor(
  T* a /* n x m x z -> m x n x z */,
  int n, int m, int z
) { // in-place
  std::vector<T> b(a, a + n * m * z);
  TransposeTensor<T>(b.data(), n, m, z, a);
}

template<typename T>
void TransposeMatrix(
  const T* a /* n x m */,
  int n, int m,
  T* b /* m x n */)
{
  TransposeTensor<T>(a, n, m, 1, b); 
}

template<typename T>
void TransposeMatrix(
  T* a /* n x m -> m x n */,
  int n, int m)
{
  std::vector<T> b(a, a + n * m);
  TransposeMatrix<T>(b, n, m, a);
}

template<typename T>
void TransposeMatrixBatched(
  const T* a /* p x n x m */,
  int p, int n, int m,
  T* b /* p x m x n */)
{
  #pragma omp parallel for collapse(3) firstprivate(a, b)
  for (int ip = 0; ip < p; ++ip)
    for (int in = 0; in < n; ++in)
      for (int im = 0; im < m; ++im)
        b[(ip * m + im) * n + in] = a[(ip * n + in) * m + im];
}

template<typename T>
void TransposeMatrixBatched(
  T* a /* p x n x m -> p x m x n */,
  int p, int n, int m)
{
  std::vector<T> b(a, a + p * n * m);
  TransposeMatrixBatched(b.data(), p, n, m, a);  
}

template<typename T>
void BatchedMatrixMultiplication(
  const T* a /* m x k */, const T* b /* p x k x n */,
  int p, int m, int n, int k,
  T* c /* p x m x n */)
{
  #pragma omp parallel for collapse(3) firstprivate(a, b, c)
  for (int ip = 0; ip < p; ++ip)
    for (int im = 0; im < m; ++im)
      for (int in = 0; in < n; ++in) {
        cdrobert::mellin::detail::acc_t<T> v = 0;
        for (int ik = 0; ik < k; ++ik)
          v += a[im * k + ik] * b[(ip * k + ik) * n + in];
        c[(ip * m + im) * n + im] += v;
      }
}

#ifdef HAVE_MKL
#include <mkl.h>

template<>
void TransposeMatrix(const float* a, int n, int m, float* b)
{ mkl_somatcopy('r', 't', n, m, 1.0, a, m, b, n); }

template<>
void TransposeMatrix(const double* a, int n, int m, double* b)
{ mkl_domatcopy('r', 't', n, m, 1.0, a, m, b, n); }

template<>
void TransposeMatrix(float* a, int n, int m)
{ mkl_simatcopy('r', 't', n, m, 1.0, a, m, n); }

template<>
void TransposeMatrix(double* a, int n, int m)
{ mkl_dimatcopy('r', 't', n, m, 1.0, a, m, n); }

template<>
void TransposeMatrixBatched(const float *a, int p, int n, int m, float *b)
{ mkl_somatcopy_batch_strided(CblasRowMajor, 't', n, m, 1.0, a, m, n * m, b, n, n * m, p); }

template<>
void TransposeMatrixBatched(const double *a, int p, int n, int m, double *b)
{ mkl_domatcopy_batch_strided(CblasRowMajor, 't', n, m, 1.0, a, m, n * m, b, n, n * m, p); }

template<>
void TransposeMatrixBatched(float *a, int p, int n, int m)
{ mkl_simatcopy_batch_strided(CblasRowMajor, 't', n, m, 1.0, a, n, m, n * m, p); }

template<>
void TransposeMatrixBatched(double *a, int p, int n, int m)
{ mkl_dimatcopy_batch_strided(CblasRowMajor, 't', n, m, 1.0, a, n, m, n * m, p); }

template<>
void BatchedMatrixMultiplication(
  const float* a /* m x k */, const float* b /* p x k x n */,
  int p, int m, int n, int k,
  float* c /* p x m x n */)
{
  cblas_sgemm_batch_strided(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    m, n, k, 1.0, a, k, 0, b, n, n * k, 0.0, c, n, m * n, p
  );
}

template<>
void BatchedMatrixMultiplication(
  const double* a /* m x k */, const double* b /* p x k x n */,
  int p, int m, int n, int k,
  double* c /* p x m x n */)
{
  cblas_dgemm_batch_strided(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    m, n, k, 1.0, a, k, 0, b, n, n * k, 0.0, c, n, m * n, p
  );
}

#else
  #warning "MKL not found! *MM CPU routines will suck!"
#endif