// Copyright 2018 Sean Robertson

#pragma once

#ifndef TEST_UTILS_CUDA_H_
#define TEST_UTILS_CUDA_H_

#include <cuda_runtime.h>
#include <assert.h>
#include <vector>

#include <iostream>

inline cudaError_t CheckCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(result));
  }
  return result;
}

template <class T>
void CopyFCuda(const T* f_in, int c_in, int c_out, int n, bool transposed,
               T *f_out) {
  if (transposed) {
   for (int co = 0; co < c_out; ++co) {
     for (int ci = 0; ci < c_in; ++ci) {
       CheckCuda(cudaMemcpy(f_out + (ci * c_out + co) * n,
                            f_in + (co * c_in + ci) * n,
                            n * sizeof(T),
                            cudaMemcpyHostToDevice));
     }
   }
  } else {
   CheckCuda(cudaMemcpy(f_out,
                        f_in,
                        c_out * c_in * n * sizeof(T),
                        cudaMemcpyHostToDevice));
  }
}

template <class T>
void CopyGCuda(const T *g_in, int batch, int c_in, int n, bool transposed,
               T *g_out, cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
  if (transposed) {
   for (int bt = 0; bt < batch; ++bt) {
     for (int ci = 0; ci < c_in; ++ci) {
       CheckCuda(cudaMemcpy(g_out + (ci * batch + bt) * n,
                            g_in + ci * n,
                            n * sizeof(T),
                            kind));
     }
   }
  } else {
   for (int bt = 0; bt < batch; ++bt) {
     CheckCuda(cudaMemcpy(g_out + bt * c_in * n,
                          g_in,
                          c_in * n * sizeof(T),
                          kind));
   }
  }
}

template <class T>
void CopyHCpu(
                 const T *h_in, int batch, int c_out, int nhx, int nhy,
                 int extra_x, int extra_y, bool transposed, T *b_out) {
  if (transposed) {
    for (int bt = 0; bt < batch; ++bt) {
      for (int c = 0; c < c_out; ++c) {
        T *b_outp = b_out + (c * batch + bt) * (nhx + extra_x) * (nhy + extra_y);
        const T *h_inp = h_in + c * nhx * nhy;
        if (!extra_x && !extra_y) {
          std::memcpy(b_outp, h_inp, nhx * nhy * sizeof(T));
        } else {
          for (int x = 0; x < nhx + extra_x; ++x) {
            if (x >= nhx) {
              std::fill_n(b_outp + x * (nhy + extra_y), nhy + extra_y, static_cast<T>(0));
            } else if (extra_y <= 0) {
              std::memcpy(
                b_outp + x * (nhy + extra_y),
                h_inp + x * nhy,
                (nhy + extra_y) * sizeof(T));
            } else {
              std::memcpy(
                b_outp + x * (nhy + extra_y),
                h_inp + x * nhy,
                nhy * sizeof(T));
              std::fill_n(b_outp + x * (nhy + extra_y) + nhy, extra_y, static_cast<T>(0));
            }
          }
        }
      }
    }
  } else if (!extra_x && !extra_y) {
    for (int bt = 0; bt < batch; ++bt) {
      std::memcpy(b_out + bt * c_out * nhx * nhy,
                  h_in, c_out * nhx * nhy * sizeof(T));
    }
  } else {
    for (int bt = 0; bt < batch; ++bt) {
      for (int c = 0; c < c_out; ++c) {
        T *b_outp = b_out + (bt * c_out + c) * (nhx + extra_x) * (nhy + extra_y);
        const T *h_inp = h_in + c * nhx * nhy;
        if (!extra_x && !extra_y) {
          std::memcpy(b_outp, h_inp, nhx * nhy * sizeof(T));
        } else {
          for (int x = 0; x < nhx + extra_x; ++x) {
            if (x >= nhx) {
              std::fill_n(b_outp + x * (nhy + extra_y), nhy + extra_y, static_cast<T>(0));
            } else if (extra_y <= 0) {
              std::memcpy(
                b_outp + x * (nhy + extra_y),
                h_inp + x * nhy,
                (nhy + extra_y) * sizeof(T));
            } else {
              std::memcpy(
                b_outp + x * (nhy + extra_y),
                h_inp + x * nhy,
                nhy * sizeof(T));
              std::fill_n(b_outp + x * (nhy + extra_y) + nhy, extra_y, static_cast<T>(0));
            }
          }
        }
      }
    }
  }
}

template <class T>
void CopyHCuda(const T *h_in, int batch, int c_out, int nhx, int nhy,
               int extra_x, int extra_y, bool transposed, T *h_out) {
  // FIXME(sdrobert): don't be lazy
  int bsize = batch * c_out * (nhx + extra_x) * (nhy + extra_y);
  std::vector<T> b_out(bsize);
  CopyHCpu(h_in, batch, c_out, nhx, nhy, extra_x, extra_y, transposed, b_out.data());
  CheckCuda(cudaMemcpy(h_out, b_out.data(), bsize * sizeof(T), cudaMemcpyHostToDevice));
}

#endif  // TEST_UTILS_CUDA_H_
