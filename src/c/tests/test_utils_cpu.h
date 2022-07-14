// Copyright 2018 Sean Robertson

#pragma once

#ifndef TEST_UTILS_CPU_H_
#define TEST_UTILS_CPU_H_

#include <algorithm>

template <class T>
void CopyFCpu(const T *f_in, int c_in, int c_out, int n, bool transposed,
            T *f_out) {
  if (transposed) {
    #pragma omp parallel for collapse(2) firstprivate(f_out, f_in)
    for (int co = 0; co < c_out; ++co) {
      for (int ci = 0; ci < c_in; ++ci) {
        std::memcpy(f_out + (ci * c_out + co) * n,
                    f_in + (co * c_in + ci) * n,
                    n * sizeof(T));
      }
    }
  } else {
    std::memcpy(f_out, f_in, c_out * c_in * n * sizeof(T));
  }
}

template <class T>
void CopyGOrHCpu(
                 const T *b_in, int batch, int c_inout, int nx, int ny,
                 int extra_x, int extra_y, bool transposed, T *b_out) {
  if (transposed) {
    #pragma omp parallel for collapse(2) firstprivate(b_out, b_in)
    for (int bt = 0; bt < batch; ++bt) {
      for (int c = 0; c < c_inout; ++c) {
        T *b_outp = b_out + (c * batch + bt) * (nx + extra_x) * (ny + extra_y);
        const T *b_inp = b_in + c * nx * ny;
        if (!extra_x && !extra_y) {
          std::memcpy(b_outp, b_inp, nx * ny * sizeof(T));
        } else {
          for (int x = 0; x < nx + extra_x; ++x) {
            if (x >= nx) {
              std::fill_n(b_outp + x * (ny + extra_y), ny + extra_y, static_cast<T>(0));
            } else if (extra_y <= 0) {
              std::memcpy(
                b_outp + x * (ny + extra_y),
                b_inp + x * ny,
                (ny + extra_y) * sizeof(T));
            } else {
              std::memcpy(
                b_outp + x * (ny + extra_y),
                b_inp + x * ny,
                ny * sizeof(T));
              std::fill_n(b_outp + x * (ny + extra_y) + ny, extra_y, static_cast<T>(0));
            }
          }
        }
      }
    }
  } else if (!extra_x && !extra_y) {
    #pragma omp parallel for firstprivate(b_out, b_in)
    for (int bt = 0; bt < batch; ++bt) {
      std::memcpy(b_out + bt * c_inout * nx * ny,
                  b_in, c_inout * nx * ny * sizeof(T));
    }
  } else {
    #pragma omp parallel for collapse(2) firstprivate(b_out, b_in)
    for (int bt = 0; bt < batch; ++bt) {
      for (int c = 0; c < c_inout; ++c) {
        T *b_outp = b_out + (bt * c_inout + c) * (nx + extra_x) * (ny + extra_y);
        const T *b_inp = b_in + c * nx * ny;
        if (!extra_x && !extra_y) {
          std::memcpy(b_outp, b_inp, nx * ny * sizeof(T));
        } else {
          for (int x = 0; x < nx + extra_x; ++x) {
            if (x >= nx) {
              std::fill_n(b_outp + x * (ny + extra_y), ny + extra_y, static_cast<T>(0));
            } else if (extra_y <= 0) {
              std::memcpy(
                b_outp + x * (ny + extra_y),
                b_inp + x * ny,
                (ny + extra_y) * sizeof(T));
            } else {
              std::memcpy(
                b_outp + x * (ny + extra_y),
                b_inp + x * ny,
                ny * sizeof(T));
              std::fill_n(b_outp + x * (ny + extra_y) + ny, extra_y, static_cast<T>(0));
            }
          }
        }
      }
    }
  }
}

#endif  // TEST_UTILS_CPU_H_
