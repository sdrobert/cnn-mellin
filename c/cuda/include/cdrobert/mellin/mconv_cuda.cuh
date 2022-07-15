// Copyright 2018 Sean Robertson
/**
  \file cdrobert/mellin/mconv_cuda.cuh
  \brief Convolutional kernels on the GPU (CUDA)
*/

#pragma once

#ifndef CDROBERT_MELLIN_MCONV_CUDA_CUH_
#define CDROBERT_MELLIN_MCONV_CUDA_CUH_

#include <type_traits>
#include <limits>

#include "cdrobert/mellin/config_cuda.h"

namespace cdrobert { namespace mellin {

  // BEGIN PUBLIC INTERFACE
  using cuda_ssize_t = ptrdiff_t;

  /** \brief MConv1D on a CUDA device */
  template <class T>
  void MConv1DCuda(
    const T *f, const T *g,
    cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh,
    cuda_ssize_t s, cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h,
    cudaStream_t stream = 0
  );

  /** \brief MCorr1D on a CUDA device */
  template <class T>
  void MCorr1DCuda(
    const T *f, const T *g,
    cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh,
    cuda_ssize_t s, cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h,
    cudaStream_t stream = 0
  );

  /** \brief MConvLConv on a CUDA device */
  template <class T>
  void MConvLConvCuda(
    const T *f, const T *g,
    cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy, cuda_ssize_t nhx, cuda_ssize_t nhy,
    cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy, cuda_ssize_t px, cuda_ssize_t py,
    cuda_ssize_t ux, cuda_ssize_t uy,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h,
    cudaStream_t stream = 0
  );

  /** \brief MCorrLCorr on a CUDA device */
  template <class T>
  void MCorrLCorrCuda(
    const T *f, const T *g,
    cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy, cuda_ssize_t nhx, cuda_ssize_t nhy,
    cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy, cuda_ssize_t px, cuda_ssize_t py,
    cuda_ssize_t ux, cuda_ssize_t uy,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h,
    cudaStream_t stream = 0
  );

  /** \brief Snd2Col on a CUDA device */
  template <class T>
  void Snd2ColCuda(
    const T *g,
    cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t ngg,
    cuda_ssize_t s, cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u,
    bool transposed_g, bool transposed_gg,
    T *gg,
    cudaStream_t stream = 0
  );

  /** \brief Col2Snd on a CUDA device */
  template <class T>
  void Col2SndCuda(
    const T *gg,
    cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t ngg,
    cuda_ssize_t s, cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u,
    bool transposed_g, bool transposed_gg,
    T *g,
    cudaStream_t stream = 0
  );

  /** \brief Spec2Col on a CUDA device */
  template <typename T>
  void Spec2ColCuda(
    const T *g,
    ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nggx, ssize_t nggy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_g, bool transposed_gg,
    T *gg,
    cudaStream_t stream = 0
  );

  /** \brief Col2Spec on a CUDA device */
  template <typename T>
  void Col2SpecCuda(
    const T *gg,
    ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nggx, ssize_t nggy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_g, bool transposed_gg,
    T *g,
    cudaStream_t stream = 0
  );

  // END_PUBLIC_INTERFACE





  namespace detail {

    // the cuda implementations are mostly analogous to the CPU
    // implementations, so there won't be many comments in them.

    // accumulator. Will generally be the same as the type T unless its fewer
    // than 32 bits, at which point it's cast to the 32-bit version.
    template <typename T>
    using cuda_acc_t = typename std::common_type<
      typename std::conditional<std::is_integral<T>::value, int, float>::type,
      T
    >::type;

    // routines from
    // https://stackoverflow.com/questions/8136974/c-functions-for-integer-division-with-well-defined-rounding-strategy/33790603#33790603
    // WARNING: not appropriate for python, which uses flooring in its division
    // rather than truncation

    // ceiling division
    template <typename T1, typename T2>
    __device__
    constexpr T1 cdiv(T1 num, T2 denom) {
      return static_cast<T1>(num / denom + (((num < 0) ^ (denom > 0)) && (num % denom)));
    }

    // floor division
    template <typename T1, typename T2>
    __device__
    constexpr T1 FloorDivCuda(T1 num, T2 denom) {
      return static_cast<T1>(num / denom - (((num > 0) ^ (denom > 0)) && (num % denom)));
    }

    // take max and cast to left type
    template <typename T1, typename T2>
    __device__
    T1 LeftMaxCuda(T1 a, T2 b) {
      return max(a, static_cast<T1>(b));
    }

    // take min and cast to left type
    template <typename T1, typename T2>
    __device__
    T1 LeftMinCuda(T1 a, T2 b) {
      return min(a, static_cast<T1>(b));
    }

    // greatest common divisor
    template <class T>
    __device__
    T GcdCuda(T a, T b) {
      while (b != 0) {
        T c = b;
        b = a % b;
        a = c;
      }
      return a;
    }

    // for brevity and to make it easier to copy code between devices
    // N.B. Unlike the CPU implementations we template the index type. The
    // width can impact performance
    #ifdef A
      #define _CDROBERT_MELLIN_OLD_A A
    #endif
    #ifdef gcd
      #define _CDROBERT_MELLIN_OLD_GCD gcd
    #endif
    #ifdef cdiv
      #define _CDROBERT_MELLIN_OLD_CDIV cdiv
    #endif
    #ifdef fdiv
      #define _CDROBERT_MELLIN_OLD_FDIV fdiv
    #endif
    #ifdef lmax
      #define _CDROBERT_MELLIN_OLD_LMAX lmax
    #endif
    #ifdef lmin
      #define _CDROBERT_MELLIN_OLD_LMIN lmin
    #endif
    #define A cuda_acc_t<T>
    #define gcd GcdCuda
    #define cdiv cdiv
    #define fdiv FloorDivCuda
    #define lmax LeftMaxCuda
    #define lmin LeftMinCuda

    template <typename T>
    __device__
    T MConvValidSizeCuda(T ng, T s, T d, T p, T u) {
      return lmax(fdiv(d * (ng + u - 1), p + 1) - s + 1, 0);
    }

    template <typename T>
    __device__
    T MConvSupportSizeCuda(T nf, T ng, T s, T d, T p, T u) {
      return lmax(FloorDivCuda((nf + d - 1) * (ng + u - 1), p + 1) - s + 1, 0);
    }

    template <typename T>
    __device__
    T LConvValidSizeCuda(T ng, T s, T p, T u) {
      return lmax(FloorDivCuda(u * (ng - 1) - p, s) + 1, 0);
    }

    template <typename T>
    __device__
    T LConvSupportSizeCuda(T nf, T ng, T s, T d, T p, T u) {
      return lmax(FloorDivCuda(u * (ng - 1) + d * (nf - 1) - p,  s) + 1, 0);
    }
    
    template <typename T>
    __device__
    T MCorrValidSizeCuda(T nf, T ng, T s, T d, T p, T u) {
      return lmax(FloorDivCuda((ng + u - 1) * (p + 1), nf + d - 1) - s + 1, 0);
    }
    
    template <typename T>
    __device__
    T MCorrSupportSizeCuda(T ng, T s, T d, T p, T u) {
      return lmax(FloorDivCuda((ng + u - 1) * (p + 1), d) - s + 1, 0);
    }
    
    template <typename T>
    __device__
    T LCorrValidSizeCuda(T nf, T ng, T s, T d, T p, T u) {
      return lmax(FloorDivCuda(u * (ng - 1) - d * (nf - 1) + p, s) + 1, 0);
    }
    
    template <typename T>
    __device__
    T LCorrSupportSizeCuda(T ng, T s, T p, T u) {
      return lmax(FloorDivCuda(u * (ng - 1) + p, s) + 1, 0);
    }
  
    struct cuda_algo_1d {
      static int GetNumThreads(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s,
        cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u
      ) {
        // seems to be good for those that don't actually use the threading.
        // Whatever mechanism is chosen should be a multiple of 32
        return std::min(32, kMaxCudaThreadsPerBlock);
      }

      static int GetNumBlocks(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s,
        cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u, int threads
      ) {
        return static_cast<int>(std::min(
          std::max(
            (c_out * batch * nh + threads - 1) / threads,
            static_cast<cuda_ssize_t>(1)
          ),
          static_cast<cuda_ssize_t>(std::numeric_limits<int>::max())
        ));
      }

      template <typename T>
      static int GetDynamicSharedMemory(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s,
        cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u, int threads, int blocks
      ) {
        return 0;
      }
    };

    struct cuda_algo_2d {
      static int GetNumThreads(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy,
        cuda_ssize_t nhx, cuda_ssize_t nhy,
        cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy,
        cuda_ssize_t px, cuda_ssize_t py, cuda_ssize_t ux, cuda_ssize_t uy
      ) {
        return std::min(32, kMaxCudaThreadsPerBlock);
      }

      static int GetNumBlocks(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy,
        cuda_ssize_t nhx, cuda_ssize_t nhy,
        cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy,
        cuda_ssize_t px, cuda_ssize_t py, cuda_ssize_t ux, cuda_ssize_t uy,
        int threads
      ) {
        return static_cast<int>(std::min(
          std::max(
            (c_out * batch * nhx * nhy + threads - 1) / threads,
            static_cast<cuda_ssize_t>(1)
          ),
          static_cast<cuda_ssize_t>(std::numeric_limits<int>::max())
        ));
      }

      template <typename T>
      static int GetDynamicSharedMemory(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy,
        cuda_ssize_t nhx, cuda_ssize_t nhy,
        cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy,
        cuda_ssize_t px, cuda_ssize_t py, cuda_ssize_t ux, cuda_ssize_t uy,
        int threads, int blocks
      ) {
        return 0;
      }
    };

    template <int> struct mconv1d_cuda_algo : cuda_algo_1d {};
    typedef mconv1d_cuda_algo<1> mconv1d_cuda_algo_v1;
    typedef mconv1d_cuda_algo<2> mconv1d_cuda_algo_v2;

    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void MConv1DCuda(
      const T* __restrict__ f, const T* __restrict__ g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T* __restrict__ h, mconv1d_cuda_algo_v1)
    {
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      const I Nh = lmin(MConvSupportSizeCuda(nf, ng, s, d, p, u), nh);
      ++p;
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      for (I i = i_start; i < nh * c_out * batch; i += i_stride) {
        const I hx = i % nh, ip = i / nh;
        if (hx >= Nh) continue;
        const I co = transposed_h ? (ip / batch) : (ip % c_out);
        const I bt = transposed_h ? (ip % batch) : (ip / c_out);
        const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
        const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
        A acc = 0;
        const I num = p * (hx + s);
        const I min_fx = lmax(cdiv(num, ng + u - 1) - d, 0);
        const I max_fx = lmin(cdiv(num + 1, u) - d, nf);
        for (I fx = min_fx; fx < max_fx; ++fx) {
          const I denom = fx + d;
          if (!(num % denom)) {
            const I gx = num / denom - u;
            const T *f_co_x = f_co + fx, *g_bt_x = g_bt + gx;
            for (I ci = 0; ci < c_in; ++ci)
              acc += f_co_x[ci * f_ci_stride] * g_bt_x[ci * g_ci_stride];
          }
        }
        h[i] += acc;
      }
    }

    template <>
    struct mconv1d_cuda_algo<2> : cuda_algo_1d {
      static int GetNumThreads(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s,
        cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u
      ) {
        return std::min(64, kMaxCudaThreadsPerBlock);
      }

      static int GetNumBlocks(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s,
        cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u, int threads
      ) {
        const cuda_ssize_t hx_per_chunk = std::min(nh, static_cast<cuda_ssize_t>(threads));
        const cuda_ssize_t fx_per_chunk = std::min(nf, std::max((threads * threads) / hx_per_chunk, static_cast<cuda_ssize_t>(1)));
        const cuda_ssize_t ci_per_chunk = std::min(c_in, std::max((threads * threads) / (hx_per_chunk * fx_per_chunk), static_cast<cuda_ssize_t>(1)));
        const cuda_ssize_t nf_chunk = (nf + fx_per_chunk - 1) / fx_per_chunk;
        const cuda_ssize_t Nh_chunk = (nh + hx_per_chunk - 1) / hx_per_chunk;
        const cuda_ssize_t c_in_chunk = (c_in + ci_per_chunk - 1) / ci_per_chunk;
        const cuda_ssize_t num_chunks = Nh_chunk * c_in_chunk * nf_chunk;
        return static_cast<int>(std::min(
          batch * c_out * num_chunks,
          static_cast<cuda_ssize_t>(std::numeric_limits<int>::max())
        ));
      }

      template <typename T>
      static int GetDynamicSharedMemory(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s,
        cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u, int threads, int blocks
      ) {
        return threads * threads * sizeof(A);
      }
    };

    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void MConv1DCuda(
      const T* __restrict__ f, const T* __restrict__ g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T* __restrict__ h, mconv1d_cuda_algo_v2)
    {
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      const I Nh = lmin(MConvSupportSizeCuda(nf, ng, s, d, p, u), nh);
      ++p;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;

      extern __shared__ char chunk_c[];
      A *chunk = (A*) chunk_c;
      const I buffer_size = blockDim.x * blockDim.x;
      const I hx_per_chunk = lmin(Nh, blockDim.x);
      const I fx_per_chunk = lmin(nf, lmax(buffer_size / hx_per_chunk, 1));
      const I ci_per_chunk = lmin(c_in, lmax(buffer_size / (hx_per_chunk * fx_per_chunk), 1));
      const I cifx_per_chunk = fx_per_chunk * ci_per_chunk;
      const I nf_chunk = cdiv(nf, fx_per_chunk);
      const I Nh_chunk = cdiv(Nh, hx_per_chunk);
      const I c_in_chunk = cdiv(c_in, ci_per_chunk);
      const I num_chunks = Nh_chunk * c_in_chunk * nf_chunk;
      
      // clear shared memory
      for (I j = threadIdx.x; j < buffer_size; j += blockDim.x)
        chunk[j] = 0;

      for (I i = blockIdx.x; i < batch * c_out * num_chunks; i += gridDim.x) {
        const I ch = i % num_chunks, ip = i / num_chunks;
        T *h_bt_co = h + ip * nh;
        const I co = transposed_h ? (ip / batch) : (ip % c_out);
        const I bt = transposed_h ? (ip % batch) : (ip / c_out);
        const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
        const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
        
        const I ch_fx = ch % nf_chunk,
                chp = ch / nf_chunk;
        const I ch_ci = chp % c_in_chunk, ch_hx = chp / c_in_chunk;
        const I fx_start = ch_fx * fx_per_chunk,
                ci_start = ch_ci * ci_per_chunk,
                hx_start = ch_hx * hx_per_chunk;
        const I fx_end = lmin(fx_start + fx_per_chunk, nf),
                ci_end = lmin(ci_start + ci_per_chunk, c_in),
                hx_end = lmin(hx_start + hx_per_chunk, Nh);

        __syncthreads();  // phase 1 - accumulate into chunks.
        // no thread can touch an fx/ci other than the one it's assigned to
        for (I j = threadIdx.x; j < (ci_end - ci_start) * (fx_end - fx_start); j += blockDim.x) {
          const I fx_offs = j % (fx_end - fx_start),
                  ci_offs = j / (fx_end - fx_start);
          const I fx = fx_start + fx_offs,
                  ci = ci_start + ci_offs;

          const I gcd_x = gcd(p, fx + d);
          const I h_x_stride = (fx + d) / gcd_x;
          const I g_x_stride = p / gcd_x;
          const I min_xs = lmax(
            cdiv(u, g_x_stride),
            cdiv(hx_start + s, h_x_stride)
          );
          const I max_xs = lmin(
            cdiv(ng + u, g_x_stride),
            cdiv(hx_end + s, h_x_stride)
          );
          const T *f_co_ci_x = f_co + ci * f_ci_stride + fx,
                *g_bt_ci_x = g_bt + ci * g_ci_stride + min_xs * g_x_stride - u;
          A *chunk_hx_ci_fx = chunk + ((min_xs * h_x_stride - s - hx_start) * ci_per_chunk + ci_offs) * fx_per_chunk + fx_offs;

          for (I xs = min_xs; xs < max_xs; ++xs) {
            *chunk_hx_ci_fx = (*f_co_ci_x) * (*g_bt_ci_x);
            chunk_hx_ci_fx += h_x_stride * cifx_per_chunk;
            g_bt_ci_x += g_x_stride;
          }
        }

        __syncthreads();  // phase 2 - sum channels into h and clear chunk
        // no thread can touch an hx other than the one it's assigned to
        for (I hx_offs = threadIdx.x; hx_offs < hx_end - hx_start; hx_offs += blockDim.x) {
          const I hx = hx_start + hx_offs;
          A acc = 0, *chunk_hx = chunk + hx_offs * cifx_per_chunk;
          for (I j = 0; j < cifx_per_chunk; ++j) {
            acc += chunk_hx[j];
            chunk_hx[j] = 0;
          }
          h_bt_co[hx] += acc;
        }
      }
    }
    
    template <int> struct mcorr1d_cuda_algo : cuda_algo_1d { };
    typedef mcorr1d_cuda_algo<1> mcorr1d_cuda_algo_v1;
    typedef mcorr1d_cuda_algo<2> mcorr1d_cuda_algo_v2;
    
    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void MCorr1DCuda(
      const T* __restrict__ f, const T* __restrict__ g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T* __restrict__ h, mcorr1d_cuda_algo_v1)
    {
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      const I Nh = min(MCorrSupportSizeCuda(ng, s, d, p, u), nh);
      ++p;
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      for (I i = i_start; i < nh * c_out * batch; i += i_stride) {
        const I hx = i % nh, ip = i / nh;
        if (hx >= Nh) continue;
        const I co = transposed_h ? (ip / batch) : (ip % c_out);
        const I bt = transposed_h ? (ip % batch) : (ip / c_out);
        const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
        const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
        const I min_fx = lmax(cdiv(u * p, s + hx) - d, 0);
        const I max_fx = lmin(cdiv(p * (ng + u), s + hx) - d, nf);
        A acc = 0;
        for (I fx = min_fx; fx < max_fx; ++fx) {
          const I num = (s + hx) * (fx + d);
          if (!(num % p)) {
            const I gx = num / p - u;
            const T *f_co_x = f_co + fx, *g_bt_x = g_bt + gx;
            for (I ci = 0; ci < c_in; ++ci)
              acc += f_co_x[ci * f_ci_stride] * g_bt_x[ci * g_ci_stride];
          }
        }
        h[i] += acc;
      }
    }

    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void MCorr1DCuda(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mcorr1d_cuda_algo_v2)
    {
      // The same as MCorr1d v2 on the CPU
      const I Nh = lmin(MCorrSupportSizeCuda(ng, s, d, p, u), nh);
      ++p;
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      for (I i = i_start; i < nh * c_out * batch; i += i_stride) {
        const I hx = i % nh, ip = i / nh;
        if (hx >= Nh) continue;
        const I co = transposed_h ? (ip / batch) : (ip % c_out);
        const I bt = transposed_h ? (ip % batch) : (ip / c_out);
        const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
        const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
        const I gcd_x = gcd(p, hx + s);
        const I f_x_stride = p / gcd_x;
        const I g_x_stride = (hx + s) / gcd_x;
        const I min_xs = lmax(cdiv(d, f_x_stride), cdiv(u, g_x_stride));
        const I max_xs = lmin(
          cdiv(nf + d, f_x_stride),
          cdiv(ng + u, g_x_stride)
        );
        const T *g_bt_x = g_bt + min_xs * g_x_stride - u;
        const T *f_co_x = f_co + min_xs * f_x_stride - d;
        A acc = 0;
        for (I xs = min_xs; xs < max_xs; ++xs) {
          for (I ci = 0; ci < c_in; ++ci)
            acc += f_co_x[ci * f_ci_stride] * g_bt_x[ci * g_ci_stride];
          g_bt_x += g_x_stride;
          f_co_x += f_x_stride;
        }
        h[i] += acc;
      }
    }
    
    template <int> struct mconvlconv_cuda_algo : cuda_algo_2d { };
    typedef mconvlconv_cuda_algo<1> mconvlconv_cuda_algo_v1;
    
    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void MConvLConvCuda(
      const T* __restrict__ f, const T* __restrict__ g,
      I c_out, I c_in, I batch, I nfx, I nfy, I ngx, I ngy, I nhx, I nhy,
      I sx, I sy, I dx, I dy, I px, I py, I ux, I uy,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T* __restrict__ h, mconvlconv_cuda_algo_v1)
    {
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      const I Nhx = min(MConvSupportSizeCuda(nfx, ngx, sx, dx, px, ux), nhx);
      ++px;
      const I Nhy = min(LConvSupportSizeCuda(nfy, ngy, sy, dy, py, uy), nhy);
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      const I f_x_stride = nfy, g_x_stride = ngy;
      const I nf = nfx * nfy, ng = ngx * ngy;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      for (I i = i_start; i < nhx * nhy * c_out * batch; i += i_stride) {
        const I hy = i % nhy, ip = i / nhy;
        const I hx = ip % nhx, ipp = ip / nhx;
        if (hx >= Nhx || hy >= Nhy) continue;
        const I co = transposed_h ? ipp / batch : ipp % c_out;
        const I bt = transposed_h ? ipp % batch : ipp / c_out;
        const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
        const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
        const I num_x = px * (hx + sx);
        const I min_fx = lmax(cdiv(num_x, ngx + ux - 1) - dx, 0);
        const I max_fx = lmin(cdiv(num_x + 1, ux) - dx, nfx);
        const I num_y_part = sy * hy + py;
        const I min_fy = lmax(cdiv(num_y_part - uy * (ngy - 1), dy), 0);
        const I max_fy = lmin(cdiv(num_y_part + 1, dy), nfy);
        A acc = 0;
        for (I fx = min_fx; fx < max_fx; ++fx) {
          const I denom_x = fx + dx;
          if (num_x % denom_x) continue;
          const I gx = num_x / denom_x - ux;
          const T *f_co_x = f_co + fx * f_x_stride, *g_bt_x = g_bt + gx * g_x_stride;
          for (I fy = min_fy; fy < max_fy; ++fy) {
              const I num_y = num_y_part - dy * fy;
              if (num_y % uy) continue;
              const I gy = num_y / uy;
              const T *f_co_x_y = f_co_x + fy, *g_bt_x_y = g_bt_x + gy;
              for (I ci = 0; ci < c_in; ++ci)
                acc += f_co_x_y[ci * f_ci_stride] * g_bt_x_y[ci * g_ci_stride];
          }
        }
        h[i] += acc;
      }
    }
    
    template <int> struct mcorrlcorr_cuda_algo : cuda_algo_2d { };
    typedef mcorrlcorr_cuda_algo<1> mcorrlcorr_cuda_algo_v1;
    // typedef mcorrlcorr_cuda_algo<2> mcorrlcorr_cuda_algo_v2;
    
    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void MCorrLCorrCuda(
      const T* __restrict__ f, const T* __restrict__ g,
      I c_out, I c_in, I batch, I nfx, I nfy, I ngx, I ngy, I nhx, I nhy,
      I sx, I sy, I dx, I dy, I px, I py, I ux, I uy,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T* __restrict__ h, mcorrlcorr_cuda_algo_v1)
    { 
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      const I Nhx = min(MCorrSupportSizeCuda(ngx, sx, dx, px, ux), nhx);
      ++px;
      const I Nhy = min(LCorrSupportSizeCuda(ngy, sy, py, uy), nhy);
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      const I f_x_stride = nfy, g_x_stride = ngy;
      const I nf = nfx * nfy, ng = ngx * ngy;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      for (I i = i_start; i < nhx * nhy * c_out * batch; i += i_stride) {
        const I hy = i % nhy, ip = i / nhy;
        const I hx = ip % nhx, ipp = ip / nhx;
        if (hx >= Nhx || hy >= Nhy) continue;
        const I co = transposed_h ? ipp / batch : ipp % c_out;
        const I bt = transposed_h ? ipp % batch : ipp / c_out;
        const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
        const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
        const I min_fx = lmax(cdiv(ux * px, sx + hx) - dx, 0);
        const I max_fx = lmin(cdiv(px * (ngx + ux), sx + hx) - dx, nfx);
        const I num_y_part = sy * hy - py;
        const I min_fy = lmax(cdiv(-num_y_part, dy), 0);
        const I max_fy = lmin(cdiv(uy * ngy - num_y_part, dy), nfy);
        A acc = 0;
        for (I fx = min_fx; fx < max_fx; ++fx) {
          const I num_x = (sx + hx) * (fx + dx);
          if (num_x % px) continue;
          const I gx = num_x / px - ux;
          const T *f_co_x = f_co + fx * f_x_stride, *g_bt_x = g_bt + gx * g_x_stride;
          for (I fy = min_fy; fy < max_fy; ++fy) {
              const I num_y = num_y_part + dy * fy;
              if (num_y % uy) continue;
              const I gy = num_y / uy;
              const T *f_co_x_y = f_co_x + fy, *g_bt_x_y = g_bt_x + gy;
              for (I ci = 0; ci < c_in; ++ci)
                acc += f_co_x_y[ci * f_ci_stride] * g_bt_x_y[ci * g_ci_stride];
          }
        }
        h[i] += acc;
      }
    }

    template <int> struct snd2col_cuda_algo : cuda_algo_1d {
      static int GetNumBlocks(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s,
        cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u, int threads
      ) {
        return static_cast<int>(std::min(
          std::max(
            (c_in * batch * nf * nh + threads - 1) / threads,
            static_cast<cuda_ssize_t>(1)
          ),
          static_cast<cuda_ssize_t>(std::numeric_limits<int>::max())
        ));
      };
    };
    typedef snd2col_cuda_algo<1> snd2col_cuda_algo_v1;

    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void Snd2ColCuda(
      const T* __restrict__ g,
      I c_in, I batch, I nf, I ng, I ngg, I s, I d, I p, I u,
      bool transposed_g, bool transposed_gg,
      T* __restrict__ gg, snd2col_cuda_algo_v1)
    {
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      ++p;
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      for (I i = i_start; i < batch * c_in * nf * ngg; i += i_stride) {
        const I ggx = i % ngg, ip = i / ngg;
        const I fx = ip % nf, ipp = ip / nf;
        const I num = (s + ggx) * (fx + d);
        const I gx = (num % p) ? ng : (num / p - u);
        T val;
        if (gx >= 0 && gx < ng) {
          const I ci = transposed_gg ? (ipp / batch) : (ipp % c_in);
          const I bt = transposed_gg ? (ipp % batch) : (ipp / c_in);
          val = g[(transposed_g ? (ci * batch + bt) : (bt * c_in + ci)) * ng + gx];
        } else {
          val = 0;
        }
        gg[i] = val;
      }
    }

    template <int> struct col2snd_cuda_algo : cuda_algo_1d {
      static int GetNumBlocks(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s,
        cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u, int threads
      ) {
        return static_cast<int>(std::min(
          std::max(
            (c_in * batch * ng + threads - 1) / threads,
            static_cast<cuda_ssize_t>(1)
          ),
          static_cast<cuda_ssize_t>(std::numeric_limits<int>::max())
        ));
      };
    };
    typedef col2snd_cuda_algo<1> col2snd_cuda_algo_v1;

    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void Col2SndCuda(
      const T* __restrict__ gg,
      I c_in, I batch, I nf, I ng, I ngg, I s, I d, I p, I u,
      bool transposed_g, bool transposed_gg,
      T* __restrict__ g, col2snd_cuda_algo_v1)
    {
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      ++p;
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      for (I i = i_start; i < batch * c_in * ng; i += i_stride) {
        const I gx = i % ng, ip = i / ng;
        const I ci = transposed_g ? (ip / batch) : (ip % c_in);
        const I bt = transposed_g ? (ip % batch) : (ip / c_in);
        A val = 0;
        const I num = p * (gx + u);
        const T *gg_bt_ci = gg + (transposed_gg ? (ci * batch + bt) : (bt * c_in + ci)) * nf * ngg;
        for (I fx = 0; fx < nf; ++fx) {
          const I denom = fx + d;
          const I ggx = (num % denom) ? ngg : (num / denom - s);
          if (ggx >= 0 && ggx < ngg)
            val += gg_bt_ci[fx * ngg + ggx];
        }
        g[i] = val;
      }
    }

    template <int> struct spec2col_cuda_algo : cuda_algo_2d {
      static int GetNumBlocks(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy,
        cuda_ssize_t nhx, cuda_ssize_t nhy,
        cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy,
        cuda_ssize_t px, cuda_ssize_t py, cuda_ssize_t ux, cuda_ssize_t uy,
        int threads
      ) {
        return static_cast<int>(std::min(
          std::max(
            (c_in * batch * nfx * nfy * nhx * nhy + threads - 1) / threads,
            static_cast<cuda_ssize_t>(1)
          ),
          static_cast<cuda_ssize_t>(std::numeric_limits<int>::max())
        ));
      }
    };
    typedef spec2col_cuda_algo<1> spec2col_cuda_algo_v1;

    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void Spec2ColCuda(
      const T* __restrict__ g,
      I c_in, I batch, I nfx, I nfy, I ngx, I ngy, I nggx, I nggy,
      I sx, I sy, I dx, I dy, I px, I py, I ux, I uy,
      bool transposed_g, bool transposed_gg,
      T* __restrict__ gg, spec2col_cuda_algo_v1)
    {
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      ++px;
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      for (I i = i_start; i < batch * c_in * nfx * nfy * nggx * nggy; i += i_stride) {
        const I ggy = i % nggy; I ip = i / nggy;
        const I ggx = ip % nggx; ip /= nggx;
        const I fy = ip % nfy; ip /= nfy;
        const I fx = ip % nfx; ip /= nfx;
        const I numx = (sx + ggx) * (fx + dx);
        const I gx = (numx % px) ? ngx : (numx / px - ux);
        const I numy = sy * ggy + dy * fy - py;
        const I gy = (numy % uy) ? ngy : (numy / uy);
        T val;
        if (gx >= 0 && gx < ngx && gy >= 0 && gy < ngy) {
          const I ci = transposed_gg ? (ip / batch) : (ip % c_in);
          const I bt = transposed_gg ? (ip % batch) : (ip / c_in);
          val = g[((transposed_g ? (ci * batch + bt) : (bt * c_in + ci)) * ngx + gx) * ngy + gy];
        } else {
          val = 0;
        }
        gg[i] = val;
      }
    }

    template <int> struct col2spec_cuda_algo : cuda_algo_2d {
      static int GetNumBlocks(
        cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
        cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy,
        cuda_ssize_t nhx, cuda_ssize_t nhy,
        cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy,
        cuda_ssize_t px, cuda_ssize_t py, cuda_ssize_t ux, cuda_ssize_t uy,
        int threads
      ) {
        return static_cast<int>(std::min(
          std::max(
            (c_in * batch * ngx * ngy + threads - 1) / threads,
            static_cast<cuda_ssize_t>(1)
          ),
          static_cast<cuda_ssize_t>(std::numeric_limits<int>::max())
        ));
      }
    };
    typedef col2spec_cuda_algo<1> col2spec_cuda_algo_v1;

    template<typename T, typename I>
    __launch_bounds__(kMaxCudaThreadsPerBlock)
    __global__
    void Col2SpecCuda(
      const T* __restrict__ gg,
      I c_in, I batch, I nfx, I nfy, I ngx, I ngy, I nggx, I nggy,
      I sx, I sy, I dx, I dy, I px, I py, I ux, I uy,
      bool transposed_g, bool transposed_gg,
      T* __restrict__ g, col2spec_cuda_algo_v1)
    {
      if (blockIdx.y || threadIdx.y || blockIdx.z || threadIdx.z) return;
      ++px;
      const I i_start = blockDim.x * blockIdx.x + threadIdx.x;
      const I i_stride = blockDim.x * gridDim.x;
      for (I i = i_start; i < batch * c_in * ngx * ngy; i += i_stride) {
        const I gy = i % ngy; I ip = i / ngy;
        const I gx = ip % ngx; ip = ip / ngx;
        const I ci = transposed_g ? (ip / batch) : (ip % c_in);
        const I bt = transposed_g ? (ip % batch) : (ip / c_in);
        const I numx = px * (gx + ux);
        const T *gg_bt_ci = gg + (transposed_gg ? (ci * batch + bt) : (bt * c_in + ci)) * nfx * nfy * nggx * nggy;
        A val = 0;
        for (I fx = 0; fx < nfx; ++fx) {
          const I denomx = fx + dx;
          const I ggx = (numx % denomx) ? nggx : (numx / denomx - sx);
          if (ggx >= 0 && ggx < nggx) {
            const T *gg_bt_ci_x = gg_bt_ci + (fx * nfy * nggx + ggx) * nggy;
            for (I fy = 0; fy < nfy; ++fy) {
              const I numy = uy * gy - dy * fy + py;
              const I ggy = (numy % sy) ? nggy : (numy / sy);
              if (ggy >= 0 && ggy < nggy)
                val += gg_bt_ci_x[fy * nggx * nggy + ggy];
            }
          }
        }
        g[i] = val;
      }
    }


    #undef A
    #undef gcd
    #undef cdiv
    #undef fdiv
    #undef lmax
    #undef lmin
    #ifdef _CDROBERT_MELLIN_OLD_A
      #define A _CDROBERT_MELLIN_OLD_A A
      #undef _CDROBERT_MELLIN_OLD_A
    #endif
    #ifdef _CDROBERT_MELLIN_OLD_GCD
      #define gcd _CDROBERT_MELLIN_OLD_GCD
      #undef _CDROBERT_MELLIN_OLD_GCD
    #endif
    #ifdef _CDROBERT_MELLIN_OLD_CDIV
      #define cdiv _CDROBERT_MELLIN_OLD_CDIV
      #undef _CDROBERT_MELLIN_OLD_CDIV
    #endif
    #ifdef _CDROBERT_MELLIN_OLD_FDIV
      #define fdiv _CDROBERT_MELLIN_OLD_FDIV
      #undef _CDROBERT_MELLIN_OLD_FDIV
    #endif
    #ifdef _CDROBERT_MELLIN_OLD_LMAX
      #define lmax _CDROBERT_MELLIN_OLD_LMAX
      #undef _CDROBERT_MELLIN_OLD_LMAX
    #endif
    #ifdef _CDROBERT_MELLIN_OLD_LMIN
      #define lmin _CDROBERT_MELLIN_OLD_LMIN
      #undef _CDROBERT_MELLIN_OLD_LMIN
    #endif
  }  // namespace detail

  #define _CDROBERT_MELLIN_DISPATCH(N, ...) \
    if (N > std::numeric_limits<int>::max()) { \
      using I = cuda_ssize_t; \
      __VA_ARGS__(); \
    } else { \
      using I = int; \
      __VA_ARGS__(); \
    }

  template <class T>
  void MConv1DCuda(
    const T *f, const T *g,
    cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s, cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u,
    bool transposed_f, bool transposed_g, bool transposed_h, T *h,
    cudaStream_t stream)
  {
    using algo_ver = detail::mconv1d_cuda_algo<kMConv1DCudaAlgorithmVersion>;
    static_assert(std::is_arithmetic<T>::value, "f, g, and h type must be arithmetic");
    int threads = kCudaSerial ? 1 : algo_ver::GetNumThreads(c_out, c_in, batch, nf, ng, nh, s, d, p, u);
    int blocks = kCudaSerial ? 1 : algo_ver::GetNumBlocks(c_out, c_in, batch, nf, ng, nh, s, d, p, u, threads);
    int shared_memory = algo_ver::GetDynamicSharedMemory<T>(c_out, c_in, batch, nf, ng, nh, s, d, p, u, threads, blocks);
    
    _CDROBERT_MELLIN_DISPATCH(c_out * batch * nh, [&]{
      detail::MConv1DCuda<T, I><<<blocks, threads, shared_memory, stream>>>(
        f, g, c_out, c_in, batch, nf, ng, nh, s, d, p, u,
        transposed_f, transposed_g, transposed_h, h, algo_ver()
      );
    });
  }

  template <class T>
  void MCorr1DCuda(
    const T *f, const T *g,
    cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t nh, cuda_ssize_t s, cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u,
    bool transposed_f, bool transposed_g, bool transposed_h, T *h,
    cudaStream_t stream)
  {
    using algo_ver = detail::mcorr1d_cuda_algo<kMCorr1DCudaAlgorithmVersion>;
    static_assert(std::is_arithmetic<T>::value, "f, g, and h type must be arithmetic");
    int threads = kCudaSerial ? 1 : algo_ver::GetNumThreads(c_out, c_in, batch, nf, ng, nh, s, d, p, u);
    int blocks = kCudaSerial ? 1 : algo_ver::GetNumBlocks(c_out, c_in, batch, nf, ng, nh, s, d, p, u, threads);
    int shared_memory = algo_ver::GetDynamicSharedMemory<T>(c_out, c_in, batch, nf, ng, nh, s, d, p, u, threads, blocks);

    _CDROBERT_MELLIN_DISPATCH(c_out * batch * nh, [&]{
      detail::MCorr1DCuda<T><<<blocks, threads, shared_memory, stream>>>(
        f, g, c_out, c_in, batch, nf, ng, nh, s, d, p, u,
        transposed_f, transposed_g, transposed_h, h, algo_ver()
      );
    });
  }

  template <class T>
  void MConvLConvCuda(
    const T *f, const T *g,
    cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy, cuda_ssize_t nhx, cuda_ssize_t nhy,
    cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy, cuda_ssize_t px, cuda_ssize_t py, cuda_ssize_t ux, cuda_ssize_t uy,
    bool transposed_f, bool transposed_g, bool transposed_h, T *h,
    cudaStream_t stream)
  {
    using algo_ver = detail::mconvlconv_cuda_algo<kMConvLConvCudaAlgorithmVersion>;
    static_assert(std::is_arithmetic<T>::value, "f, g, and h type must be arithmetic");
    int threads = kCudaSerial ? 1 : algo_ver::GetNumThreads(c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy);
    int blocks = kCudaSerial ? 1 : algo_ver::GetNumBlocks(c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy, threads);
    int shared_memory = algo_ver::GetDynamicSharedMemory<T>(c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy, threads, blocks);
  
    _CDROBERT_MELLIN_DISPATCH(c_out * batch * nhx * nhy, [&]{
      detail::MConvLConvCuda<T><<<blocks, threads, shared_memory, stream>>>(
        f, g, c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy,
        sx, sy, dx, dy, px, py, ux, uy,
        transposed_f, transposed_g, transposed_h, h, algo_ver()
      );
    });
  }

  template <class T>
  void MCorrLCorrCuda(
    const T *f, const T *g,
    cuda_ssize_t c_out, cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nfx, cuda_ssize_t nfy, cuda_ssize_t ngx, cuda_ssize_t ngy, cuda_ssize_t nhx, cuda_ssize_t nhy,
    cuda_ssize_t sx, cuda_ssize_t sy, cuda_ssize_t dx, cuda_ssize_t dy, cuda_ssize_t px, cuda_ssize_t py, cuda_ssize_t ux, cuda_ssize_t uy,
    bool transposed_f, bool transposed_g, bool transposed_h, T *h,
    cudaStream_t stream)
  {
    using algo_ver = detail::mcorrlcorr_cuda_algo<kMCorrLCorrCudaAlgorithmVersion>;
    static_assert(std::is_arithmetic<T>::value, "f, g, and h type must be arithmetic");
    int threads = kCudaSerial ? 1 : algo_ver::GetNumThreads(c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy);
    int blocks = kCudaSerial ? 1 : algo_ver::GetNumBlocks(c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy, threads);
    int shared_memory = algo_ver::GetDynamicSharedMemory<T>(c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx, dy, px, py, ux, uy, threads, blocks);
  
    _CDROBERT_MELLIN_DISPATCH(c_out * batch * nhx * nhy, [&]{
      detail::MCorrLCorrCuda<T><<<blocks, threads, shared_memory, stream>>>(
        f, g, c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy,
        sx, sy, dx, dy, px, py, ux, uy,
        transposed_f, transposed_g, transposed_h, h, algo_ver()
      );
    });
  }

  template <class T>
  void Snd2ColCuda(
    const T *g,
    cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t ngg,
    cuda_ssize_t s, cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u,
    bool transposed_g, bool transposed_gg, T *gg,
    cudaStream_t stream)
  {
    using algo_ver = detail::snd2col_cuda_algo<kSnd2ColCudaAlgorithmVersion>;
    static_assert(std::is_arithmetic<T>::value, "g and gg type must be arithmetic");
    int threads = kCudaSerial ? 1 : algo_ver::GetNumThreads(1, c_in, batch, nf, ng, ngg, s, d, p, u);
    int blocks = kCudaSerial ? 1 : algo_ver::GetNumBlocks(1, c_in, batch, nf, ng, ngg, s, d, p, u, threads);
    int shared_memory = algo_ver::GetDynamicSharedMemory<T>(1, c_in, batch, nf, ng, ngg, s, d, p, u, threads, blocks);

    _CDROBERT_MELLIN_DISPATCH(c_in * batch * nf * ngg, [&]{
      detail::Snd2ColCuda<T><<<blocks, threads, shared_memory, stream>>>(
        g, c_in, batch, nf, ng, ngg, s, d, p, u,
        transposed_g, transposed_gg, gg, algo_ver()
      );
    });
  }

  template <class T>
  void Col2SndCuda(
    const T *gg,
    cuda_ssize_t c_in, cuda_ssize_t batch,
    cuda_ssize_t nf, cuda_ssize_t ng, cuda_ssize_t ngg,
    cuda_ssize_t s, cuda_ssize_t d, cuda_ssize_t p, cuda_ssize_t u,
    bool transposed_g, bool transposed_gg, T *g,
    cudaStream_t stream)
  {
    using algo_ver = detail::col2snd_cuda_algo<kCol2SndCudaAlgorithmVersion>;
    static_assert(std::is_arithmetic<T>::value, "g and gg type must be arithmetic");
    int threads = kCudaSerial ? 1 : algo_ver::GetNumThreads(1, c_in, batch, nf, ng, ngg, s, d, p, u);
    int blocks = kCudaSerial ? 1 : algo_ver::GetNumBlocks(1, c_in, batch, nf, ng, ngg, s, d, p, u, threads);
    int shared_memory = algo_ver::GetDynamicSharedMemory<T>(1, c_in, batch, nf, ng, ngg, s, d, p, u, threads, blocks);

    _CDROBERT_MELLIN_DISPATCH(c_in * batch * nf * ngg, [&]{
      detail::Col2SndCuda<T><<<blocks, threads, shared_memory, stream>>>(
        gg, c_in, batch, nf, ng, ngg, s, d, p, u,
        transposed_g, transposed_gg, g, algo_ver()
      );
    });
  }


  template <typename T>
  void Spec2ColCuda(
    const T *g,
    ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nggx, ssize_t nggy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_g, bool transposed_gg,
    T *gg,
    cudaStream_t stream)
  {
    using algo_ver = detail::spec2col_cuda_algo<kSpec2ColCudaAlgorithmVersion>;
    static_assert(std::is_arithmetic<T>::value, "g and gg type must be arithmetic");
    int threads = kCudaSerial ? 1 : algo_ver::GetNumThreads(1, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy, sx, sy, dx, dy, px, py, ux, uy);
    int blocks = kCudaSerial ? 1 : algo_ver::GetNumBlocks(1, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy, sx, sy, dx, dy, px, py, ux, uy, threads);
    int shared_memory = algo_ver::GetDynamicSharedMemory<T>(1, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy, sx, sy, dx, dy, px, py, ux, uy, threads, blocks);

    _CDROBERT_MELLIN_DISPATCH(c_in * batch * nfx * nfy * nggx * nggy, [&]{
      detail::Spec2ColCuda<T><<<blocks, threads, shared_memory, stream>>>(
        g, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy,
        sx, sy, dx, dy, px, py, ux, uy,
        transposed_g, transposed_gg, gg, algo_ver()
      );
    });
  }

  template <typename T>
  void Col2SpecCuda(
    const T *gg,
    ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nggx, ssize_t nggy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_g, bool transposed_gg,
    T *g,
    cudaStream_t stream)
  {
    using algo_ver = detail::col2spec_cuda_algo<kCol2SpecCudaAlgorithmVersion>;
    static_assert(std::is_arithmetic<T>::value, "g and gg type must be arithmetic");
    int threads = kCudaSerial ? 1 : algo_ver::GetNumThreads(1, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy, sx, sy, dx, dy, px, py, ux, uy);
    int blocks = kCudaSerial ? 1 : algo_ver::GetNumBlocks(1, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy, sx, sy, dx, dy, px, py, ux, uy, threads);
    int shared_memory = algo_ver::GetDynamicSharedMemory<T>(1, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy, sx, sy, dx, dy, px, py, ux, uy, threads, blocks);

    _CDROBERT_MELLIN_DISPATCH(c_in * batch * ngx * ngy, [&]{
      detail::Col2SpecCuda<T><<<blocks, threads, shared_memory, stream>>>(
        gg, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy,
        sx, sy, dx, dy, px, py, ux, uy,
        transposed_g, transposed_gg, g, algo_ver()
      );
    });
  }

#undef _CDROBERT_MELLIN_DISPATCH
}}  // namespace cdrobert::mellin

#endif  // CDROBERT_MELLIN_MCONV_CUDA_CUH_
