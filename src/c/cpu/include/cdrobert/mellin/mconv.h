// Copyright 2018 Sean Robertson
/** \file cdrobert/mellin/mconv.h
  \brief Convolutional kernels on the CPU
*/
#pragma once

#ifndef CDROBERT_MELLIN_MCONV_H_
#define CDROBERT_MELLIN_MCONV_H_

#include <cstddef>
#include <type_traits>
#include <vector>

#include "cdrobert/mellin/config_cpu.h"

namespace cdrobert { namespace mellin {
  using ssize_t = std::ptrdiff_t;

  // BEGIN PUBLIC INTERFACE

  /**
    \brief Discrete Mellin convolution over 1D signals

      h[n, co, hx] = sum_ci,fx f[co, ci, fx]
                               g[n, ci, (p + 1)(hx + s)/(fx + d) - u]

    We extend f and g to include channels and batches.

    \param f[in] c-contiguous array of shape (c_out, c_in, nf) if transposed_f is
                false, otherwise (c_in, c_out, nf)
    \param g[in] c-contiguous array of shape (batch, c_in, ng) if transposed_g is
                false, otherwise (c_in, batch, ng)
    \param c_out[in] number of output channels
    \param c_in[in] number of input channels
    \param batch[in] batch size
    \param nf[in] length of f in the Mellin direction
    \param ng[in] length of g in the Mellin direction
    \param nh[in] length of h allocated in the Mellin direction
    \param s[in]  the "stride" analogue in the Mellin direction
    \param d[in]  the "dilation" analogue in the Mellin direction
    \param p[in]  the "padding" analogue in the Mellin direction
    \param u[in]  the "upsampling" (dilation of g) analogue in the Mellin direction
    \param transposed_f[in] whether the first two dimensions of f are transposed
    \param transposed_g[in] whether the first two dimensions of g are transposed
    \param transposed_h[in] whether the first two dimensions of h are transposed
    \param h[in,out] c-contiguous array of shape (batch, c_out, nh) if
                    transposed_h is false, otherwise (c_out, batch, nh). h is
                    assumed to be properly allocated and initialized; summed
                    values are added to the existing values of h. This is so that
                    things like bias can be added to h prior to calling
  */
  template <typename T>
  void MConv1D(
    const T *f, const T *g,
    ssize_t c_out, ssize_t c_in, ssize_t batch,
    ssize_t nf, ssize_t ng, ssize_t nh,
    ssize_t s, ssize_t d, ssize_t p, ssize_t u,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h
  );

  /**
    \brief Discrete Mellin correlation over 1D signals

      h[n, co, hx] = sum_ci,fx f[co, ci, fx] g[(hx + s)(fx + d)/(p + 1) - u]

    We extend f and g to include channels and batches.

    \param f[in] c-contiguous array of shape (c_out, c_in, nf) if transposed_f is
                false, otherwise (c_in, c_out, nf)
    \param g[in] c-contiguous array of shape (batch, c_in, ng) if transposed_g is
                false, otherwise (c_in, batch, ng)
    \param c_out[in] number of output channels
    \param c_in[in] number of input channels
    \param batch[in] batch size
    \param nf[in] length of f in the Mellin direction
    \param ng[in] length of g in the Mellin direction
    \param nh[in] length of h allocated in the Mellin direction
    \param s[in]  the "stride" analogue in the Mellin direction
    \param d[in]  the "dilation" analogue in the Mellin direction
    \param p[in]  the "padding" analogue in the Mellin direction
    \param u[in]  the "upsampling" (dilation of g) analogue in the Mellin direction
    \param transposed_f[in] whether the first two dimensions of f are transposed
    \param transposed_g[in] whether the first two dimensions of g are transposed
    \param transposed_h[in] whether the first two dimensions of h are transposed
    \param h[in,out] c-contiguous array of shape (batch, c_out, nh) if
                    transposed_h is false, otherwise (c_out, batch, nh). h is
                    assumed to be properly allocated and initialized; summed
                    values are added to the existing values of h. This is so that
                    things like bias can be added to h prior to calling
  */
  template <typename T>
  void MCorr1D(
    const T *f, const T *g,
    ssize_t c_out, ssize_t c_in, ssize_t batch,
    ssize_t nf, ssize_t ng, ssize_t nh,
    ssize_t s, ssize_t d, ssize_t p, ssize_t u,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h
  );

  /**
    \brief Discrete Mellin convolution over one axis, linear convolution over next

      h[n, co, hx, hy] = sum_ci,fx,fy f[co, ci, fx, fy]
                                      g[n, ci, (px + 1)(hx + sx)/(fx + dx) - ux,
                                               (sy hy - dy fy + py) / uy]

    Where we use -1 and py to shift the response to the region after and
    including the 0-index. fx and hx index the nfx and nhx axes of f and h,
    whereas fy and hy index the nfy and nhy axes of f and h. We extend f and g
    to include channels and batches: n for batch, ci for c_in, and co for
    c_out.

    \param f[in] c-contiguous array of shape (c_out, c_in, nfx, nfy) if
                transposed_f is false, otherwise (c_in, c_out, nfx, nfy)
    \param g[in] c-contiguous array of shape (batch, c_in, ngx, ngy) if
                transposed_g is false, otherwise (c_in, batch, ngx, ngy)
    \param c_out[in] number of output channels
    \param c_in[in] number of input channels
    \param batch[in] batch size
    \param nfx[in] the length of f in the Mellin direction
    \param nfy[in] the length of f in the linear direction
    \param ngx[in] the length of g in the Mellin direction
    \param ngy[in] the length of g in the linear direction
    \param nhx[in] the length of h allocated in the Mellin direction
    \param nhy[in] the length of h allocated in the linear direction
    \param dx[in] the "dilation" analogue in the Mellin direction
    \param dy[in] the dilation of the kernel in the y direction. Controls the
                  factor by which the kernel is expanded in the y direction,
                  padded with dy - 1 many zeros between elements
    \param sx[in] the "stride" analogue in the Mellin direction
    \param sy[in] the stride of the kernel in the y direction. Controls the
                  factor by which the linear convolution will downsample the
                  result along the y axis
    \param px[in] the "padding" analogue in the Mellin direction
    \param py[in] shifts the response of h in the linear direction this number
                  of samples backward (e.g. hy to hy - 1).
    \param ux[in] the "upsampling" (dilation of g) analogue in the Mellin
                  direction
    \param uy[in] upsampling, or the dilation of the signal in the linear
                  direction. Controls the factor by which the signal is expanded
                  in the y direction, padded with uy - 1 many zeros between
                  elements
    \param transposed_f[in] whether the first two dimensions of f are transposed
    \param transposed_g[in] whether the first two dimensions of g are transposed
    \param transposed_h[in] whether the first two dimensions of h are transposed
    \param h[in,out] c-contiguous array of shape (batch, c_out, nhx, nhy) if
                    transposed_h is false, otherwise (c_out, batch, nhx, nhy). h
                    is assumed to be properly allocated and initialized; summed
                    values are added to the existing values of h. This is so
                    that things like bias can be added to h prior to calling
  */
  template <typename T>
  void MConvLConv(
    const T *f, const T *g,
    ssize_t c_out, ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nhx, ssize_t nhy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h
  );

  /**
    \brief Discrete Mellin correlation over one axis, linear correlation over next

      h[n, co, hx, hy] =
        sum_ci,fx,fy f[co, ci, fx, fy]
                     g[n, ci, (hx + sx)(fx + dx)/(px + 1) - ux,
                          (sy hy + dy fy - py) / uy]

    Where we use -1 and py to shift the response to the region after and
    including the 0-index. fx and hx index the nfx and nhx axes of f and h,
    whereas fy and hy index the nfy and nhy axes of f and h. We extend f and g
    to include channels and batches: n for batch, ci for c_in, and co for
    c_out.

    \param f[in] c-contiguous array of shape (c_out, c_in, nfx, nfy) if
                transposed_f is false, otherwise (c_in, c_out, nfx, nfy)
    \param g[in] c-contiguous array of shape (batch, c_in, ngx, ngy) if
                transposed_g is false, otherwise (c_in, batch, ngx, ngy)
    \param c_out[in] number of output channels
    \param c_in[in] number of input channels
    \param batch[in] batch size
    \param nfx[in] the length of f in the Mellin direction
    \param nfy[in] the length of f in the linear direction
    \param ngx[in] the length of g in the Mellin direction
    \param ngy[in] the length of g in the linear direction
    \param nhx[in] the length of h allocated in the Mellin direction
    \param nhy[in] the length of h allocated in the linear direction
    \param dx[in] the "dilation" analogue in the Mellin direction
    \param dy[in] the dilation of the kernel in the y direction. Controls the
                  factor by which the kernel is expanded in the y direction,
                  padded with dy - 1 many zeros between elements
    \param sx[in] the "stride" analogue in the Mellin direction
    \param sy[in] the stride of the kernel in the y direction. Controls the
                  factor by which the linear convolution will downsample the
                  result along the y axis
    \param px[in] the "padding" analog in the Mellin direction
    \param py[in] shifts the response of h in the linear direction this number
                  of samples left (e.g. hy to hy - 1).
    \param ux[in] the "upsampling" (dilation of g) analogue in the Mellin
                  direction
    \param uy[in] upsampling, or the dilation of the signal in the linear
                  direction. Controls the factor by which the signal is expanded
                  in the y direction, padded with uy - 1 many zeros between
                  elements
    \param transposed_f[in] whether the first two dimensions of f are transposed
    \param transposed_g[in] whether the first two dimensions of g are transposed
    \param transposed_h[in] whether the first two dimensions of h are transposed
    \param h[in,out] c-contiguous array of shape (batch, c_out, nhx, nhy) if
                    transposed_h is false, otherwise (c_out, batch, nhx, nhy). h
                    is assumed to be properly allocated and initialized; summed
                    values are added to the existing values of h. This is so
                    that things like bias can be added to h prior to calling
  */
  template <typename T>
  void MCorrLCorr(
    const T *f, const T *g,
    ssize_t c_out, ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nhx, ssize_t nhy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h
  );

  /**
    \brief Gather signal tensor for GEMM-style Mellin Correlation.

    Similar to im2col, this function converts a tensor (g) of shape (batch,
    c_in, ng) into one (gg) of shape (batch, c_in, nf, ngg). If a filter (f)
    with shape (c_out, c_in, nf) and gg are matrix-multiplied together and
    added to a buffer:

      h[n, co, hx] += sum_ci,fx f[co, ci, fx] gg[n, ci, fx, hx]
    
    Then the result is identical to that of applying MCorr1D.

    The goal of doing it this way is to (hopefully) speed up computation.

    \param g[in] c-contiguous array of shape (batch, c_in, ng) if transposed_g
                 is false, otherwise (c_in, batch, ng)
    \param c_in[in] number of input channels
    \param batch[in] batch size
    \param nf[in] length of f in the Mellin direction
    \param ng[in] length of g in the Mellin direction
    \param ngg[in] length of gg allocated in the Mellin direction
    \param s[in]  the "stride" analogue in the Mellin direction
    \param d[in]  the "dilation" analogue in the Mellin direction
    \param p[in]  the "padding" analogue in the Mellin direction
    \param u[in]  the "upsampling" (dilation of g) analogue in the Mellin direction
    \param transposed_g[in] whether the first two dimensions of g are transposed
    \param transposed_gg[in] whether the first two dimensions of gg are transposed
    \param gg[in,out] c-contiguous array of shape (batch, c_out, nf, ngg) if
                      transposed_gg is false, otherwise (c_out, batch, nf,
                      ngg). gg is assumed to be properly allocated but does not
                      need to be initialized.
  */
  template <typename T>
  void Snd2Col(
    const T *g,
    ssize_t c_in, ssize_t batch,
    ssize_t nf, ssize_t ng, ssize_t ngg,
    ssize_t s, ssize_t d, ssize_t p, ssize_t u,
    bool transposed_g, bool transposed_gg,
    T *gg
  );

  /**
    \brief Scatter and sum snd2col-style tensor into a signal.

    Similar to an inverse of "snd2col," this function converts a tensor (gg) of
    shape (batch, c_in, nf, ngg) back into one (g) of shape (batch, c_in, ng).
    The operation effectively scatters the values of gg into a tensor of shape
    (batch, c_in, ng, ngg) and then sums the final dimension. It is not quite
    an inverse since "col2snd" might duplicate values to produce gg which will
    be summed together in col2snd. This function is used in backpropagation
    through snd2col.

    \param gg[in] c-contiguous array of shape (batch, c_out, nf, ngg) if
                  transposed_gg is false, otherwise (c_out, batch, nf, ngg).
    \param c_in[in] number of input channels
    \param batch[in] batch size
    \param nf[in] length of f in the Mellin direction
    \param ng[in] length of g in the Mellin direction
    \param ngg[in] length of gg allocated in the Mellin direction
    \param s[in]  the "stride" analogue in the Mellin direction
    \param d[in]  the "dilation" analogue in the Mellin direction
    \param p[in]  the "padding" analogue in the Mellin direction
    \param u[in]  the "upsampling" (dilation of g) analogue in the Mellin direction
    \param transposed_g[in] whether the first two dimensions of g are transposed
    \param transposed_gg[in] whether the first two dimensions of gg are transposed
    \param g[in,out] c-contiguous array of shape (batch, c_in, ng) if transposed_g
                     is false, otherwise (c_in, batch, ng). g is assumed to be
                     allocated but does not need to be initialized.

  */
  template <typename T>
  void Col2Snd(
    const T *gg,
    ssize_t c_in, ssize_t batch,
    ssize_t nf, ssize_t ng, ssize_t ngg,
    ssize_t s, ssize_t d, ssize_t p, ssize_t u,
    bool transposed_g, bool transposed_gg,
    T *g
  );

  /**
    \brief Gather spectrogram tensor for GEMM-style Mellin-Linear Correlation.

    Similar to im2col, this function converts a tensor (g) of shape (batch,
    c_in, ngx, ngy) into one (gg) of shape (batch, c_in, nf, nggx, nggy). If a
    filter (f) with shape (c_out, c_in, nfx, nfy) and gg are matrix-multiplied
    together and added to a buffer:

      h[n, co, hx, hy] += sum_ci,fx,fy f[co, ci, fx, fy]
                                       gg[n, ci, fx, fy, hx, hy]
    
    Then the result is identical to that of MCorrLCorr.

    The goal of doing it this way is to (hopefully) speed up computation.

    \param g[in] c-contiguous array of shape (batch, c_in, ngx, ngy) if
                transposed_g is false, otherwise (c_in, batch, ngx, ngy)
    \param c_in[in] number of input channels
    \param batch[in] batch size
    \param nfx[in] the length of f in the Mellin direction
    \param nfy[in] the length of f in the linear direction
    \param ngx[in] the length of g in the Mellin direction
    \param ngy[in] the length of g in the linear direction
    \param nggx[in] the length of gg allocated in the Mellin direction
    \param nggy[in] the length of gg allocated in the linear direction
    \param dx[in] the "dilation" analogue in the Mellin direction
    \param dy[in] the dilation of the kernel in the y direction. Controls the
                  factor by which the kernel is expanded in the y direction,
                  padded with dy - 1 many zeros between elements
    \param sx[in] the "stride" analogue in the Mellin direction
    \param sy[in] the stride of the kernel in the y direction. Controls the
                  factor by which the linear convolution will downsample the
                  result along the y axis
    \param px[in] the "padding" analog in the Mellin direction
    \param py[in] shifts the response of h in the linear direction this number
                  of samples left (e.g. hy to hy - 1).
    \param ux[in] the "upsampling" (dilation of g) analogue in the Mellin
                  direction
    \param uy[in] upsampling, or the dilation of the signal in the linear
                  direction. Controls the factor by which the signal is expanded
                  in the y direction, padded with uy - 1 many zeros between
                  elements
    \param transposed_g[in] whether the first two dimensions of g are transposed
    \param transposed_gg[in] whether the first two dimensions of gg are transposed
    \param gg[in,out] c-contiguous array of shape (batch, c_out, nfx, nfy, nggx,
                    nggy) if transposed_gg is false, otherwise (c_out, batch,
                    nfx, nfy, nggx, nggy). gg is assumed to be properly
                    allocated but does not need to be initialized.
  */
  template <typename T>
  void Spec2Col(
    const T *g,
    ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nggx, ssize_t nggy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_g, bool transposed_gg,
    T *gg
  );

  /**
    \brief Scatter and sum spec2col-like tensor into a spectrogram

    Similar to an inverse of "spec2col," this function converts a tensor (gg)
    of shape (batch, c_in, nfx, nfy, nggx, nggy) back into one (g) of shape
    (batch, c_in, ngx, ngy). The operation effectively scatters the values of
    gg into a tensor of shape (batch, c_in, ngx, ngy, nggx, nggy) and then sums
    the final two dimensions. It is not quite an inverse since col2spec might
    duplicate values to produce gg which will be summed together in col2spec.
    This function is used in backpropagation through spec2col.

    \param gg[in] c-contiguous array of shape (batch, c_out, nfx, nfy, nggx,
                  nggy) if transposed_gg is false, otherwise (c_out, batch,
                  nfx, nfy, nggx, nggy).
    \param c_in[in] number of input channels
    \param batch[in] batch size
    \param nfx[in] the length of f in the Mellin direction
    \param nfy[in] the length of f in the linear direction
    \param ngx[in] the length of g in the Mellin direction
    \param ngy[in] the length of g in the linear direction
    \param nggx[in] the length of gg allocated in the Mellin direction
    \param nggy[in] the length of gg allocated in the linear direction
    \param dx[in] the "dilation" analogue in the Mellin direction
    \param dy[in] the dilation of the kernel in the y direction. Controls the
                  factor by which the kernel is expanded in the y direction,
                  padded with dy - 1 many zeros between elements
    \param sx[in] the "stride" analogue in the Mellin direction
    \param sy[in] the stride of the kernel in the y direction. Controls the
                  factor by which the linear convolution will downsample the
                  result along the y axis
    \param px[in] the "padding" analog in the Mellin direction
    \param py[in] shifts the response of h in the linear direction this number
                  of samples left (e.g. hy to hy - 1).
    \param ux[in] the "upsampling" (dilation of g) analogue in the Mellin
                  direction
    \param uy[in] upsampling, or the dilation of the signal in the linear
                  direction. Controls the factor by which the signal is expanded
                  in the y direction, padded with uy - 1 many zeros between
                  elements
    \param transposed_g[in] whether the first two dimensions of g are transposed
    \param transposed_gg[in] whether the first two dimensions of gg are transposed
    \param g[in,out] c-contiguous array of shape (batch, c_in, ngx, ngy) if
                     transposed_g is false, otherwise (c_in, batch, ngx, ngy).
                     g is assumed to be allocated but does not need to be
                     initialized.
  */
  template <typename T>
  void Col2Spec(
    const T *gg,
    ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nggx, ssize_t nggy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_g, bool transposed_gg,
    T *g
  );

  // forward dec.
  namespace detail {
    template<typename T1, typename T2> constexpr T1 FloorDiv(T1 num, T2 denom);
  }

  /**
    \brief The min size of h that contains all valid indices for a Mellin conv

    A valid index of output h is some index i whose coefficient is computed
    using only values within the support of g. The minimal size of h that
    contains all valid indices is therefore the maximal valid index i + 1.

    \param ng[in] [0, ng - 1] is the support of g along the Mellin dimension.
    \param s[in]  the "stride" analogue in the Mellin direction
    \param d[in]  the "dilation" analogue in the Mellin direction
    \param p[in]  the "padding" analogue in the Mellin direction
    \param u[in]  the "upsampling" (dilation of g) analogue in the Mellin direction
    \return nh, the minimum length of h containing all valid indices of h.

    \note The Mellin convolution is computed using sums of

      f[k]g[(p + 1)(i + s)/(k + d) - u] = f[k]g[m]
    
    Setting m <= ng - 1 and k = 0 can solve for nh.
  */
  constexpr ssize_t MConvValidSize(
    ssize_t ng, ssize_t s, ssize_t d, ssize_t p, ssize_t u
  ) {
    return std::max(
      detail::FloorDiv(d * (ng + u - 1), p + 1) - s + 1,
      static_cast<ssize_t>(0)
    );
  }

  /**
    \brief The min size of h that contains all nonzero indices for a Mellin conv

    A supported index of output h is some index i whose coefficient could be
    nonzero for compactly supported f and g. The minimal size of h that
    contains all supported indices is therefore the maximal supported index i +
    1.

    \param nf[in] [0, nf - 1] is the support of f along the Mellin dimension.
    \param ng[in] [0, ng - 1] is the support of g along the Mellin dimension.
    \param s[in]  the "stride" analogue in the Mellin direction
    \param d[in]  the "dilation" analogue in the Mellin direction
    \param p[in]  the "padding" analogue in the Mellin direction
    \param u[in]  the "upsampling" (dilation of g) analogue in the Mellin direction
    \return nh, the minimum length of h containing all supported indices of h.

    \note The Mellin convolution is computed using sums of

      f[k]g[(p + 1)(i + s)/(k + d) - u] = f[k]g[m]
    
    Setting m <= ng - 1 and k = nf - 1 can solve for nh.
  */
  constexpr ssize_t MConvSupportSize(
    ssize_t nf, ssize_t ng, ssize_t s, ssize_t d, ssize_t p, ssize_t u
  ) {
    return std::max(
      detail::FloorDiv((nf + d - 1) * (ng + u - 1), p + 1) - s + 1,
      static_cast<ssize_t>(0)
    );
  }

  /**
    \brief The min size of h that contains all valid indices for a linear conv

    A valid index of output h is some index i whose coefficient is computed
    using only values within the support of g. The minimal size of h that
    contains all valid indices is therefore the maximal valid index i + 1.

    \param ng[in] [0, ng - 1] is the support of g along the linear dimension.
    \param s[in]  The stride parameter.
    \param p[in]  The (left-)padding parameter.
    \param u[in]  The upsampling parameter.
    \return nh, the minimum length of h containing all valid indices of h.

    \note The linear convolution is computed using sums of

      f[k]g[(si - dk + p) / u] = f[k]g[m]
    
    Setting m <= ng - 1 and k = 0 can solve for nh.
  */
  constexpr ssize_t LConvValidSize(
    ssize_t ng, ssize_t s, ssize_t p, ssize_t u
  ) {
    return std::max(
      detail::FloorDiv(u * (ng - 1) - p, s) + 1,
      static_cast<ssize_t>(0)
    );
  }

  /**
    \brief The min size of h that contains all nonzero indices for a linear conv

    A supported index of output h is some index i whose coefficient could be
    nonzero for compactly supported f and g. The minimal size of h that
    contains all supported indices is therefore the maximal supported index i +
    1.

    \param nf[in] [0, nf - 1] is the support of f along the linear dimension.
    \param ng[in] [0, ng - 1] is the support of g along the linear dimension.
    \param s[in]  The stride parameter.
    \param d[in]  The dilation parameter.
    \param p[in]  The (left-)padding parameter.
    \param u[in]  The upsampling parameter.
    \return nh, the minimum length of h containing all supported indices of h.

    \note The linear convolution is computed using sums of

      f[k]g[(si - dk + p) / u] = f[k]g[m]
    
    Setting m <= ng - 1 and k = nf - 1 can solve for nh.
  */
  constexpr ssize_t LConvSupportSize(
    ssize_t nf, ssize_t ng, ssize_t s, ssize_t d, ssize_t p, ssize_t u
  ) {
    return std::max(
      detail::FloorDiv(u * (ng - 1) + d * (nf - 1) - p, s) + 1,
      static_cast<ssize_t>(0)
    );
  }

  /**
    \brief The min size of h that contains all valid indices for a Mellin corr

    A valid index of output h is some index i whose coefficient is computed
    using only values within the support of g. The minimal size of h that
    contains all valid indices is therefore the maximal valid index i + 1.

    \param nf[in] [0, nf - 1] is the support of f along the Mellin dimension.
    \param ng[in] [0, ng - 1] is the support of g along the Mellin dimension.
    \param s[in]  the "stride" analogue in the Mellin direction
    \param d[in]  the "dilation" analogue in the Mellin direction
    \param p[in]  the "padding" analogue in the Mellin direction
    \param u[in]  the "upsampling" (dilation of g) analogue in the Mellin direction
    \return nh, the minimum length of h containing all valid indices of h.

    \note The Mellin correlation is computed using sums of

      f[k]g[(i + s)(k + d)/(p + 1) - u] = f[k]g[m]
    
    Setting m <= ng - 1 and k = nf - 1 can solve for nh.
  */
  constexpr ssize_t MCorrValidSize(
    ssize_t nf, ssize_t ng, ssize_t s, ssize_t d, ssize_t p, ssize_t u
  ) {
    return std::max(
      detail::FloorDiv((ng + u - 1) * (p + 1), nf + d - 1) - s + 1,
      static_cast<ssize_t>(0)
    );
  }

  /**
    \brief The min size of h that contains all nonzero indices for a Mellin corr

    A supported index of output h is some index i whose coefficient could be
    nonzero for compactly supported f and g. The minimal size of h that
    contains all supported indices is therefore the maximal supported index i +
    1.

    \param ng[in] [0, ng - 1] is the support of g along the Mellin dimension.
    \param s[in]  the "stride" analogue in the Mellin direction
    \param d[in]  the "dilation" analogue in the Mellin direction
    \param p[in]  the "padding" analogue in the Mellin direction
    \param u[in]  the "upsampling" (dilation of g) analogue in the Mellin direction
    \return nh, the minimum length of h containing all supported indices of h.

    \note The Mellin correlation is computed using sums of

      f[k]g[(i + s)(k + d)/(p + 1) - u] = f[k]g[m]
    
    Setting m <= ng - 1 and k = 0 can solve for nh.
  */
  constexpr ssize_t MCorrSupportSize(
    ssize_t ng, ssize_t s, ssize_t d, ssize_t p, ssize_t u
  ) {
    // (nh - 1 + s)d / (p + 1) - u <= ng - 1
    // nh - 1 + s <= (ng + u - 1) * (p + 1) / d
    // nh <= (ng + u - 1) * (p + 1) / d - s + 1
    return std::max(
      detail::FloorDiv((ng + u - 1) * (p + 1), d) - s + 1,
      static_cast<ssize_t>(0)
    );
  } // nf=2, ng=2, s=1, d=3, p=3, u=1


  /**
    \brief The min size of h that contains all valid indices for a linear corr

    A valid index of output h is some index i whose coefficient is computed
    using only values within the support of g. The minimal size of h that
    contains all valid indices is therefore the maximal valid index i + 1.

    \param nf[in] [0, nf - 1] is the support of f along the linear dimension.
    \param ng[in] [0, ng - 1] is the support of g along the linear dimension.
    \param s[in]  The stride parameter.
    \param d[in]  The dilation parameter.
    \param p[in]  The (left-)padding parameter.
    \param u[in]  The upsampling parameter.
    \return nh, the minimum length of h containing all valid indices of h.

    \note The linear correlation is computed using sums of

      f[k]g[(si + dk - p)/u] = f[k]g[m]
    
    Setting m <= ng - 1 and k = nf - 1 can solve for nh.
  */
  constexpr ssize_t LCorrValidSize(
    ssize_t nf, ssize_t ng, ssize_t s, ssize_t d, ssize_t p, ssize_t u
  ) {
    return std::max(
      detail::FloorDiv(u * (ng - 1) - d * (nf - 1) + p, s) + 1,
      static_cast<ssize_t>(0)
    );
  }

  /**
    \brief The min size of h that contains all nonzero indices for a linear corr

    A supported index of output h is some index i whose coefficient could be
    nonzero for compactly supported f and g. The minimal size of h that
    contains all supported indices is therefore the maximal supported index i +
    1.

    \param ng[in] [0, ng - 1] is the support of g along the linear dimension.
    \param s[in]  The stride parameter.
    \param p[in]  The (left-)padding parameter.
    \param u[in]  The upsampling parameter.
    \return nh, the minimum length of h containing all supported indices of h.

    \note The linear correlation is computed using sums of

      f[k]g[(si + dk - p)/u] = f[k]g[m]
    
    Setting m <= ng - 1 and k = 0 can solve for nh.
  */
  constexpr ssize_t LCorrSupportSize(
    ssize_t ng, ssize_t s, ssize_t p, ssize_t u
  ) {
    return std::max(
      detail::FloorDiv(u * (ng - 1) + p, s) + 1,
      static_cast<ssize_t>(0)
    );
  }

  // END PUBLIC INTERFACE


  namespace detail {

    /*
    Notes on index access bounds for posterity.

    For kernel f of length nf and kernel g of length ng, the supported indices
    are obviously the bounds [0, nf) and [0, ng) resp. For a given
    convolution/correlation, however, the accessed index of g, gx, is a
    function of the accessed index of f, fx. The appropriate range of indices
    involved in the operation can be expressed as a subset of f's supported
    indices [0, nf). We take the intersection of [0, nf) and the range induced
    by transforming [0, ng) into a range w.r.t. fx. The latter always involves
    integer division. If x >= (nd + r) / d for r in [0, d-1] and b is the
    minimum integer satisfying the bound, b = n if r = 0 and b = n + 1
    otherwise. Likewise, x < (nd + r) / d has maximum integer b = n if r = 0
    and b = n + 1 otherwise. This is just ceiling division.

    mconv:
      
      gx = (p + 1) * (s + hx) / (fx + d) - u <= ng - 1
        fx >= (p + 1) * (s + hx) / (ng + u - 1) - d
        fx >= cdiv((p + 1) * (s + hx), ng + u - 1) - d
      
      gx = (p + 1) * (s + hx) / (fx + d) - u >= 0
        fx <= (p + 1) * (s + hx) / u - d
        fx < (p + 1) * (s + hx) / u - d + eps
        fx < cdiv((p + 1) * (s + hx) + 1, u) - d

      alt.

      fx = (p + 1) * (s + hx) / (gx + u) - d <= nf - 1
        gx >= (p + 1) * (s + hx) / (nf + d - 1) - u
        gx >= cdiv((p + 1) * (s + hx), nf + d - 1) - u
      
      fx = (p + 1) * (s + hx) / (gx + u) - d >= 0
        gx <= (p + 1) * (s + hx) / d - u
        gx < (p + 1) * (s + hx) / d - u + eps
        gx < cdiv((p + 1) * (s + hx) + 1, d) - u

    mcorr:

      gx = (s + hx) * (fx + d) / (p + 1) - u < ng
        fx < (ng + u) * (p + 1) / (s + hx) - d
        fx < cdiv((ng + u) * (p + 1), s + hx) - d

      gx = (s + hx) * (fx + d) / (p + 1) - u >= 0
        fx >= u * (p + 1) / (s + hx) - d
        fx >= cdiv(u * (p + 1), s + hx) - d

    lconv:

      gx = (s * hx - d * fx + p) / u <= ng - 1
        fx >= (s * hx - u * (ng - 1) + p) / d
        fx >= cdiv(s * hx - u * (ng - 1) + p, d)
      
      gx = (s * hx - d * fx + p) / u >= 0
        fx <= (s * hx + p) / d
        fx < (s * hx + p) / d + eps
        fx < cdiv(s * hx + p + 1, d)

    lcorr:

      gx = (s * hx + d * fx - p) / u < ng
        fx < (u * ng - s * hx + p) / d
        fx < cdiv(u * ng - s * hx + p, d)

      gx = (s * hx + d * fx - p) / u >= 0
        fx >= (-s * hx + p) / d
        fx >= cdiv(-s * hx + p, d)

    */

    template <typename T>
    using acc_t = typename std::common_type<
      typename std::conditional<std::is_integral<T>::value, int, float>::type,
      T
    >::type;

    // greatest common divisor
    template <class T>
    constexpr T Gcd(T a, T b) {
      return (b == 0) ? a : Gcd(b, a % b);
    }

    // routines from
    // https://stackoverflow.com/questions/8136974/c-functions-for-integer-division-with-well-defined-rounding-strategy/33790603#33790603
    // WARNING: not appropriate for python, which uses flooring in its division
    // rather than truncation

    // ceiling division
    template <typename T1, typename T2>
    constexpr T1 CeilDiv(T1 num, T2 denom) {
      return static_cast<T1>(
        num / denom + (((num < 0) ^ (denom > 0)) && (num % denom)));
    }

    // floor division
    template <typename T1, typename T2>
    constexpr T1 FloorDiv(T1 num, T2 denom) {
      return static_cast<T1>(
        num / denom - (((num > 0) ^ (denom > 0)) && (num % denom)));
    }

    // take max and cast to left type
    template <typename T1, typename T2>
    constexpr T1 LeftMax(T1 a, T2 b) {
      return std::max(a, static_cast<T1>(b));
    }

    // take min and cast to left type
    template <typename T1, typename T2>
    constexpr T1 LeftMin(T1 a, T2 b) {
      return std::min(a, static_cast<T2>(b));
    }

    // for brevity and to make it easier to copy code between devices
    #ifdef I
      #define _CDROBERT_MELLIN_OLD_I I
    #endif
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
    #define I ssize_t
    #define A acc_t<T>
    #define gcd Gcd
    #define cdiv CeilDiv
    #define fdiv FloorDiv
    #define lmax LeftMax
    #define lmin LeftMin

    template <int> struct mconv1d_algo {};
    typedef mconv1d_algo<1> mconv1d_algo_v1;
    typedef mconv1d_algo<2> mconv1d_algo_v2;
    typedef mconv1d_algo<3> mconv1d_algo_v3;

    template<typename T>
    void MConv1D(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mconv1d_algo_v1)
    {
      // naive
      const I Nh = lmin(MConvSupportSize(nf, ng, s, d, p, u), nh);
      ++p;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      #pragma omp parallel for firstprivate(f, g, h)
      for (I i = 0; i < batch * c_out * nh; ++i) {
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

    template<typename T>
    void MConv1D(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mconv1d_algo_v2)
    {
      const I Nh = lmin(MConvSupportSize(nf, ng, s, d, p, u), nh);
      ++p;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      #pragma omp parallel for firstprivate(f, g, h)
      for (I i = 0; i < batch * c_out; ++i) {
        T *h_bt_co = h + i * nh;
        const I co = transposed_h ? (i / batch) : (i % c_out);
        const I bt = transposed_h ? (i % batch) : (i / c_out);
        const T *f_co_x = f + (transposed_f ? co : (co * c_in)) * nf;
        const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
        for (I fx = 0; fx < nf; ++fx) {
          // products are of the form f[fx]g[p(hx + s)/(fx + d) - u]
          // (note we've already added 1 to p)
          // calling p(hx + s)/(fx + d) - u = gx, we take the least
          // common multiple of fx + d and p + 1, such that if p(hx +
          // s) % (fx + d) == 0, then p(hx + lcm_x/p + s) / (fx +
          // d) - u = gx + lcm_x/(fx + d)
          //
          // assuming
          // hx = xs lcm_x/p - s = xs h_x_stride - s
          // gx = xs lcm_x/(fx + d) - u = xs g_x_stride - u
          //
          // gx < ng
          //  xs g_x_stride - u < ng
          //  xs < cdiv(ng + u, g_x_stride)
          // hx < Nh
          //  xs h_x_stride - s < Nh
          //  xs < cdiv(Nh + s, h_x_stride)
          // gx >= 0
          //  xs g_x_stride - u >= 0
          //  xs >= cdiv(u, g_x_stride)
          // hx >= 0
          //  xs h_x_stride - s >= 0
          //  xs >= cdiv(s, h_x_stride)
          const I gcd_x = gcd(p, fx + d);
          const I h_x_stride = (fx + d) / gcd_x;
          const I g_x_stride = p / gcd_x;
          const I min_xs = lmax(cdiv(u, g_x_stride), cdiv(s, h_x_stride));
          const I max_xs = lmin(
            cdiv(ng + u, g_x_stride),
            cdiv(Nh + s, h_x_stride)
          );
          const T *g_bt_x = g_bt + min_xs * g_x_stride - u;
          T *h_bt_co_x = h_bt_co + min_xs * h_x_stride - s;
          for (I xs = min_xs; xs < max_xs; ++xs) {
            A acc = 0;
            for (I ci = 0; ci < c_in; ++ci)
              acc += f_co_x[ci * f_ci_stride] * g_bt_x[ci * g_ci_stride];
            *h_bt_co_x += acc;
            h_bt_co_x += h_x_stride;
            g_bt_x += g_x_stride;
          }

          ++f_co_x;
        }
      }
    }


    template<typename T>
    void MConv1D(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mconv1d_algo_v3)
    {
      const I Nh = lmin(MConvSupportSize(nf, ng, s, d, p, u), nh);
      ++p;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      const I map_size = nf * Nh;

      std::vector<I> idx_map_v(map_size);
      I *idx_map = idx_map_v.data();

      #pragma omp parallel
      {
        // first determine the relevant indices and store them in the
        // index map
      #pragma omp for
        for (I i = 0; i < map_size; ++i) {
          const I fx = i % nf, hx = i / nf;
          const I num = p * (hx + s), denom = fx + d;
          const I gx = num / denom - u, mod = num % denom;
          idx_map[i] = (mod || gx >= ng) ? static_cast<I>(-1) : gx;
        }

        // then use the index map to accumulate the proper values
      #pragma omp for
        for (I i = 0; i < batch * c_out * nh; ++i) {
          const I hx = i % nh, ip = i / nh;
          if (hx >= Nh) continue;
          const I co = transposed_h ? (ip / batch) : (ip % c_out);
          const I bt = transposed_h ? (ip % batch) : (ip / c_out);
          const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
          const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
          I *idx_map_hx = idx_map + hx * nf;
          A acc = 0;
          for (I fx = 0; fx < nf; ++fx) {
            const I gx = idx_map_hx[fx];
            if (gx < 0) continue;
            const T *g_bt_x = g_bt + gx;
            const T *f_co_x = f_co + fx;
            for (I ci = 0; ci < c_in; ++ci)
              acc += f_co_x[ci * f_ci_stride] * g_bt_x[ci * g_ci_stride];
          }
          h[i] += acc;
        }
      }
    }

    template <int> struct mcorr1d_algo {};
    typedef mcorr1d_algo<1> mcorr1d_algo_v1;
    typedef mcorr1d_algo<2> mcorr1d_algo_v2;
    typedef mcorr1d_algo<3> mcorr1d_algo_v3;

    template<typename T>
    void MCorr1D(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mcorr1d_algo_v1)
    {
      const I Nh = lmin(MCorrSupportSize(ng, s, d, p, u), nh);
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      ++p;
      #pragma omp parallel for firstprivate(f, g, h)
      for (I i = 0; i < batch * c_out * nh; ++i) {
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

    template<typename T>
    void MCorr1D(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mcorr1d_algo_v2)
    {
      const I Nh = lmin(MCorrSupportSize(ng, s, d, p, u), nh);
      ++p;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      #pragma omp parallel for firstprivate(f, g, h)
      for (I i = 0; i < batch * c_out * nh; ++i) {
        const I hx = i % nh, ip = i / nh;
        if (hx >= Nh) continue;
        const I co = transposed_h ? (ip / batch) : (ip % c_out);
        const I bt = transposed_h ? (ip % batch) : (ip / c_out);
        const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
        const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
        // products are of the form f[fx]g[(hx + s)(fx + d)/p - u]
        // (note we've already added 1 to p)
        // calling (hx + s)(fx + d)/p - u = gx, we take the least
        // common multiple of hx + s and p + 1, such that if
        // (hx + s)(fx + d) % p == 0, then
        // (hx + s)(fx + lcm_x / (hx + s) + d) / p - u =
        //    gx + lcm_x/p
        //
        // assuming
        // fx = xs lcm_x/(hx + s) - d = xs f_x_stride - d
        // gx = xs lcm_x/p - u = xs g_x_stride - u
        //
        // fx < nf
        //  xs f_x_stride - d < nf
        //  xs < cdiv(nf + d, f_x_stride)
        // gx < ng
        //  xs g_x_stride - u < ng
        //  xs < cdiv(ng + u, g_x_stride)
        // fx >= 0
        //  xs f_x_stride - d >= 0
        //  xs >= cdiv(d, f_x_stride)
        // gx >= 0
        //  xs g_x_stride - u >= 0
        //  xs >= cdiv(u, g_x_stride)
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

    template<typename T>
    void MCorr1D(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nf, I ng, I nh, I s, I d, I p, I u,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mcorr1d_algo_v3)
    {
      const I Nh = lmin(MCorrSupportSize(ng, s, d, p, u), nh);
      ++p;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      const I map_size = nf * Nh;

      std::vector<I> idx_map_v(map_size);
      I *idx_map = idx_map_v.data();

      #pragma omp parallel
      {
        // first determine the relevant indices and store them in the
        // index map
      #pragma omp for
        for (I i = 0; i < map_size; ++i) {
          const I fx = i % nf, hx = i / nf;
          const I num = (fx + d) * (hx + s);
          const I gx = num / p - u, mod = num % p;
          idx_map[i] = (mod || gx >= ng) ? static_cast<I>(-1) : gx;
        }

        // then use the index map to accumulate the proper values
      #pragma omp for
        for (I i = 0; i < batch * c_out * nh; ++i) {
          const I hx = i % nh, ip = i / nh;
          if (hx >= Nh) continue;
          const I co = transposed_h ? (ip / batch) : (ip % c_out);
          const I bt = transposed_h ? (ip % batch) : (ip / c_out);
          const T *f_co = f + (transposed_f ? co : (co * c_in)) * nf;
          const T *g_bt = g + (transposed_g ? bt : (bt * c_in)) * ng;
          I *idx_map_hx = idx_map + hx * nf;
          A acc = 0;
          for (I fx = 0; fx < nf; ++fx) {
            const I gx = idx_map_hx[fx];
            if (gx < 0) continue;
            const T *g_bt_x = g_bt + gx;
            const T *f_co_x = f_co + fx;
            for (I ci = 0; ci < c_in; ++ci)
              acc += f_co_x[ci * f_ci_stride] * g_bt_x[ci * g_ci_stride];
          }
          h[i] += acc;
        }
      }
    }

    template <int> struct mconvlconv_algo { };
    typedef mconvlconv_algo<1> mconvlconv_algo_v1;
    // typedef mconvlconv_algo<2> mconvlconv_algo_v2;

    template <typename T>
    void MConvLConv(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nfx, I nfy, I ngx, I ngy, I nhx, I nhy,
      I sx, I sy, I dx, I dy, I px, I py, I ux, I uy,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mconvlconv_algo_v1)
    {
      const I Nhx = lmin(MConvSupportSize(nfx, ngx, sx, dx, px, ux), nhx);
      ++px;
      const I Nhy = lmin(LConvSupportSize(nfy, ngy, sy, dy, py, uy), nhy);
      const I f_x_stride = nfy, g_x_stride = ngy;
      const I nf = nfx * nfy, ng = ngx * ngy;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      #pragma omp parallel for firstprivate(f, g, h)
      for (I i = 0; i < nhx * nhy * batch * c_out; ++i) {
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

    template <int> struct mcorrlcorr_algo { };
    typedef mcorrlcorr_algo<1> mcorrlcorr_algo_v1;
    // typedef mcorrlcorr_algo<2> mcorrlcorr_algo_v2;

    template <typename T>
    void MCorrLCorr(
      const T *f, const T *g,
      I c_out, I c_in, I batch, I nfx, I nfy, I ngx, I ngy, I nhx, I nhy,
      I sx, I sy, I dx, I dy, I px, I py, I ux, I uy,
      bool transposed_f, bool transposed_g, bool transposed_h,
      T *h, mcorrlcorr_algo_v1)
    {
      const I Nhx = lmin(MCorrSupportSize(ngx, sx, dx, px, ux), nhx);
      ++px;
      const I Nhy = lmin(LCorrSupportSize(ngy, sy, py, uy), nhy);
      const I f_x_stride = nfy, g_x_stride = ngy;
      const I nf = nfx * nfy, ng = ngx * ngy;
      const I f_ci_stride = transposed_f ? c_out * nf : nf;
      const I g_ci_stride = transposed_g ? batch * ng : ng;
      #pragma omp parallel for firstprivate(f, g, h)
      for (I i = 0; i < nhx * nhy * batch * c_out; ++i) {
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

    template <int> struct snd2col_algo { };
    typedef snd2col_algo<1> snd2col_algo_v1;

    template <typename T>
    void Snd2Col(
      const T *g,
      I c_in, I batch, I nf, I ng, I ngg,
      I s, I d, I p, I u,
      bool transposed_g, bool transposed_gg,
      T *gg, snd2col_algo_v1)
    {
      // naive gather
      ++p;
      #pragma omp parallel for firstprivate(g, gg)
      for (I i = 0; i < batch * c_in * nf * ngg; ++i) {
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

    template <int> struct col2snd_algo { };
    typedef col2snd_algo<1> col2snd_algo_v1;

    template <typename T>
    void Col2Snd(
      const T *gg,
      I c_in, I batch, I nf, I ng, I ngg,
      I s, I d, I p, I u,
      bool transposed_g, bool transposed_gg,
      T *g, col2snd_algo_v1)
    {
      // direct (naive) accumulation
      ++p;
      #pragma omp parallel for firstprivate(g, gg)
      for (I i = 0; i < batch * c_in * ng; ++i) {
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

    template <int> struct spec2col_algo { };
    typedef spec2col_algo<1> spec2col_algo_v1;

    template <typename T>
    void Spec2Col(
      const T *g,
      I c_in, I batch,
      I nfx, I nfy, I ngx, I ngy, I nggx, I nggy,
      I sx, I sy, I dx, I dy, I px, I py, I ux, I uy,
      bool transposed_g, bool transposed_gg,
      T *gg, spec2col_algo_v1)
    {
      // naive gather
      ++px;
      #pragma omp parallel for firstprivate(g, gg)
      for (I i = 0; i < batch * c_in * nfx * nfy * nggx * nggy; ++i) {
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

    template <int> struct col2spec_algo { };
    typedef col2spec_algo<1> col2spec_algo_v1;

    template <typename T>
    void Col2Spec(
      const T *gg,
      I c_in, I batch,
      I nfx, I nfy, I ngx, I ngy, I nggx, I nggy,
      I sx, I sy, I dx, I dy, I px, I py, I ux, I uy,
      bool transposed_g, bool transposed_gg,
      T *g, col2spec_algo_v1)
    {
      // direct (naive) accumulation
      ++px;
      #pragma omp parallel for firstprivate(g, gg)
      for (I i = 0; i < batch * c_in * ngx * ngy; ++i) {
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

    #undef I
    #undef A
    #undef gcd
    #undef cdiv
    #undef fdiv
    #undef lmax
    #undef lmin
    #ifdef _CDROBERT_MELLIN_OLD_I
      #define I _CDROBERT_MELLIN_OLD_I
      #undef _CDROBERT_MELLIN_OLD_I
    #endif
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

  template <typename T>
  void MConv1D(
    const T *f, const T *g,
    ssize_t c_out, ssize_t c_in, ssize_t batch,
    ssize_t nf, ssize_t ng, ssize_t nh,
    ssize_t s, ssize_t d, ssize_t p, ssize_t u,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h
  ) {
    static_assert(std::is_arithmetic<T>::value, "f, g, and h type must be arithmetic");
    detail::MConv1D<T>(
      f, g, c_out, c_in, batch, nf, ng, nh, s, d, p, u,
      transposed_f, transposed_g, transposed_h, h,
      detail::mconv1d_algo<kMConv1DAlgorithmVersion>()
    );
  }

  template <typename T>
  void MCorr1D(
    const T *f, const T *g,
    ssize_t c_out, ssize_t c_in, ssize_t batch,
    ssize_t nf, ssize_t ng, ssize_t nh,
    ssize_t s, ssize_t d, ssize_t p, ssize_t u,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h
  ) {
    static_assert(std::is_arithmetic<T>::value, "f, g, and h type must be arithmetic");
    detail::MCorr1D<T>(
      f, g, c_out, c_in, batch, nf, ng, nh, s, d, p, u,
      transposed_f, transposed_g, transposed_h, h,
      detail::mcorr1d_algo<kMCorr1DAlgorithmVersion>()
    );
  }

  template <typename T>
  void MConvLConv(
    const T *f, const T *g,
    ssize_t c_out, ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nhx, ssize_t nhy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h
  ) {
    static_assert(std::is_arithmetic<T>::value, "f, g, and h type must be arithmetic");
    detail::MConvLConv<T>(
      f, g, c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx,
      dy, px, py, ux, uy, transposed_f, transposed_g, transposed_h, h,
      detail::mconvlconv_algo<kMConvLConvAlgorithmVersion>()
    );
  }

  template <typename T>
  void MCorrLCorr(
    const T *f, const T *g,
    ssize_t c_out, ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nhx, ssize_t nhy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_f, bool transposed_g, bool transposed_h,
    T *h)
  {
    static_assert(std::is_arithmetic<T>::value, "f, g, and h type must be arithmetic");
    detail::MCorrLCorr<T>(
      f, g, c_out, c_in, batch, nfx, nfy, ngx, ngy, nhx, nhy, sx, sy, dx,
      dy, px, py, ux, uy, transposed_f, transposed_g, transposed_h, h,
      detail::mcorrlcorr_algo<kMCorrLCorrAlgorithmVersion>()
    );
  }

  template <typename T>
  void Snd2Col(
    const T *g,
    ssize_t c_in, ssize_t batch,
    ssize_t nf, ssize_t ng, ssize_t ngg,
    ssize_t s, ssize_t d, ssize_t p, ssize_t u,
    bool transposed_g, bool transposed_gg,
    T *gg)
  {
    static_assert(std::is_arithmetic<T>::value, "g and gg type must be arithmetic");
    detail::Snd2Col<T>(
      g, c_in, batch, nf, ng, ngg, s, d, p, u,
      transposed_g, transposed_gg, gg,
      detail::snd2col_algo<kSnd2ColAlgorithmVersion>()
    );
  }

  template <typename T>
  void Col2Snd(
    const T *gg,
    ssize_t c_in, ssize_t batch,
    ssize_t nf, ssize_t ng, ssize_t ngg,
    ssize_t s, ssize_t d, ssize_t p, ssize_t u,
    bool transposed_g, bool transposed_gg,
    T *g)
  {
    static_assert(std::is_arithmetic<T>::value, "g and gg type must be arithmetic");
    detail::Col2Snd<T>(
      gg, c_in, batch, nf, ng, ngg, s, d, p, u,
      transposed_g, transposed_gg, g,
      detail::col2snd_algo<kCol2SndAlgorithmVersion>()
    );
  }

  template <typename T>
  void Spec2Col(
    const T *g,
    ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nggx, ssize_t nggy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_g, bool transposed_gg,
    T *gg
  ) {
    static_assert(std::is_arithmetic<T>::value, "g and gg type must be arithmetic");
    detail::Spec2Col<T>(
      g, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy,
      sx, sy, dx, dy, px, py, ux, uy,
      transposed_g, transposed_gg, gg,
      detail::spec2col_algo<kSpec2ColAlgorithmVersion>()
    );
  }

  template <typename T>
  void Col2Spec(
    const T *gg,
    ssize_t c_in, ssize_t batch,
    ssize_t nfx, ssize_t nfy, ssize_t ngx, ssize_t ngy, ssize_t nggx, ssize_t nggy,
    ssize_t sx, ssize_t sy, ssize_t dx, ssize_t dy, ssize_t px, ssize_t py,
    ssize_t ux, ssize_t uy,
    bool transposed_g, bool transposed_gg,
    T *g
  ) {
    static_assert(std::is_arithmetic<T>::value, "g and gg type must be arithmetic");
    detail::Col2Spec<T>(
      gg, c_in, batch, nfx, nfy, ngx, ngy, nggx, nggy,
      sx, sy, dx, dy, px, py, ux, uy,
      transposed_g, transposed_gg, g,
      detail::col2spec_algo<kCol2SpecAlgorithmVersion>()
    );
  }

}}  // namespace cdrobert::mellin

#endif  // CDROBERT_MELLIN_MCONV_H_
