// Copyright 2018 Sean Robertson
#include <torch/extension.h>
#include <c10/util/Optional.h>

#include "cdrobert/mellin/mconv.h"

// defined in torch.cpp
std::pair<torch::Tensor, torch::Tensor> CheckAndInferArgumentsConv(
  torch::CheckedFrom c, int64_t dim,
  const torch::Tensor& f, const torch::Tensor& g,
  const c10::optional<torch::Tensor>& bias,
  int64_t& c_out, int64_t& c_in, int64_t& batch,
  int64_t& nfx, int64_t& ngx,
  bool& transposed_f, bool& transposed_g,
  int64_t* nfy = nullptr, int64_t* ngy = nullptr
);
torch::Tensor CheckAndInferArgumentsCol(
  torch::CheckedFrom c, int64_t dim,
  const torch::Tensor& g,
  int64_t& c_in, int64_t& batch, int64_t& ng1,
  bool& transposed_g,
  int64_t* ng2 = nullptr, int64_t* ng3 = nullptr, int64_t* ng4 = nullptr
);
torch::Tensor BuildOutput(
  const torch::Tensor& f, const c10::optional<torch::Tensor>& bias,
  at::IntArrayRef size
);

torch::Tensor mconv1d_cpu(
  const torch::Tensor& input, const torch::Tensor& weight,
  const c10::optional<torch::Tensor>& bias,
  int64_t s, int64_t d, int64_t p, int64_t u, int64_t r
) {
  int64_t c_out, c_in, batch, nf, ng;
  bool transposed_f = false, transposed_g = false;
  torch::Tensor f, g;
  std::tie(f, g) = CheckAndInferArgumentsConv(
    __func__, 3,
    weight, input, bias,
    c_out, c_in, batch, nf, ng,
    transposed_f, transposed_g
  );
  int64_t nh = cdrobert::mellin::MConvValidSize(ng, s, d, p, u) + r;
  torch::Tensor h = BuildOutput(f, bias, {batch, c_out, nh});
  AT_DISPATCH_ALL_TYPES(f.scalar_type(), __func__, ([&]{
    cdrobert::mellin::MConv1D(
      f.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
      c_out, c_in, batch,
      nf, ng, nh,
      s, d, p, u,
      transposed_f, transposed_g, false,
      h.data_ptr<scalar_t>());
  }));
  return h;
}

torch::Tensor mcorr1d_cpu(
  const torch::Tensor& input, const torch::Tensor& weight,
  const c10::optional<torch::Tensor>& bias,
  int64_t s, int64_t d, int64_t p, int64_t u, int64_t r
) {
  int64_t c_out, c_in, batch, nf, ng;
  bool transposed_f = false, transposed_g = false;
  torch::Tensor f, g;
  std::tie(f, g) = CheckAndInferArgumentsConv(
    __func__, 3,
    weight, input, bias,
    c_out, c_in, batch, nf, ng,
    transposed_f, transposed_g
  );
  int64_t nh = cdrobert::mellin::MCorrValidSize(nf, ng, s, d, p, u) + r;
  torch::Tensor h = BuildOutput(f, bias, {batch, c_out, nh});
  AT_DISPATCH_ALL_TYPES(f.scalar_type(), __func__, ([&]{
    cdrobert::mellin::MCorr1D(
      f.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
      c_out, c_in, batch,
      nf, ng, nh,
      s, d, p, u,
      transposed_f, transposed_g, false,
      h.data_ptr<scalar_t>());
  }));
  return h;
}

torch::Tensor mconvlconv_cpu(
  const torch::Tensor& input, const torch::Tensor& weight,
  const c10::optional<torch::Tensor>& bias,
  at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p,
  at::IntArrayRef u, at::IntArrayRef r
) {
  int64_t c_out, c_in, batch, nfx, nfy, ngx, ngy;
  bool transposed_f = false, transposed_g = false;
  torch::Tensor f, g;
  std::tie(f, g) = CheckAndInferArgumentsConv(
    __func__, 4,
    weight, input, bias,
    c_out, c_in, batch, nfx, ngx,
    transposed_f, transposed_g,
    &nfy, &ngy
  );
  int64_t nhx = cdrobert::mellin::MConvValidSize(ngx, s[0], d[0], p[0], u[0]) + r[0];
  int64_t nhy = cdrobert::mellin::LConvValidSize(ngy, s[1], p[1], u[1]) + r[1];
  torch::Tensor h = BuildOutput(f, bias, {batch, c_out, nhx, nhy});
  AT_DISPATCH_ALL_TYPES(f.scalar_type(), __func__, ([&]{
    cdrobert::mellin::MConvLConv(
      f.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
      c_out, c_in, batch,
      nfx, nfy, ngx, ngy, nhx, nhy,
      s[0], s[1], d[0], d[1], p[0], p[1], u[0], u[1],
      transposed_f, transposed_g, false,
      h.data_ptr<scalar_t>());
  }));
  return h;
}

torch::Tensor mcorrlcorr_cpu(
  const torch::Tensor& input, const torch::Tensor& weight,
  const c10::optional<torch::Tensor>& bias,
  at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p,
  at::IntArrayRef u, at::IntArrayRef r
) {
  int64_t c_out, c_in, batch, nfx, nfy, ngx, ngy;
  bool transposed_f = false, transposed_g = false;
  torch::Tensor f, g;
  std::tie(f, g) = CheckAndInferArgumentsConv(
    __func__, 4,
    weight, input, bias,
    c_out, c_in, batch, nfx, ngx,
    transposed_f, transposed_g,
    &nfy, &ngy
  );
  int64_t nhx = cdrobert::mellin::MCorrValidSize(nfx, ngx, s[0], d[0], p[0], u[0]) + r[0];
  int64_t nhy = cdrobert::mellin::LCorrValidSize(nfy, ngy, s[1], d[1], p[1], u[1]) + r[1];
  torch::Tensor h = BuildOutput(f, bias, {batch, c_out, nhx, nhy});
  AT_DISPATCH_ALL_TYPES(f.scalar_type(), __func__, ([&]{
    cdrobert::mellin::MCorrLCorr(
      f.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
      c_out, c_in, batch,
      nfx, nfy, ngx, ngy, nhx, nhy,
      s[0], s[1], d[0], d[1], p[0], p[1], u[0], u[1],
      transposed_f, transposed_g, false,
      h.data_ptr<scalar_t>());
  }));
  return h;
}

torch::Tensor snd2col_cpu(
  const torch::Tensor& input, int64_t nf,
  int64_t s, int64_t d, int64_t p, int64_t u, int64_t r)
{
  int64_t c_in, batch, ng;
  bool transposed_g = false;
  torch::Tensor g = CheckAndInferArgumentsCol(
    __func__, 3,
    input,
    c_in, batch, ng,
    transposed_g
  );
  int64_t ngg = cdrobert::mellin::MCorrValidSize(nf, ng, s, d, p, u) + r;
  torch::Tensor gg = g.new_empty({batch, c_in, nf, ngg});
  AT_DISPATCH_ALL_TYPES(g.scalar_type(), __func__, ([&]{
    cdrobert::mellin::Snd2Col(
      g.data_ptr<scalar_t>(),
      c_in, batch, nf, ng, ngg,
      s, d, p, u,
      transposed_g, false,
      gg.data_ptr<scalar_t>());
  }));
  return gg;
}

torch::Tensor col2snd_cpu(
  const torch::Tensor& input, ssize_t ng,
  int64_t s, int64_t d, int64_t p, int64_t u)
{
  int64_t c_in, batch, nf, ngg;
  bool transposed_gg = false;
  torch::Tensor gg = CheckAndInferArgumentsCol(
    __func__, 4,
    input,
    c_in, batch, nf,
    transposed_gg, &ngg
  );
  torch::Tensor g = gg.new_empty({batch, c_in, ng});
  AT_DISPATCH_ALL_TYPES(gg.scalar_type(), __func__, ([&]{
    cdrobert::mellin::Col2Snd(
      gg.data_ptr<scalar_t>(),
      c_in, batch, nf, ng, ngg,
      s, d, p, u,
      false, transposed_gg,
      g.data_ptr<scalar_t>());
  }));
  return g;
}

torch::Tensor spec2col_cpu(
  const torch::Tensor& input, at::IntArrayRef nf,
  at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p, at::IntArrayRef u,
  at::IntArrayRef r)
{
  int64_t c_in, batch, ngx, ngy;
  bool transposed_g = false;
  torch::Tensor g = CheckAndInferArgumentsCol(
    __func__, 4,
    input,
    c_in, batch, ngx,
    transposed_g, &ngy
  );
  int64_t nggx = cdrobert::mellin::MCorrValidSize(nf[0], ngx, s[0], d[0], p[0], u[0]) + r[0];
  int64_t nggy = cdrobert::mellin::LCorrValidSize(nf[1], ngy, s[1], d[1], p[1], u[1]) + r[1];
  torch::Tensor gg = g.new_empty({batch, c_in, nf[0], nf[1], nggx, nggy});
  AT_DISPATCH_ALL_TYPES(g.scalar_type(), __func__, ([&]{
    cdrobert::mellin::Spec2Col(
      g.data_ptr<scalar_t>(),
      c_in, batch, nf[0], nf[1], ngx, ngy, nggx, nggy,
      s[0], s[1], d[0], d[1], p[0], p[1], u[0], u[1],
      transposed_g, false,
      gg.data_ptr<scalar_t>());
  }));
  return gg;
}

torch::Tensor col2spec_cpu(
  const torch::Tensor& input, at::IntArrayRef ng,
  at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p, at::IntArrayRef u)
{
  int64_t c_in, batch, nfx, nfy, nggx, nggy;
  bool transposed_gg = false;
  torch::Tensor gg = CheckAndInferArgumentsCol(
    __func__, 6,
    input,
    c_in, batch, nfx,
    transposed_gg, &nfy, &nggx, &nggy
  );
  torch::Tensor g = gg.new_empty({batch, c_in, ng[0], ng[1]});
  AT_DISPATCH_ALL_TYPES(g.scalar_type(), __func__, ([&]{
    cdrobert::mellin::Col2Spec(
      gg.data_ptr<scalar_t>(),
      c_in, batch, nfx, nfy, ng[0], ng[1], nggx, nggy,
      s[0], s[1], d[0], d[1], p[0], p[1], u[0], u[1],
      false, transposed_gg,
      g.data_ptr<scalar_t>());
  }));
  return g;
}

TORCH_LIBRARY_IMPL(mellin, CPU, m) {
  m.impl("mconv1d", mconv1d_cpu);
  m.impl("mcorr1d", mcorr1d_cpu);
  m.impl("mconvlconv", mconvlconv_cpu);
  m.impl("mcorrlcorr", mcorrlcorr_cpu);
  m.impl("snd2col", snd2col_cpu);
  m.impl("col2snd", col2snd_cpu);
  m.impl("spec2col", spec2col_cpu);
  m.impl("col2spec", col2spec_cpu);
}
