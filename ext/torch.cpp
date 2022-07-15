#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <ATen/autocast_mode.h>
#include <vector>
#include <iostream>

#include "cdrobert/mellin/mconv.h"

// boilerplate for getting C++ interface arguments from tensors (convolutions)
std::pair<torch::Tensor, torch::Tensor> CheckAndInferArgumentsConv(
  torch::CheckedFrom c, int64_t dim,
  const torch::Tensor& f, const torch::Tensor& g,
  const c10::optional<torch::Tensor>& bias,
  int64_t& c_out, int64_t& c_in, int64_t& batch,
  int64_t& nfx, int64_t& ngx,
  bool& transposed_f, bool& transposed_g,
  int64_t* nfy = nullptr, int64_t* ngy = nullptr
) {
  // g = input = first arg, f = weight = second arg
  torch::TensorArg f_arg(f, "weight", 2), g_arg(g, "input", 1);
  // torch::checkAllSameType(c, {f_arg, g_arg}); // already true from dispatch?
  torch::checkDim(c, f_arg, dim);
  torch::checkDim(c, g_arg, dim);
  auto f_sizes = f.sizes(), g_sizes = g.sizes();
  c_out = f_sizes[transposed_f ? 1 : 0];
  c_in = f_sizes[transposed_f ? 0 : 1];
  batch = g_sizes[transposed_g ? 1 : 0];
  nfx = f_sizes[2];
  ngx = g_sizes[2];
  if (dim == 4) {
    *nfy = f_sizes[3];
    *ngy = g_sizes[3];
  }
  torch::checkSize(c, g_arg, transposed_g ? 0 : 1, c_in);
  if (bias) {
    torch::TensorArg bias_arg(*bias, "bias", 3);
    torch::checkDim(c, bias_arg, 1);
    torch::checkSize(c, bias_arg, 0, c_out);
  }
  torch::Tensor fc, gc;
  if (f.transpose(0, 1).is_contiguous()) {
    transposed_f = !transposed_f;
    fc = f.transpose(0, 1);
  } else {
    fc = f.contiguous();
  }
  if (g.transpose(0, 1).is_contiguous()) {
    transposed_g = !transposed_g;
    gc = g.transpose(0, 1);
  } else {
    gc = g.contiguous();
  }
  return std::make_pair(fc, gc);
}

// boilerplate for getting C++ interface arguments from tensors (column ops)
torch::Tensor CheckAndInferArgumentsCol(
  torch::CheckedFrom c, int64_t dim,
  const torch::Tensor& g,
  int64_t& c_in, int64_t& batch, int64_t& ng1,
  bool& transposed_g,
  int64_t* ng2 = nullptr, int64_t* ng3 = nullptr, int64_t* ng4 = nullptr
) {
  torch::TensorArg g_arg(g, "input", 1);
  torch::checkDim(c, g_arg, dim);
  auto g_sizes = g.sizes();
  c_in = g_sizes[transposed_g ? 0 : 1];
  batch = g_sizes[transposed_g ? 1 : 0];
  ng1 = g_sizes[2];
  torch::Tensor gc;
  if (dim >= 4) {
    if (ng2 != nullptr) *ng2 = g_sizes[3];
    if (dim >= 5) {
      if (ng3 != nullptr) *ng3 = g_sizes[4];
      if (dim >= 6 && ng4 != nullptr) *ng4 = g_sizes[5];
    }
  }
  if (g.transpose(0, 1).is_contiguous()) {
    transposed_g = !transposed_g;
    gc = g.transpose(0, 1);
  } else {
    gc = g.contiguous();
  }
  return gc;
}

// boilerplate for populating the output tensor, possibly using the bias
torch::Tensor BuildOutput(
  const torch::Tensor& f, const c10::optional<torch::Tensor>& bias,
  at::IntArrayRef size
) {
  torch::Tensor out = f.new_empty(size);
  if (bias) {
    std::vector<int64_t> view_size {1, size[1]};
    for (uint i = 0; i < size.size() - 2; ++i) view_size.push_back(1);
    out.copy_(bias->view(view_size).expand(size));
  } else {
    out.zero_();
  }
  return out;
}


#define _1D_CONV_SIG \
  ( \
    const torch::Tensor& input, const torch::Tensor& weight, \
    const c10::optional<torch::Tensor>& bias, \
    int64_t s, int64_t d, int64_t p, int64_t u, int64_t r \
  )
#define _1D_CONV_ARGS input, weight, bias, s, d, p, u, r

#define _ND_CONV_SIG \
  ( \
    const torch::Tensor& input, const torch::Tensor& weight, \
    const c10::optional<torch::Tensor>& bias, \
    at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p, \
    at::IntArrayRef u, at::IntArrayRef r \
  )
#define _ND_CONV_ARGS input, weight, bias, s, d, p, u, r

#define _SND2COL_SIG \
  ( \
    const torch::Tensor& input, int64_t kernel_width, \
    int64_t s, int64_t d, int64_t p, int64_t u, int64_t r \
  )
#define _SND2COL_ARGS input, kernel_width, s, d, p, u, r

#define _COL2SND_SIG \
  ( \
    const torch::Tensor& input, int64_t snd_width, \
    int64_t s, int64_t d, int64_t p, int64_t u \
  )
#define _COL2SND_ARGS input, snd_width, s, d, p, u

#define _SPEC2COL_SIG \
  ( \
    const torch::Tensor& input, at::IntArrayRef kernel_size, \
    at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p, \
    at::IntArrayRef u, at::IntArrayRef r \
  )
#define _SPEC2COL_ARGS input, kernel_size, s, d, p, u, r

#define _COL2SPEC_SIG \
  ( \
    const torch::Tensor& input, at::IntArrayRef spec_size, \
    at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p, \
    at::IntArrayRef u \
  )
#define _COL2SPEC_ARGS input, spec_size, s, d, p, u


// functions for calling function through dispatch
// mostly boilerplate
// https://pytorch.org/tutorials/advanced/dispatcher.html
#define _DISPATCH_FUNC(name, sig, args) \
torch::Tensor name sig \
{ \
  static auto op = torch::Dispatcher::singleton() \
    .findSchemaOrThrow("mellin::" #name, "") \
    .typed<decltype(name)>(); \
  return op.call(args); \
}

_DISPATCH_FUNC(mcorr1d, _1D_CONV_SIG, _1D_CONV_ARGS);
_DISPATCH_FUNC(mconv1d, _1D_CONV_SIG, _1D_CONV_ARGS);
_DISPATCH_FUNC(mcorrlcorr, _ND_CONV_SIG, _ND_CONV_ARGS);
_DISPATCH_FUNC(mconvlconv, _ND_CONV_SIG, _ND_CONV_ARGS);
_DISPATCH_FUNC(snd2col, _SND2COL_SIG, _SND2COL_ARGS);
_DISPATCH_FUNC(col2snd, _COL2SND_SIG, _COL2SND_ARGS);
_DISPATCH_FUNC(spec2col, _SPEC2COL_SIG, _SPEC2COL_ARGS);
_DISPATCH_FUNC(col2spec, _COL2SPEC_SIG, _COL2SPEC_ARGS);

#undef _DISPATCH_FUNC

// autograd
class MCorr1DOp : public torch::autograd::Function<MCorr1DOp> {
  public:
    static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& input, const torch::Tensor& weight,
      const c10::optional<torch::Tensor>& bias,
      int64_t s, int64_t d, int64_t p, int64_t u, int64_t r
    ) {
      ctx->save_for_backward({input, weight, (bias ? (*bias) : torch::Tensor())});
      ctx->saved_data["s"] = s;
      ctx->saved_data["d"] = d;
      ctx->saved_data["p"] = p;
      ctx->saved_data["u"] = u;
      at::AutoDispatchBelowADInplaceOrView guard;
      return mcorr1d(input, weight, bias, s, d, p, u, r);
    }

    static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs
    ) {
      auto saved = ctx->get_saved_variables();
      torch::Tensor grad_out = grad_outputs[0],
                    input = saved[0],
                    weight = saved[1],
                    bias = saved[2];
      int64_t s = ctx->saved_data["s"].toInt(),
              d = ctx->saved_data["d"].toInt(),
              p = ctx->saved_data["p"].toInt(),
              u = ctx->saved_data["u"].toInt(),
              nf = weight.size(2),
              ng = input.size(2),
              nh = grad_out.size(2);
      int64_t r;
      
      // grad_input is mconv1d(weight, grad_out, u, s, p, d, *)
      // weight.transpose(0, 1) of (c_in, c_out, nf) ~ input' of (batch', c_in', ng')
      // grad_out of (batch, c_out, nh) ~ weight' of (c_out', c_in', nf')
      // grad_input of (batch, c_in, ng) ~ out'.transpose(0, 1) of (c_out', batch', nh')
      r = ng - cdrobert::mellin::MConvValidSize(
        nf /* ng' */, u /* s' */, s /* d' */, p /* p' */, d /* u' */);
      torch::Tensor grad_input = mconv1d(
        weight.transpose(0, 1), grad_out, {}, u, s, p, d, r
      ).transpose(0, 1);

      // grad_weight is mcorr1d(input, grad_out, d, s, p, u, *)
      // input.transpose(0, 1) of (c_in, batch, ng) ~ input' of (batch', c_in', ng')
      // grad_out.transpose(0, 1) of (c_out, batch, nh) ~ weight' of (c_out', c_in', nf')
      // grad_weight of (c_out, c_in, nf) ~ out'.transpose(0, 1) of (c_out', batch', nh')
      r = nf - cdrobert::mellin::MCorrValidSize(
        nh /* nf' */, ng /* ng' */, d /* s' */, s /* d' */, p /* p' */, u /* u' */);
      torch::Tensor grad_weight = mcorr1d(
        input.transpose(0, 1), grad_out.transpose(0, 1), {}, d, s, p, u, r
      ).transpose(0, 1);

      torch::Tensor grad_bias = bias.defined() ? grad_out.sum(0).sum(1) : torch::Tensor();

      return {
        grad_input, grad_weight, grad_bias,
        torch::Tensor() /* s */, torch::Tensor() /* d */,
        torch::Tensor() /* p */, torch::Tensor() /* u */,
        torch::Tensor() /* r */
      };
    }
};


class MCorrLCorrOp : public torch::autograd::Function<MCorrLCorrOp> {
  public:
    static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& input, const torch::Tensor& weight,
      const c10::optional<torch::Tensor>& bias,
      at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p,
      at::IntArrayRef u, at::IntArrayRef r
    ) {
      ctx->save_for_backward({input, weight, (bias ? (*bias) : torch::Tensor())});
      ctx->saved_data["s"] = s;
      ctx->saved_data["d"] = d;
      ctx->saved_data["p"] = p;
      ctx->saved_data["u"] = u;
      at::AutoDispatchBelowADInplaceOrView guard;
      return mcorrlcorr(input, weight, bias, s, d, p, u, r);
    }

    static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs
    ) {
      auto saved = ctx->get_saved_variables();
      torch::Tensor grad_out = grad_outputs[0],
                    input = saved[0],
                    weight = saved[1],
                    bias = saved[2];
      auto s = ctx->saved_data["s"].toIntVector(),
           d = ctx->saved_data["d"].toIntVector(),
           p = ctx->saved_data["p"].toIntVector(),
           u = ctx->saved_data["u"].toIntVector();
      int64_t nfx = weight.size(2),
              nfy = weight.size(3),
              ngx = input.size(2),
              ngy = input.size(3),
              nhx = grad_out.size(2),
              nhy = grad_out.size(3);
      int64_t rx, ry;
      
      // same logic as mcorr1d, being careful with linear dim

      rx = ngx - cdrobert::mellin::MConvValidSize(
        nfx /* ng' */, u[0] /* s' */, s[0] /* d' */, p[0] /* p' */, d[0] /* u' */);
      ry = ngy - cdrobert::mellin::LConvValidSize(
        nfy /* ng' */, u[1] /* s' */, p[1] /* p' */, d[1] /* u' */);
      torch::Tensor grad_input = mconvlconv(
        weight.transpose(0, 1), grad_out, {}, u, s, p, d, {rx, ry}
      ).transpose(0, 1);

      rx = nfx - cdrobert::mellin::MCorrValidSize(
        nhx /* nf' */, ngx /* ng' */, d[0] /* s' */, s[0] /* d' */, p[0] /* p' */, u[0] /* u' */);
      ry = nfy - cdrobert::mellin::LCorrValidSize(
        nhy /* nf' */, ngy /* ng' */, d[1] /* s' */, s[1] /* d' */, p[1] /* p' */, u[1] /* u' */);
      torch::Tensor grad_weight = mcorrlcorr(
        input.transpose(0, 1), grad_out.transpose(0, 1), {}, d, s, p, u, {rx, ry}
      ).transpose(0, 1);

      torch::Tensor grad_bias = bias.defined() ? grad_out.sum(0).flatten(1).sum(1) : torch::Tensor();

      return {
        grad_input, grad_weight, grad_bias,
        torch::Tensor() /* s */, torch::Tensor() /* d */,
        torch::Tensor() /* p */, torch::Tensor() /* u */,
        torch::Tensor() /* r */
      };
    }
};

class Snd2ColOp : public torch::autograd::Function<Snd2ColOp> {
  public:
    static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& input, int64_t kernel_width,
      int64_t s, int64_t d, int64_t p, int64_t u, int64_t r
    ) {
      ctx->saved_data["s"] = s;
      ctx->saved_data["d"] = d;
      ctx->saved_data["p"] = p;
      ctx->saved_data["u"] = u;
      ctx->saved_data["snd_width"] = input.size(2);
      at::AutoDispatchBelowADInplaceOrView guard;
      return snd2col(input, kernel_width, s, d, p, u, r);
    }

    static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs
    ) {
      auto saved = ctx->get_saved_variables();
      torch::Tensor grad_out = grad_outputs[0];
      int64_t s = ctx->saved_data["s"].toInt(),
              d = ctx->saved_data["d"].toInt(),
              p = ctx->saved_data["p"].toInt(),
              u = ctx->saved_data["u"].toInt(),
              snd_width = ctx->saved_data["snd_width"].toInt();
      
      torch::Tensor grad_input = col2snd(grad_out, snd_width, s, d, p, u);

      return {
        grad_input, torch::Tensor() /* kernel_width */,
        torch::Tensor() /* s */, torch::Tensor() /* d */,
        torch::Tensor() /* p */, torch::Tensor() /* u */,
        torch::Tensor() /* r */
      };
    }
};

class Spec2ColOp : public torch::autograd::Function<Spec2ColOp> {
  public:
    static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& input, at::IntArrayRef kernel_width,
      at::IntArrayRef s, at::IntArrayRef d, at::IntArrayRef p,
      at::IntArrayRef u, at::IntArrayRef r
    ) {
      ctx->saved_data["s"] = s;
      ctx->saved_data["d"] = d;
      ctx->saved_data["p"] = p;
      ctx->saved_data["u"] = u;
      ctx->saved_data["spec_size"] = at::IntArrayRef({input.size(2), input.size(3)});
      at::AutoDispatchBelowADInplaceOrView guard;
      return spec2col(input, kernel_width, s, d, p, u, r);
    }

    static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs
    ) {
      auto saved = ctx->get_saved_variables();
      torch::Tensor grad_out = grad_outputs[0];
      auto s = ctx->saved_data["s"].toIntVector(),
              d = ctx->saved_data["d"].toIntVector(),
              p = ctx->saved_data["p"].toIntVector(),
              u = ctx->saved_data["u"].toIntVector(),
              spec_size = ctx->saved_data["spec_size"].toIntVector();
      
      torch::Tensor grad_input = col2spec(grad_out, spec_size, s, d, p, u);

      return {
        grad_input, torch::Tensor() /* kernel_size */,
        torch::Tensor() /* s */, torch::Tensor() /* d */,
        torch::Tensor() /* p */, torch::Tensor() /* u */,
        torch::Tensor() /* r */
      };
    }
};


#define _AUTOGRAD_FUNC(name, op_name, sig, args) \
  torch::Tensor name ## _autograd sig \
  { return op_name::apply(args); } \
  TORCH_LIBRARY_IMPL(mellin, Autograd, m) { \
    m.impl(#name, name ## _autograd); \
  }

_AUTOGRAD_FUNC(mcorr1d, MCorr1DOp, _1D_CONV_SIG, _1D_CONV_ARGS);
_AUTOGRAD_FUNC(mcorrlcorr, MCorrLCorrOp, _ND_CONV_SIG, _ND_CONV_ARGS);
_AUTOGRAD_FUNC(snd2col, Snd2ColOp, _SND2COL_SIG, _SND2COL_ARGS);
_AUTOGRAD_FUNC(spec2col, Spec2ColOp, _SPEC2COL_SIG, _SPEC2COL_ARGS);

#undef _AUTOGRAD_FUNC

// follow the torch standard of casting convolution args to half
// https://pytorch.org/docs/stable/amp.html#id6
#define _AUTOCAST_FUNC(name, sig) \
torch::Tensor name ## _autocast sig \
{ \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast); \
  if (bias) { \
    return name( \
      at::autocast::cached_cast(at::kHalf, input), \
      at::autocast::cached_cast(at::kHalf, weight), \
      at::autocast::cached_cast(at::kHalf, *bias), \
      s, d, p, u, r \
    ); \
  } else { \
    return name( \
      at::autocast::cached_cast(at::kHalf, input), \
      at::autocast::cached_cast(at::kHalf, weight), \
      bias, \
      s, d, p, u, r \
    ); \
  } \
} \
TORCH_LIBRARY_IMPL(mellin, Autocast, m) { \
  m.impl(#name, name ## _autocast); \
}

_AUTOCAST_FUNC(mcorr1d, _1D_CONV_SIG);
_AUTOCAST_FUNC(mconv1d, _1D_CONV_SIG);
_AUTOCAST_FUNC(mcorrlcorr, _ND_CONV_SIG);
_AUTOCAST_FUNC(mconvlconv, _ND_CONV_SIG);

#undef _AUTOCAST_FUNC
#define _AUTOCAST_FUNC(name, sig, ...) \
torch::Tensor name ## _autocast sig \
{ \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast); \
  return name(at::autocast::cached_cast(at::kHalf, input), __VA_ARGS__); \
} \
TORCH_LIBRARY_IMPL(mellin, Autocast, m) { \
  m.impl(#name, name ## _autocast); \
}

_AUTOCAST_FUNC(snd2col, _SND2COL_SIG, kernel_width, s, d, p, u, r);
_AUTOCAST_FUNC(col2snd, _COL2SND_SIG, snd_width, s, d, p, u);
_AUTOCAST_FUNC(spec2col, _SPEC2COL_SIG, kernel_size, s, d, p, u, r);
_AUTOCAST_FUNC(col2spec, _COL2SPEC_SIG, spec_size, s, d, p, u);

#undef _AUTOCAST_FUNC

#undef _1D_CONV_SIG
#undef _1D_CONV_ARGS
#undef _ND_CONV_SIG
#undef _ND_CONV_ARGS
#undef _SND2COL_SIG
#undef _SND2COL_ARGS
#undef _COL2SND_SIG
#undef _COL2SND_ARGS
#undef _SPEC2COL_SIG
#undef _SPEC2COL_ARGS
#undef _COL2SPEC_SIG
#undef _COL2SPEC_ARGS

TORCH_LIBRARY(mellin, m) {
  m.def("mconv1d(Tensor input, Tensor weight, Tensor? bias = None, int s = 1, int d = 1, int p = 0, int u = 1, int r = 0) -> Tensor");
  m.def("mcorr1d(Tensor input, Tensor weight, Tensor? bias = None, int s = 1, int d = 1, int p = 0, int u = 1, int r = 0) -> Tensor");
  m.def("mcorrlcorr(Tensor input, Tensor weight, Tensor? bias = None, int[2] s = 1, int[2] d = 1, int[2] p = 0, int[2] u = 1, int[2] r = 0) -> Tensor");
  m.def("mconvlconv(Tensor input, Tensor weight, Tensor? bias = None, int[2] s = 1, int[2] d = 1, int[2] p = 0, int[2] u = 1, int[2] r = 0) -> Tensor");
  m.def("snd2col(Tensor input, int kernel_width, int s = 1, int d = 1, int p = 0, int u = 1, int r = 0) -> Tensor");
  m.def("col2snd(Tensor input, int snd_width, int s = 1, int d = 1, int p = 0, int u = 1) -> Tensor");
  m.def("spec2col(Tensor input, int[2] kernel_size, int[2] s = 1, int[2] d = 1, int[2] p = 0, int[2] u = 1, int[2] r = 0) -> Tensor");
  m.def("col2spec(Tensor input, int[2] spec_size, int[2] s = 1, int[2] d = 1, int[2] p = 0, int[2] u = 1) -> Tensor");
}

PYBIND11_MODULE(_torch_ext, m) {
  m.doc() = "PyTorch wrapper of libmellin (import, then use torch.ops)";
}