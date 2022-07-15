# I ported the checking functions from
# https://github.com/pytorch/pytorch/blob/v1.8.1/aten/src/ATen/TensorUtils.cpp
# and the design pattern for correlation layers
# https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/modules/conv.py
# which is BSD-style licensed:
# https://github.com/pytorch/pytorch/blob/v1.8.1/LICENSE


import os
import warnings

from typing import Any, List, Optional, Tuple, Union

try:
    from typing_extensions import Literal
except ImportError:
    from typing import Literal

import torch
import numpy as np

__all__ = [
    "lconv_support_size",
    "lconv_valid_size",
    "lcorr_support_size",
    "lcorr_valid_size",
    "mconv_support_size",
    "mconv_valid_size",
    "mcorr_support_size",
    "mcorr_valid_size",
    "mcorr1d",
    "MCorr1d",
    "mcorrlcorr",
    "MCorrLCorr",
    "snd2col",
    "spec2col",
]

try:
    _cdir = os.path.join(os.path.dirname(__file__), "ext")
    _idir = os.path.join(_cdir, "include")
    _csrc = [os.path.join(_cdir, x) for x in ("torch_cpu.cpp", "torch.cpp")]
    _icppsrc = [
        os.path.join(_idir, "cdrobert", "mellin", x)
        for x in ("config_cpu.h", "config.h", "mconv.h")
    ]
    _icusrc = (
        os.path.join(_idir, "cdrobert", "mellin", x)
        for x in ("config_cuda.h", "mconv_cuda.cuh")
    )
    if torch.cuda.is_available() and all(os.path.isfile(x) for x in _icusrc):
        _csrc.append(os.path.join(_cdir, "torch_cuda.cu"))
    _missing = [x for x in _csrc + _icppsrc if not os.path.isfile(x)]
    assert not len(_missing), f"{_missing} don't exist"

    from torch.utils.cpp_extension import load

    load("mellin", _csrc, extra_include_paths=[_idir], is_python_module=False)

    _ext_mcorr1d = torch.ops.mellin.mcorr1d
    _ext_mcorrlcorr = torch.ops.mellin.mcorrlcorr
    _ext_snd2col = torch.ops.mellin.snd2col
    _ext_spec2col = torch.ops.mellin.spec2col
except Exception as e:
    warnings.warn(
        f"Could not JIT compile extensions: {e}. Falling back to native versions."
    )
    _ext_mcorr1d = _ext_mcorrlcorr = _ext_snd2col = _ext_spec2col = None

_TensorArg = Tuple[torch.Tensor, str, int]


def lconv_support_size(
    kernel_size: int, input_size: int, s: int = 1, d: int = 1, p: int = 0, u: int = 1
) -> int:
    return max((u * (input_size - 1) + d * (kernel_size - 1) - p) // s + 1, 0)


def lconv_valid_size(
    kernel_size: int, input_size: int, s: int = 1, d: int = 1, p: int = 0, u: int = 1
) -> int:
    return max((u * (input_size - 1) - p) // s + 1, 0)


def lcorr_support_size(
    kernel_size: int, input_size: int, s: int = 1, d: int = 1, p: int = 0, u: int = 1
) -> int:
    return max((u * (input_size - 1) + p) // s + 1, 0)


def lcorr_valid_size(
    kernel_size: int, input_size: int, s: int = 1, d: int = 1, p: int = 0, u: int = 1
) -> int:
    return max((u * (input_size - 1) - d * (kernel_size - 1) + p) // s + 1, 0)


def mconv_support_size(
    kernel_size: int, input_size: int, s: int = 1, d: int = 1, p: int = 0, u: int = 1
) -> int:
    return max(((kernel_size + d - 1) * (input_size + u - 1)) // (p + 1) - s + 1, 0)


def mconv_valid_size(
    kernel_size: int, input_size: int, s: int = 1, d: int = 1, p: int = 0, u: int = 1
) -> int:
    return max((d * (input_size + u - 1)) // (p + 1) - s + 1, 0)


def mcorr_support_size(
    kernel_size: int, input_size: int, s: int = 1, d: int = 1, p: int = 0, u: int = 1
) -> int:
    return max(((input_size + u - 1) * (p + 1)) // d - s + 1, 0)


def mcorr_valid_size(
    kernel_size: int, input_size: int, s: int = 1, d: int = 1, p: int = 0, u: int = 1
) -> int:
    return max(((input_size + u - 1) * (p + 1)) // (kernel_size + d - 1) - s + 1, 0)


def _t_str(t: _TensorArg) -> str:
    _, name, pos = t
    return f"argument #{pos} '{name}'" if pos else f"'{name}'"


def _check_dim(c: str, t: _TensorArg, dim: int):
    if t[0].dim() != dim:
        raise RuntimeError(
            f"Expected {dim}-dimensional arg[0], but got {t[0].dim()}-dimensional "
            f"tensor for {_t_str(t)} (while checking arguments for {c})"
        )


def _check_size(c: str, t: _TensorArg, dim: int, size: int):
    if t[0].size(dim) != size:
        raise RuntimeError(
            f"Expected tensor to have size {size} at dimension {dim}, but got size "
            f"{t[0].size(dim)} for {_t_str(t)} (while checking arguments for {c})"
        )


# similar to checkSameGPU, but we soften the requirement to the same device
# similar to checkSameGPU, but we soften the requirement to the same device
def _check_same_device(c: str, t1: _TensorArg, t2: _TensorArg):
    if str(t1[0].device) != str(t2[0].device):
        raise RuntimeError(
            f"Expected tensor for {_t_str(t1)} to have the same device as tensor for "
            f"{_t_str(t2)}, but device {t1[0].device} does not equal {t2[0].device} "
            f"(while checking arguments for {c})"
        )


def _check_all_same_device(c: str, tensors: List[Optional[_TensorArg]]):
    t0: Optional[_TensorArg] = None
    for t in tensors:
        if t is None:
            continue
        elif t0 is None:
            t0 = t
        else:
            _check_same_device(c, t0, t)


def _tuple_error_msg(c: str, name: str, len_: int, x: Any) -> str:
    return (
        f"Expected '{name}' to be either an integer or a tuple of {len_} integers, "
        f"got {x} (while checking argumengs for {c})"
    )


def _tuple_native(c: str, name: str, len_: int, x: Any) -> Tuple[int, ...]:
    if isinstance(x, int):
        return tuple(x for _ in range(len_))
    try:
        x = tuple(x)
        if len(x) == len_ and all(isinstance(y, int) for y in x):
            return x
    except:
        pass
    raise RuntimeError(_tuple_error_msg(c, name, len_, x))


def _pair(c: str, name: str, x: Any) -> Tuple[int, int]:
    if torch.jit.is_scripting():
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, List[int]) and len(x) == 2:
            return (x[0], x[1])
        elif isinstance(x, Tuple[int, int]):
            return x
        else:
            raise RuntimeError(_tuple_error_msg(c, name, 2, x))
    else:
        return _tuple_native(c, name, 2, x)


def _single(c: str, name: str, x: Any) -> Tuple[int]:
    if torch.jit.is_scripting():
        if isinstance(x, int):
            return (x,)
        elif isinstance(x, List[int]) and len(x) == 1:
            return (x[0],)
        elif isinstance(x, Tuple[int]):
            return x
        else:
            raise RuntimeError(_tuple_error_msg(c, name, 1, x))
    else:
        return _tuple_native(c, name, 1, x)


# we don't care if f or g is transposed: the code is agnostic to contiguity
# we do need to check the device of tensors since we're not explicitly dispatched
@torch.jit.script
def _check_and_infer_arguments_conv(
    c: str, dim: int, f: torch.Tensor, g: torch.Tensor, bias: Optional[torch.Tensor]
) -> Tuple[int, int, int, int, int, int, int]:
    f_arg, g_arg = (f, "weight", 2), (g, "input", 1)
    _check_dim(c, f_arg, dim)
    _check_dim(c, g_arg, dim)
    f_sizes, g_sizes = f.shape, g.shape
    c_out, c_in, batch = f_sizes[0], f_sizes[1], g_sizes[0]
    nfx, ngx = f_sizes[2], g_sizes[2]
    nfy, ngy = (f_sizes[3], g_sizes[3]) if dim == 4 else (1, 1)
    _check_size(c, g_arg, 1, c_in)
    bias_arg: Optional[_TensorArg] = None
    if bias is not None:
        bias_arg = (bias, "bias", 3)
        _check_dim(c, bias_arg, 1)
        _check_size(c, bias_arg, 0, c_out)
    _check_all_same_device(c, [f_arg, g_arg, bias_arg])
    return c_out, c_in, batch, nfx, ngx, nfy, ngy


@torch.jit.script
def _check_and_infer_arguments_col(
    c: str, dim: int, g: torch.Tensor
) -> Tuple[int, int, int, int]:
    g_arg = (g, "input", 1)
    _check_dim(c, g_arg, dim)
    g_sizes = g.shape
    ngx = g_sizes[2]
    ngy = g_sizes[3] if dim >= 4 else 1
    ngz = g_sizes[4] if dim >= 5 else 1
    ngq = g_sizes[5] if dim >= 6 else 1
    return ngx, ngy, ngz, ngq


@torch.jit.script
def _build_output(
    f: torch.Tensor, bias: Optional[torch.Tensor], size: List[int]
) -> torch.Tensor:
    # not sure why it has to be this way as opposed to just expand.contiguous,
    # but the numerical gradient will mess up otherwise...
    out = f.new_empty(size)
    if bias is None:
        out.zero_()
    else:
        view_size = [1, size[1]] + [1] * (len(size) - 2)
        out.copy_(bias.view(view_size).expand(size))
    return out


# warning: not appropriate for c++, which uses truncation in its division rather than
# flooring
@torch.jit.script
def _ceil_div(num: int, denom: int) -> int:
    return -(-num // denom)


@torch.jit.script
def _native_op_lconv1d_inner(
    f: torch.Tensor, g: torch.Tensor, h: torch.Tensor, s: int, d: int, p: int, u: int
):
    nf, ng, nh = f.size(2), g.size(2), h.size(2)
    for hx in range(nh):
        num_x_part = s * hx + p
        min_fx = max(_ceil_div(num_x_part - u * (ng - 1), d), 0)
        max_fx = min(_ceil_div(num_x_part + 1, d), nf)
        for fx in range(min_fx, max_fx):
            num_x = num_x_part - d * fx
            if num_x % u:
                continue
            # f[..., k] is (c_out, c_in)
            # g[..., i - k + p] is (batch, c_in)
            # h[..., i] is (batch, c_out)
            f_p = f[..., fx]
            g_p = g[..., num_x // u]
            h[..., hx] += torch.mm(g_p, f_p.transpose(0, 1))


@torch.jit.script
def _native_op_lcorr1d_inner(
    f: torch.Tensor, g: torch.Tensor, h: torch.Tensor, s: int, d: int, p: int, u: int
):
    nf, ng, nh = f.size(2), g.size(2), h.size(2)
    for hx in range(nh):
        num_x_part = s * hx - p
        min_fx = max(_ceil_div(-num_x_part, d), 0)
        max_fx = min(_ceil_div(u * ng - num_x_part, d), nf)
        for fx in range(min_fx, max_fx):
            num_x = num_x_part + d * fx
            if num_x % u:
                continue
            f_p = f[..., fx]
            g_p = g[..., num_x // u]
            h[..., hx] += torch.mm(g_p, f_p.transpose(0, 1))


@torch.jit.script
def _native_op_mconv1d_inner(
    f: torch.Tensor, g: torch.Tensor, h: torch.Tensor, s: int, d: int, p: int, u: int
):
    p += 1
    nf, ng, nh = f.size(2), g.size(2), h.size(2)
    for hx in range(nh):
        num_x = p * (hx + s)
        min_fx = max(_ceil_div(num_x, ng + u - 1) - d, 0)
        max_fx = min(_ceil_div(num_x + 1, u) - d, nf)
        for fx in range(min_fx, max_fx):
            denom = fx + d
            if num_x % denom:
                continue
            gx = num_x // denom - u
            f_p = f[..., fx]
            g_p = g[..., gx]
            h[..., hx] += torch.mm(g_p, f_p.transpose(0, 1))


@torch.jit.script
def _native_op_mconv1d_outer(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    s: int,
    d: int,
    p: int,
    u: int,
    r: int,
) -> torch.Tensor:
    c_out, _, batch, nf, ng, _, _ = _check_and_infer_arguments_conv(
        "mconv1d", 3, weight, input, bias
    )
    nh = mconv_valid_size(nf, ng, s, d, p, u) + r
    h = _build_output(weight, bias, [batch, c_out, nh])
    _native_op_mconv1d_inner(weight, input, h, s, d, p, u)
    return h


@torch.jit.script
def _native_op_mcorr1d_inner(
    f: torch.Tensor, g: torch.Tensor, h: torch.Tensor, s: int, d: int, p: int, u: int
):
    p += 1
    nf, ng, nh = f.size(2), g.size(2), h.size(2)
    for hx in range(nh):
        min_fx = max(_ceil_div(u * p, s + hx) - d, 0)
        max_fx = min(_ceil_div(p * (ng + u), s + hx) - d, nf)
        for fx in range(min_fx, max_fx):
            num_x = (hx + s) * (fx + d)
            if num_x % p:
                continue
            gx = num_x // p - u
            f_p = f[..., fx]
            g_p = g[..., gx]
            h[..., hx] += torch.mm(g_p, f_p.transpose(0, 1))


@torch.jit.script
def _native_op_mcorr1d_outer(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    s: int,
    d: int,
    p: int,
    u: int,
    r: int,
) -> torch.Tensor:
    c_out, _, batch, nf, ng, _, _ = _check_and_infer_arguments_conv(
        "mconv1d", 3, weight, input, bias
    )
    nh = mcorr_valid_size(nf, ng, s, d, p, u) + r
    h = _build_output(weight, bias, [batch, c_out, nh])
    _native_op_mcorr1d_inner(weight, input, h, s, d, p, u)
    return h


@torch.jit.script
def _native_op_mconvlconv_inner(
    f: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    sx: int,
    sy: int,
    dx: int,
    dy: int,
    px: int,
    py: int,
    ux: int,
    uy: int,
):
    px += 1
    nfx, ngx, nhx = f.size(2), g.size(2), h.size(2)
    for hx in range(nhx):
        num_x = px * (hx + sx)
        min_fx = max(_ceil_div(num_x, ngx + ux - 1) - dx, 0)
        max_fx = min(_ceil_div(num_x + 1, ux) - dx, nfx)
        for fx in range(min_fx, max_fx):
            denom_x = fx + dx
            if num_x % denom_x:
                continue
            gx = num_x // denom_x - ux
            _native_op_lconv1d_inner(
                f[:, :, fx], g[:, :, gx], h[:, :, hx], sy, dy, py, uy
            )


@torch.jit.script
def _native_op_mconvlconv_outer(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
    r: Tuple[int, int],
):
    c_out, _, batch, nfx, ngx, nfy, ngy = _check_and_infer_arguments_conv(
        "mconvlconv", 4, weight, input, bias
    )
    nhx = mconv_valid_size(nfx, ngx, s[0], d[0], p[0], u[0]) + r[0]
    nhy = lconv_valid_size(nfy, ngy, s[1], d[1], p[1], u[1]) + r[1]
    h = _build_output(weight, bias, [batch, c_out, nhx, nhy])
    _native_op_mconvlconv_inner(
        weight, input, h, s[0], s[1], d[0], d[1], p[0], p[1], u[0], u[1]
    )
    return h


@torch.jit.script
def _native_op_mcorrlcorr_inner(
    f: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    sx: int,
    sy: int,
    dx: int,
    dy: int,
    px: int,
    py: int,
    ux: int,
    uy: int,
):
    px += 1
    nfx, ngx, nhx = f.size(2), g.size(2), h.size(2)
    for hx in range(nhx):
        min_fx = max(_ceil_div(ux * px, sx + hx) - dx, 0)
        max_fx = min(_ceil_div(px * (ngx + ux), sx + hx) - dx, nfx)
        for fx in range(min_fx, max_fx):
            num_x = (hx + sx) * (fx + dx)
            if num_x % px:
                continue
            gx = num_x // px - ux
            _native_op_lcorr1d_inner(
                f[:, :, fx], g[:, :, gx], h[:, :, hx], sy, dy, py, uy
            )


@torch.jit.script
def _native_op_mcorrlcorr_outer(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
    r: Tuple[int, int],
):
    c_out, _, batch, nfx, ngx, nfy, ngy = _check_and_infer_arguments_conv(
        "mcorrlcorr", 4, weight, input, bias
    )
    nhx = mcorr_valid_size(nfx, ngx, s[0], d[0], p[0], u[0]) + r[0]
    nhy = lcorr_valid_size(nfy, ngy, s[1], d[1], p[1], u[1]) + r[1]
    h = _build_output(weight, bias, [batch, c_out, nhx, nhy])
    _native_op_mcorrlcorr_inner(
        weight, input, h, s[0], s[1], d[0], d[1], p[0], p[1], u[0], u[1]
    )
    return h


# XXX(sdrobert): We compute indices for snd2col and col2snd on the CPU before (possibly)
# sending it to the GPU. This is primarily because of some nondeterminism I found with
# the remainder and div operators (not scatter, strangely enough). However, since the
# idx tensors have long type (64-bit), it may be faster to do this anyways


@torch.jit.script
def _native_op_snd2col_inner(
    g: torch.Tensor, nf: int, ngg: int, s: int, d: int, p: int, u: int
) -> torch.Tensor:
    device = g.device
    batch, c_in, ng = g.size()
    ggx = torch.arange(ngg)
    fx = torch.arange(nf)
    num = (ggx.unsqueeze(0) + s) * (fx.unsqueeze(1) + d)
    if p:
        div = num.div(p + 1, rounding_mode="floor")
        gx = div - (num.remainder(p + 1) * (div + 1)) - u
    else:
        gx = num - u
    gx = gx.to(device).view(1, 1, nf, ngg).expand(batch, c_in, nf, ngg)
    return (
        g.unsqueeze(-1)
        .expand(batch, c_in, ng, ngg)
        .gather(2, gx % ng)
        .masked_fill_((gx >= ng) | (gx < 0), 0)
    )


@torch.jit.script
def _native_op_snd2col_outer(
    input: torch.Tensor,
    kernel_width: int,
    s: int = 1,
    d: int = 1,
    p: int = 0,
    u: int = 1,
    r: int = 0,
) -> torch.Tensor:
    ng = _check_and_infer_arguments_col("snd2col", 3, input)[0]
    ngg = mcorr_valid_size(kernel_width, ng, s, d, p, u) + r
    return _native_op_snd2col_inner(input, kernel_width, ngg, s, d, p, u)


@torch.jit.script
def _native_op_col2snd_inner(
    gg: torch.Tensor, ng: int, s: int, d: int, p: int, u: int
) -> torch.Tensor:
    device = gg.device
    batch, c_in, nf, ngg = gg.size()
    ggx = torch.arange(ngg)
    fx = torch.arange(nf)
    num = (ggx.unsqueeze(0) + s) * (fx.unsqueeze(1) + d)
    if p:
        div = num.div(p + 1, rounding_mode="floor")
        gx = div - (num.remainder(p + 1) * (div + 1)) - u
    else:
        gx = num - u
    gx = gx.to(device).view(1, 1, nf, ngg).expand(batch, c_in, nf, ngg)
    return (
        torch.zeros((batch, c_in, ng, ngg), device=device, dtype=gg.dtype)
        .scatter_add_(2, gx % ng, gg.masked_fill((gx >= ng) | (gx < 0), 0))
        .sum(3)
    )


@torch.jit.script
def _native_op_col2snd_outer(
    input: torch.Tensor, snd_width: int, s: int, d: int, p: int, u: int
) -> torch.Tensor:
    _check_and_infer_arguments_col("col2snd", 4, input)
    return _native_op_col2snd_inner(input, snd_width, s, d, p, u)


@torch.jit.script
def _native_op_spec2col_inner(
    g: torch.Tensor,
    nf: Tuple[int, int],
    ngg: Tuple[int, int],
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
) -> torch.Tensor:
    device = g.device
    batch, c_in, ngx, ngy = g.size()
    nggx, nggy = ngg
    nfx, nfy = nf
    sx, sy = s
    dx, dy = d
    px, py = p
    ux, uy = u
    ggx, ggy = torch.arange(nggx), torch.arange(nggy)
    fx, fy = torch.arange(nfx), torch.arange(nfy)
    numx = (ggx.unsqueeze(0) + sx) * (fx.unsqueeze(1) + dx)
    if px:
        divx = numx.div(px + 1, rounding_mode="floor")
        gx = divx - numx.remainder(px + 1) * (divx + 1) - ux
    else:
        gx = numx - ux
    gx = gx.to(device).view(1, 1, nfx, 1, nggx).expand(batch, c_in, nfx, ngy, nggx)
    g = (
        g.unsqueeze(4)
        .expand(batch, c_in, ngx, ngy, nggx)
        .gather(2, gx % ngx)
        .masked_fill_((gx >= ngx) | (gx < 0), 0)
    )
    numy = sy * ggy.unsqueeze(0) + dy * fy.unsqueeze(1) - py
    if uy:
        divy = numy.div(uy, rounding_mode="floor")
        gy = divy - numy.remainder(uy) * (divy + 1)
    else:
        gy = numy
    gy = (
        gy.to(device)
        .view(1, 1, 1, nfy, 1, nggy)
        .expand(batch, c_in, nfx, nfy, nggx, nggy)
    )
    return (
        g.unsqueeze(-1)
        .expand(batch, c_in, nfx, ngy, nggx, nggy)
        .gather(3, gy % ngy)
        .masked_fill_((gy >= ngy) | (gy < 0), 0)
    )


@torch.jit.script
def _native_op_spec2col_outer(
    input: torch.Tensor,
    kernel_size: Tuple[int, int],
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
    r: Tuple[int, int],
) -> torch.Tensor:
    ng = _check_and_infer_arguments_col("spec2col", 4, input)[:2]
    nggx = mcorr_valid_size(kernel_size[0], ng[0], s[0], d[0], p[0], u[0]) + r[0]
    nggy = lcorr_valid_size(kernel_size[1], ng[1], s[1], d[1], p[1], u[1]) + r[1]
    return _native_op_spec2col_inner(input, kernel_size, (nggx, nggy), s, d, p, u)


@torch.jit.script
def _native_op_col2spec_inner(
    gg: torch.Tensor,
    ng: Tuple[int, int],
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
) -> torch.Tensor:
    device = gg.device
    batch, c_in, nfx, nfy, nggx, nggy = gg.size()
    ngx, ngy = ng
    sx, sy = s
    dx, dy = d
    px, py = p
    ux, uy = u
    ggx, ggy = torch.arange(nggx), torch.arange(nggy)
    fx, fy = torch.arange(nfx), torch.arange(nfy)
    numx = (ggx.unsqueeze(0) + sx) * (fx.unsqueeze(1) + dx)
    if px:
        divx = numx.div(px + 1, rounding_mode="floor")
        gx = divx - numx.remainder(px + 1) * (divx + 1) - ux
    else:
        gx = numx - ux
    gx = (
        gx.to(device)
        .view(1, 1, nfx, 1, nggx, 1)
        .expand(batch, c_in, nfx, nfy, nggx, nggy)
    )
    numy = sy * ggy.unsqueeze(0) + dy * fy.unsqueeze(1) - py
    if uy:
        divy = numy.div(uy, rounding_mode="floor")
        gy = divy - numy.remainder(uy) * (divy + 1)
    else:
        gy = numy
    gy = (
        gy.to(device)
        .view(1, 1, 1, nfy, 1, nggy)
        .expand(batch, c_in, nfx, nfy, nggx, nggy)
    )
    nfxy, ngxy, nggxy = nfx * nfy, ngx * ngy, nggx * nggy
    gg = gg.masked_fill((gx >= ngx) | (gx < 0) | (gy >= ngy) | (gy < 0), 0).view(
        batch, c_in, nfxy, nggxy
    )
    gxy = ((gx % ngx) * ngy + gy % ngy).view(batch, c_in, nfxy, nggxy)
    return (
        torch.zeros((batch, c_in, ngxy, nggxy), device=device, dtype=gg.dtype)
        .scatter_add_(2, gxy, gg)
        .sum(3)
        .view(batch, c_in, ngx, ngy)
    )


@torch.jit.script
def _native_op_col2spec_outer(
    input: torch.Tensor,
    spec_size: Tuple[int, int],
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
) -> torch.Tensor:
    _check_and_infer_arguments_col("col2spec", 6, input)
    return _native_op_col2spec_inner(input, spec_size, s, d, p, u)


class _NativeMCorr1dOp(torch.autograd.Function):
    @classmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(cls, ctx, input, weight, bias=None, s=1, d=1, p=0, u=1, r=0):
        ctx.save_for_backward(input, weight, bias)
        ctx.s, ctx.d, ctx.p, ctx.u = s, d, p, u
        return _native_op_mcorr1d_outer(input, weight, bias, s, d, p, u, r)

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, grad_out: torch.Tensor):
        input: torch.Tensor = ctx.saved_tensors[0]
        weight: torch.Tensor = ctx.saved_tensors[1]
        bias: torch.Tensor = ctx.saved_tensors[2]
        s: int = ctx.s
        d: int = ctx.d
        p: int = ctx.p
        u: int = ctx.u
        grad_input: Optional[torch.Tensor] = None
        grad_weight: Optional[torch.Tensor] = None
        grad_bias: Optional[torch.Tensor] = None
        nf, ng, nh = weight.size(2), input.size(2), grad_out.size(2)

        if ctx.needs_input_grad[0]:
            r = ng - mconv_valid_size(nh, nf, u, s, p, d)
            grad_input = _native_op_mconv1d_outer(
                weight.transpose(0, 1), grad_out, None, u, s, p, d, r
            ).transpose(0, 1)

        if ctx.needs_input_grad[1]:
            r = nf - mcorr_valid_size(nh, ng, d, s, p, u)
            grad_weight = _native_op_mcorr1d_outer(
                input.transpose(0, 1), grad_out.transpose(0, 1), None, d, s, p, u, r
            ).transpose(0, 1)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_out.sum(0).sum(1)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


_native_mcorr1d = _NativeMCorr1dOp.apply
_deft_mcorr1d = _native_mcorr1d if _ext_mcorr1d is None else _ext_mcorr1d


class _NativeMCorrLCorrOp(torch.autograd.Function):
    @classmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        cls,
        ctx,
        input,
        weight,
        bias=None,
        s=(1, 1),
        d=(1, 1),
        p=(0, 0),
        u=(1, 1),
        r=(0, 0),
    ):
        ctx.save_for_backward(input, weight, bias)
        ctx.s, ctx.d, ctx.p, ctx.u = s, d, p, u
        return _native_op_mcorrlcorr_outer(input, weight, bias, s, d, p, u, r)

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, grad_out: torch.Tensor):
        input: torch.Tensor = ctx.saved_tensors[0]
        weight: torch.Tensor = ctx.saved_tensors[1]
        bias: torch.Tensor = ctx.saved_tensors
        s: Tuple[int, int] = ctx.s
        d: Tuple[int, int] = ctx.d
        p: Tuple[int, int] = ctx.p
        u: Tuple[int, int] = ctx.u
        grad_input: Optional[torch.Tensor] = None
        grad_weight: Optional[torch.Tensor] = None
        grad_bias: Optional[torch.Tensor] = None
        nfx, ngx, nhx = weight.size(2), input.size(2), grad_out.size(2)
        nfy, ngy, nhy = weight.size(3), input.size(3), grad_out.size(3)

        if ctx.needs_input_grad[0]:
            rx = ngx - mconv_valid_size(nhx, nfx, u[0], s[0], p[0], d[0])
            ry = ngy - lconv_valid_size(nhy, nfy, u[1], s[1], p[1], d[0])
            grad_input = _native_op_mconvlconv_outer(
                weight.transpose(0, 1), grad_out, None, u, s, p, d, (rx, ry)
            ).transpose(0, 1)

        if ctx.needs_input_grad[1]:
            rx = nfx - mcorr_valid_size(nhx, ngx, d[0], s[0], p[0], u[0])
            ry = nfy - lcorr_valid_size(nhy, ngy, d[1], s[1], p[1], u[1])
            grad_weight = _native_op_mcorrlcorr_outer(
                input.transpose(0, 1),
                grad_out.transpose(0, 1),
                None,
                d,
                s,
                p,
                u,
                (rx, ry),
            ).transpose(0, 1)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_out.sum(0).flatten(1).sum(1)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


_native_mcorrlcorr = _NativeMCorrLCorrOp.apply
_deft_mcorrlcorr = _native_mcorrlcorr if _ext_mcorrlcorr is None else _ext_mcorrlcorr


class _NativeSnd2ColOp(torch.autograd.Function):
    @classmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        cls, ctx, input: torch.Tensor, kernel_width: int, s=1, d=1, p=0, u=1, r=0,
    ):
        ctx.s, ctx.d, ctx.p, ctx.u = s, d, p, u
        output = _native_op_snd2col_outer(input, kernel_width, s, d, p, u, r)
        ctx.snd_width = input.size(2)
        return output

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, grad_out: torch.Tensor):
        return (
            _native_op_col2snd_outer(
                grad_out, ctx.snd_width, ctx.s, ctx.d, ctx.p, ctx.u
            ),
            None,
            None,
            None,
            None,
            None,
            None,
        )


_native_snd2col = _NativeSnd2ColOp.apply
_deft_snd2col = _native_snd2col if _ext_snd2col is None else _ext_snd2col


class _NativeSpec2ColOp(torch.autograd.Function):
    @classmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        cls,
        ctx,
        input: torch.Tensor,
        kernel_size: Tuple[int, int],
        s=(1, 1),
        d=(1, 1),
        p=(0, 0),
        u=(1, 1),
        r=(0, 0),
    ):
        ctx.s, ctx.d, ctx.p, ctx.u = s, d, p, u
        output = _native_op_spec2col_outer(input, kernel_size, s, d, p, u, r)
        ctx.spec_size = input.shape[2:]
        return output

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, grad_out: torch.Tensor):
        return (
            _native_op_col2spec_outer(
                grad_out, ctx.spec_size, ctx.s, ctx.d, ctx.p, ctx.u
            ),
            None,
            None,
            None,
            None,
            None,
            None,
        )


_native_spec2col = _NativeSpec2ColOp.apply
_deft_spec2col = _native_spec2col if _ext_spec2col is None else _ext_spec2col


# XXX(sdrobert): We wrap the above implementations (rather than using _deft* directly)
# for two reasons: one, to document the functions in only one place; two, to hide the
# "u" parameter from the end user.


def mcorr1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    s: int = 1,
    d: int = 1,
    p: int = 0,
    r: int = 0,
    _impl: Literal[None, "mm", "direct"] = None,
) -> torch.Tensor:
    """Apply a 1-dimensional Mellin correlation to multi-channel input

    See :class:`MCorr1d` for more details about the parameters and output shape.

    Parameters
    ----------
    input : torch.Tensor
        Of shape ``(batch, in_channels, T_in)``.
    weight : torch.Tensor
        Of shape ``(out_channels, in_channels, T_kern)``.
    s : int, optional
        Mellin "stride" analogue.
    d : int, optional
        Mellin "dilation" analogue.
    p : int, optional
        Mellin (left-)"padding" analogue.
    r : int, optional
        Right-padding parameter.

    Returns
    -------
    out : torch.Tensor
        Of shape ``(batch, out_channels, T_out)``.
    """
    if _impl is None:
        _impl = "mm" if weight.size(2) < 10 else "direct"
    if _impl == "direct":
        return _deft_mcorr1d(input, weight, bias, s, d, p, 1, r)
    elif _impl == "mm":
        gg = _deft_snd2col(input, weight.size(2), s, d, p, 1, r)
        out = torch.matmul(weight.flatten(1, 2), gg.flatten(1, 2))
        if bias is not None:
            out = out + bias.unsqueeze(-1)
        return out
    else:
        raise NotImplementedError(
            f'_impl "{_impl}" not supported (while checking arguments for mcorr1d)'
        )


def mcorrlcorr(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    s: Any = 1,
    d: Any = 1,
    p: Any = 0,
    r: Any = 0,
    _impl: Literal[None, "mm", "direct"] = None,
) -> torch.Tensor:
    """Apply 2-dimensional multi-channel correlation: Mellin first; linear second

    See :class:`MCorrLCorr` for more details about the parameters and output shape.

    Parameters
    ----------
    input : torch.Tensor
        Of shape ``(batch, in_channels, T_in, F_in)``.
    weight : torch.Tensor
        Of shape ``(out_channels, in_channels, T_kern, F_kern)``.
    s : int or (int, int), optional
        Stride parameter. An integer `s` is the same as passing ``(s, s)``.
    d : int or (int, int), optional
        Dilation parameter. An integer `d` is the same as passing ``(d, d)``.
    p : int or (int, int), optional
        (Left-)padding parameter. An integer `p` is the same as passing ``(p, p)``.
    r : int or (int, int), optional
        Right-padding parameter. An integer `r` is the same as passing ``(r, r)``.

    Returns
    -------
    out : torch.Tensor
        Of shape ``(batch, out_channels, T_out, F_out)``.
    """
    s_ = _pair("mcorrlcorr", "s", s)
    d_ = _pair("mcorrlcorr", "d", d)
    p_ = _pair("mcorrlcorr", "p", p)
    r_ = _pair("mcorrlcorr", "r", r)
    if _impl is None:
        _impl = "mm" if weight.size(2) * weight.size(3) < 10 else "direct"
    if _impl == "direct":
        return _deft_mcorrlcorr(input, weight, bias, s_, d_, p_, (1, 1), r_)
    elif _impl == "mm":
        gg = _deft_spec2col(input, weight.shape[2:], s_, d_, p_, (1, 1), r_)
        out = torch.matmul(weight.flatten(1, 3), gg.flatten(1, 3).flatten(2)).unflatten(
            2, gg.shape[4:]
        )
        if bias is not None:
            out = out + bias.unsqueeze(-1).unsqueeze(-1)
        return out
    else:
        raise NotImplementedError(
            f'_impl "{_impl}" not supported (while checking arguments for mcorrlcorr)'
        )


def snd2col(
    input: torch.Tensor,
    kernel_width: int,
    s: int = 1,
    d: int = 1,
    p: int = 0,
    r: int = 0,
) -> torch.Tensor:
    return _deft_snd2col(input, kernel_width, s, d, p, 1, r)


def spec2col(
    input: torch.Tensor,
    kernel_size: Any,
    s: Any = 1,
    d: Any = 1,
    p: Any = 0,
    r: Any = 0,
) -> torch.Tensor:
    kernel_size_ = _pair("spec2col", "kernel_size", kernel_size)
    s_ = _pair("spec2col", "s", s)
    d_ = _pair("spec2col", "d", d)
    p_ = _pair("spec2col", "p", p)
    r_ = _pair("spec2col", "r", r)
    return _deft_spec2col(input, kernel_size_, s_, d_, p_, (1, 1), r_)


class _CorrNd(torch.nn.Module):

    __constants__ = ["s", "d", "p", "r"]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    s: Tuple[int, ...]
    d: Tuple[int, ...]
    p: Tuple[int, ...]
    r: Tuple[int, ...]

    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        s: Tuple[int, ...],
        d: Tuple[int, ...],
        p: Tuple[int, ...],
        r: Tuple[int, ...],
        bias: bool = True,
    ):
        super().__init__()
        self.s, self.d, self.p, self.r = s, d, p, r
        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return (
            f"{self.weight.size(1)}, {self.weight.size(0)}, "
            f"kernel_size={self.weight.shape[1:]}, "
            f"s={self.s}, d={self.d}, p={self.p}, r={self.r}"
        )

    def reset_parameters(self) -> None:
        # follow strategy from conv documentation (1.8.1)
        sqrt_k = (np.product(self.weight.shape[1:])) ** -0.5
        torch.nn.init.uniform_(self.weight, -sqrt_k, sqrt_k)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -sqrt_k, sqrt_k)


class MCorr1d(_CorrNd):
    r"""Apply a 1-dimensional Mellin correlation to multi-channel input

    Given `input` of shape ``(N, in_channels, T_in)``, :class:`MCorr1d` computes

    .. math::
        out[n, c_{out}] = bias[c_{out}] + \sum_{c_{in}=0}^{C_{in} - 1}
            weight[c_{out}, c_{in}] \star_m input[n, c_{in}]

    Where :math:`\psi \star_m x` is the Discrete Mellin Correlation along the third axis
    of the kernel :math:`\psi = weight[c_{out}, c_{in}, :]` and input :math:`x =
    input[n, c_{in}, :]`:

    .. math::
        (\psi \star_m x)[t] =
            \sum_{\tau=0}^{T_{kern} - 1} \psi[\tau]
                            x\left[\frac{(t + s)(\tau + d)}{(p + 1)} - 1\right]

    Parameters
    ----------
    in_channels : int
        :math:`C_{in}`, the number of channels in a given `input` signal.
    out_channels : int
        :math:`C_{out}`, the number of channels in a resulting `out` signal.
    kernel_size : int
        :math:`T_{kern}`, the number of nonzero coefficients along the Mellin dimension
        in the kernel `weight`.
    s : int, optional
        The stride parameter. In the Mellin domain, `s` shifts the output left:
        :math:`(\psi \star_m x)[t] \mapsto (\psi \star_m x)[t + s]`.
    d : int, optional
        The dilation parameter. In the Mellin domain, `d` shifts the kernel right:
        :math:`\psi[t] \mapsto \psi[t - d]`.
    p : int, optional
        The (left-)padding parameter. In the Mellin domain, `p` dilates the input:
        :math:`x[t] \mapsto x[(t + 1)/(p[0] + 1) - 1]`.
    r : int, optional
        Right-padding parameter. See the below discussion for more information.

    Returns
    -------
    out : torch.Tensor
        Of shape ``(batch, out_channels, T_out)``, where

        .. math::
            T_{out} = \left\lfloor
                    \frac{(T_{in} - 1)(p + 1)}{T_{kern} + d - 1}
                \right\rfloor - s + r + 1

        When ``r = 0``, :math:`T_{out} = T_{valid}` is the minimum size it could be to
        contain all "valid" elements of `out`. A valid element is one where the sum in
        :math:`\psi \star_m x` consists of products of coefficients of :math:`\psi` and
        :math:`x` whose indices are bound between :math:`[0, T_{kern})` and
        :math:`[0, T_{in})`, respectively. Lengths in the region

        .. math::
            T_{valid} \leq T_{out} \leq \left\lfloor
                    \frac{(T_{in} - 1)(p + 1)}{d}
                \right\rfloor - s + 1 = T_{support}

        Will include sums of products where the :math:`x` coefficient is padding, though
        one or more products will be nonzero. Choosing :math:`T_{out} > T_{support}` is
        equivalent to right-padding the difference :math:`T_{out} - T_{support}` with
        zeroes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        s: int = 1,
        d: int = 1,
        p: int = 0,
        r: int = 0,
        bias: bool = True,
    ):
        kernel_size = _single("MCorr1d", "kernel_size", kernel_size)
        s = _single("MCorr1d", "s", s)
        d = _single("MCorr1d", "d", d)
        p = _single("MCorr1d", "p", p)
        r = _single("MCorr1d", "r", r)
        super().__init__(in_channels, out_channels, kernel_size, s, d, p, r, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return mcorr1d(
            input, self.weight, self.bias, self.s[0], self.d[0], self.p[0], self.r[0]
        )


class MCorrLCorr(_CorrNd):
    r"""Apply 2-dimensional multi-channel correlation: Mellin first; linear second

    Given `input` of shape ``(N, in_channels, T_in, F_{in})``, :class:`MCorrLCorr`
    computes

    .. math::
        out[n, c_{out}] = bias[c_{out}] + \sum_{c_{in}=0}^{C_{in} - 1}
            weight[c_{out}, c_{in}] \star_{m,\ell} input[n, c_{in}]

    Where :math:`\cdot \star_{m,\ell} \cdot` is the Cartesian product between a
    1-dimensional Discrete Mellin Correlation (like :class:`MCorr1d`) and 1-dimensional
    Discrete Linear Correlation (like :class:`torch.nn.Conv1d`). Letting
    :math:`\psi = weight[c_{out}, c_{in}]` and :math:`x = input[n, c_{in}]`, the
    two-dimensional correlation is defined as

    .. math::
        (\psi \star_{m,\ell} x)[t, f] =
            \sum_{\substack{\tau\in[0,T_{kern}) \\ \omega\in[0,F_{kern})}}
                \psi[\tau, \omega]
                x\left[
                    \frac{(t + s[0])(\tau + d[0])}{p[0] + 1} - 1,
                    s[1] f + d[1] \omega - p[1]
                \right]

    Parameters
    ----------
    in_channels : int
        :math:`C_{in}`, the number of channels in a given `input` signal.
    out_channels : int
        :math:`C_{out}`, the number of channels in a resulting `out` signal.
    kernel_size : int or (int, int), optional
        :math:`(T_{kern},F_{kern})`, the number of nonzero coefficients in both the
        Mellin and linear dimensions. Setting `kernel_size` to an integer is the same as
        passing ``(kernel_size, kernel_size)``.
    s : int or (int, int), optional
        The stride parameter. In the Mellin domain, `s` shifts the output left:
        :math:`(\psi \star_{m,\ell} x)[t, f] \mapsto
        (\psi \star_{m,\ell} x)[t + s[0], f]`. In the linear domain, `s`
        compresses/downsamples the output:
        :math:`(psi \star_{m,\ell} x)[t, f] \mapsto (psi \star_{m,\ell} x)[t,s[1]f]`.
        Setting `s` to an integer is the same as passing ``(s, s)``.
    d : int or (int, int), optional
        The dilation parameter. In the Mellin domain, `d` shifts the kernel right:
        :math:`\psi[t,f] \mapsto \psi[t - d[0],f]`. In the linear domain, `d`
        dilates/upsamples the kernel: :math:`\psi[t,f] \mapsto \psi[t, f/d[1]]`.
        Setting `d` to an integer is the same as passing ``(d, d)``.
    p : int or (int, int), optional
        The (left-)padding parameter. In the Mellin domain, `p` dilates/upsamples the
        input: :math:`x[t, f] \mapsto x[(t + 1)/(p[0] + 1) - 1, f]`. In the linear
        domain, `p` shifts the input right: :math:`x[t, f] \mapsto x[t, f + p[1]]`.
        Setting `p` to an integer is the same as passing ``(p, p)``.
    r : int or (int, int), optional
        Right-padding parameter. Setting `r` to an integer is the same as passing
        ``(r, r)``. See the below discussion for more information.
    
    Returns
    -------
    out : torch.Tensor
        Of shape ``(batch, out_channels, T_out, F_out)``, where

        .. math::
            \begin{split}
                T_{out} &= \left\lfloor
                        \frac{(T_{in} - 1)(p[0] + 1)}{T_{kern} + d[0] - 1}
                    \right\rfloor - s[0] + r[0] + 1 \\
                F_{out} &= \left\lfloor
                        \frac{F_{in} - d[1](F_{kern} - 1) + p[1] - 1}{s[1]}
                    \right\rfloor + r[1] + 1
            \end{split}

        When ``r[0] = r[1] = 0``, :math:`T_{out} = T_{valid}` and
        :math:`F_{out} = F_{valid}` are the pair of sizes that together contain all the
        "valid" indices of `out`. A valid element is one where the sum in
        :math:`\psi \star_{m,\ell} x` consists of products of coefficients of
        :math:`\psi` and :math:`x` whose indices are bound between
        :math:`[0, T_{kern}) \times [0, F_{kern})` and :math:`[0, T_{in}) \times
        [0, F_{in})`, respectively. Lengths in the region

        .. math::
            \begin{split}
                T_{valid} \leq T_{out} &\leq \left\lfloor
                        \frac{(T_{in} - 1)(p[0] + 1)}{d[0]}
                    \right\rfloor - s[0] + 1 = T_{support} \\
                F_{valid} \leq F_{out} &\leq \left\lfloor
                        \frac{F_{in} + p[1] - 1}{s[1]}
                    \right\rfloor  + 1 = F_{support}
            \end{split}

        Will include sums of products where the :math:`x` coefficient is padding, though
        one or more products will be nonzero. Choosing :math:`T_{out} > T_{support}` is
        equivalent to right-padding the difference :math:`T_{out} - T_{support}` with
        zeroes, likewise for :math:`F_{out} > F_{support}`.
    
    Notes
    -----
    
    While ``s[1]`` and ``d[1]`` can be interpreted identically as the `stride` and
    `dilation` parameters in a linear correlation (:class:`torch.nn.Conv1d`), ``p[1]``
    is the asymmetric version of the `padding` parameter. ``p[1]`` effectively left-pads
    the signal and ``r[1]`` right-pads the signal; to reproduce the effects of
    `padding`, set ``p[1] = r[1] = padding``. We expose padding as an asymmetric
    operation because

    1. By setting ``p[1] = (F_kern - 1) // 2`` and ``r[1] = F_kern // 2`` (and
       (``s[1] == d[1] == 1``), the output will be the same size as the input
       ``F_in = F_out`` for both even- and odd-sized kernels.
    2. ``p[0]`` and ``r[0]`` behave differently since they are in the Mellin domain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        s: Union[int, Tuple[int, int]] = 1,
        d: Union[int, Tuple[int, int]] = 1,
        p: Union[int, Tuple[int, int]] = 0,
        r: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ):
        s = _pair("MCorrLCorr", "s", s)
        d = _pair("MCorrLCorr", "d", d)
        p = _pair("MCorrLCorr", "p", p)
        r = _pair("MCorrLCorr", "r", r)
        super().__init__(in_channels, out_channels, kernel_size, s, d, p, r, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return mcorrlcorr(input, self.weight, self.bias, self.s, self.d, self.p, self.r)
