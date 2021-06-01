"""Pytorch implementations of Mellin-related ops using only built-in ops"""

import collections
from typing import Optional, Tuple, Union
from itertools import repeat
import torch
import numpy as np

__all__ = ["dilation_lift", "DilationLift", "LogCompression", "ScaledGaussianNoise"]


@torch.jit.script
def dilation_lift(f: torch.Tensor, tau: torch.Tensor, dim: int) -> torch.Tensor:
    old_shape = f.shape
    if dim < 0:
        dim += len(old_shape)
    f = f.unsqueeze(-1).flatten(dim + 1)
    exp_ = torch.arange(
        1, old_shape[dim] + 1, dtype=f.dtype, device=f.device
    ) ** tau.to(f.dtype)
    return (f * exp_.unsqueeze(-1)).reshape(old_shape)


@torch.jit.script
def scaled_gaussian_noise(x: torch.Tensor, eps: float, dim: int) -> torch.Tensor:
    mu, _ = x.abs().max(dim, keepdim=True)
    sigma = (mu * eps) * torch.randn_like(x)
    return x + sigma


@torch.jit.script
def _mcorr1d(
    in_: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    s: int,
    d: int,
    p: int,
    u: int,
) -> None:
    X = in_.size(0)
    W = weight.size(0)
    Y = out.size(0)
    p += 1
    for y in range(Y):
        max_W = min((p * (X + u - 1)) // (y + s) - d + 1, W)
        min_W = max((p * u - 1) // (y + s) - d + 1, 0)
        for w in range(min_W, max_W):
            num = (y + s) * (w + d)
            if num % p:
                continue
            x = num // p - u
            out[y] += torch.mm(in_[x], weight[w])


@torch.jit.script
def _mconv1d(
    in_: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    s: int,
    d: int,
    p: int,
    u: int,
) -> None:
    X = in_.size(0)
    W = weight.size(0)
    Y = out.size(0)
    p += 1
    for y in range(Y):
        num = p * (y + s)
        min_W = max((num - 1) // (X + u - 1) - d + 1, 0)
        max_W = min(num // u - d + 1, W)
        for w in range(min_W, max_W):
            denom = w + d
            if num % denom:
                continue
            x = num // denom - u
            out[y] += torch.mm(in_[x], weight[w])


@torch.jit.script
def _lcorr1d(
    in_: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    s: int,
    d: int,
    p: int,
    u: int,
) -> None:
    X = in_.size(0)
    W = weight.size(0)
    Y = out.size(0)
    for y in range(Y):
        min_W = s * y - p
        max_W = min((u * (X - 1) - min_W) // d + 1, W)
        min_W = max((-min_W - 1) // d + 1, 0)
        for w in range(min_W, max_W):
            num = s * y + d * w - p
            if num % u:
                continue
            x = num // u
            out[y] += torch.mm(in_[x], weight[w])


@torch.jit.script
def _lconv1d(
    in_: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    s: int,
    d: int,
    p: int,
    u: int,
) -> None:
    X = in_.size(0)
    W = weight.size(0)
    Y = out.size(0)
    for y in range(Y):
        min_W = s * y + p
        max_W = min_W // d + 1
        min_W = max(0, (min_W - u * (X - 1) - 1) // d + 1)
        max_W = min(W, max_W)
        for w in range(min_W, max_W):
            num = s * y - d * w + p
            if num % u:
                continue
            x = num // u
            out[y] += torch.mm(in_[x], weight[w])


@torch.jit.script
def _mcorrlcorr(
    in_: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
) -> None:
    X = in_.size(0)
    W = weight.size(0)
    Y = out.size(0)
    s1, s2 = s
    d1, d2 = d
    p1, p2 = p
    u1, u2 = u
    p1 += 1
    for y in range(Y):
        max_W = min((p1 * (X + u1 - 1)) // (y + s1) - d1 + 1, W)
        min_W = max((p1 * u1 - 1) // (y + s1) - d1 + 1, 0)
        for w in range(min_W, max_W):
            num = (y + s1) * (w + d1)
            if num % p1:
                continue
            x = num // p1 - u1
            _lcorr1d(in_[x], weight[w], out[y], s2, d2, p2, u2)


@torch.jit.script
def _mconvlconv(
    in_: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
) -> None:
    X = in_.size(0)
    W = weight.size(0)
    Y = out.size(0)
    s1, s2 = s
    d1, d2 = d
    p1, p2 = p
    u1, u2 = u
    p1 += 1
    for y in range(Y):
        num = p1 * (y + s1)
        min_W = max((num - 1) // (X + u1 - 1) - d1 + 1, 0)
        max_W = min(num // u1 - d1 + 1, W)
        for w in range(min_W, max_W):
            denom = w + d1
            if num % denom:
                continue
            x = num // denom - u1
            _lconv1d(in_[x], weight[w], out[y], s2, d2, p2, u2)


def _mcorr1d_output_size(X: int, W: int, s: int, d: int, p: int, u: int, r: int) -> int:
    return max(((p + 1) * (X + u - 1) - 1) // (W + d - 1) - s + 2 + r, 0)


def _lcorr1d_output_size(X: int, W: int, s: int, d: int, p: int, u: int, r: int) -> int:
    return max((u * (X - 1) + p - d * (W - 1)) // s + 1 + r, 0)


class _MCorr1dOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        in_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        s: int = 1,
        d: int = 1,
        p: int = 0,
        r: int = 0,
    ) -> torch.Tensor:
        if in_.dim() != 3:
            raise RuntimeError("in_ must be 3 dimensional")
        if weight.dim() != 3:
            raise RuntimeError("weight must be 3 dimensional")
        X, N, C_in = in_.shape
        if weight.size(1) != C_in:
            raise RuntimeError(
                f"Number of input channels differ between in_ ({C_in}) and weight "
                f"({weight.size(1)})"
            )
        W = weight.size(0)
        C_out = weight.size(2)
        Y = _mcorr1d_output_size(X, W, s, d, p, 1, r)
        if bias is not None:
            if bias.dim() != 1:
                raise RuntimeError("bias must be 1 dimensional")
            if bias.size(0) != C_out:
                raise RuntimeError(
                    f"Number of output channels differ between weight ({C_out}) "
                    f"and bias ({bias.size(0)})"
                )
            out = bias.view(1, 1, C_out).repeat(Y, N, 1)
        else:
            out = in_.zeros(Y, N, C_out)
        ctx.save_for_backward(in_, weight, bias)
        ctx.s, ctx.d, ctx.p = s, d, p
        _mcorr1d(in_, weight, out, s, d, p, 1)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        in_, weight, bias = ctx.saved_tensors
        grad_in = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # backward op for mcorr1d's grad_in is
            # mconv1d(weight, grad_out.tranpose(1, 2), 1, s, p, d).transpose(1, 2)
            # weight (W, C_in, C_out) -> in_' of (X', N', C_in')
            # grad_out.tranpose(1, 2) (Y, C_out, N) -> weight' of (W', C_in', C_out')
            # out' (Y', N', C_out') -> grad_in.transpose(1, 2) (X, C_in, N)
            grad_in = torch.zeros_like(in_)
            _mconv1d(
                weight,
                grad_out.transpose(1, 2),
                grad_in.transpose(1, 2),
                1,
                ctx.s,
                ctx.p,
                ctx.d,
            )
        if ctx.needs_input_grad[1]:
            # backward op for mcorr1d's grad_weight is
            # mcorr1d(in_.transpose(1, 2), grad_out, d, s, p, 1)
            # in_.transpose(1, 2) (X, C_in, N) -> in_' (X', N', C_in')
            # grad_out (Y, N, C_out) -> weight' (W', C_in', C_out')
            # out' (Y', N', C_out') -> grad_weight (W, C_in, C_out)
            grad_weight = torch.zeros_like(weight)
            _mcorr1d(in_.transpose(1, 2), grad_out, grad_weight, ctx.d, ctx.s, ctx.p, 1)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_out.flatten(0, 1).sum(0)
        return grad_in, grad_weight, grad_bias, None, None, None, None


mcorr1d = _MCorr1dOp.apply


class _MCorrLCorrOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        in_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        s: Union[int, Tuple[int, int]] = 1,
        d: Union[int, Tuple[int, int]] = 1,
        p: Union[int, Tuple[int, int]] = 0,
        r: Union[int, Tuple[int, int]] = 0,
    ) -> torch.Tensor:
        if isinstance(s, int):
            s = (s, s)
        if isinstance(d, int):
            d = (d, d)
        if isinstance(p, int):
            p = (p, p)
        if isinstance(r, int):
            r = (r, r)
        if in_.dim() != 4:
            raise RuntimeError("in_ must be 4 dimensional")
        if weight.dim() != 4:
            raise RuntimeError("weight must be 4 dimensional")
        X1, X2, N, C_in = in_.shape
        W1, W2, C_in_, C_out = weight.shape
        if C_in_ != C_in:
            raise RuntimeError(
                f"Number of input channels differ between in_ ({C_in}) and weight "
                f"({C_in_})"
            )
        Y1 = _mcorr1d_output_size(X1, W1, s[0], d[0], p[0], 1, r[0])
        Y2 = _lcorr1d_output_size(X2, W2, s[1], d[1], p[1], 1, r[1])
        if bias is not None:
            if bias.dim() != 1:
                raise RuntimeError("bias must be 1 dimensional")
            if bias.size(0) != C_out:
                raise RuntimeError(
                    f"Number of output channels differ between weight ({C_out}) "
                    f"and bias ({bias.size(0)})"
                )
            out = bias.view(1, 1, 1, C_out).repeat(Y1, Y2, N, 1)
        else:
            out = in_.zeros(Y1, Y2, N, C_out)
        ctx.save_for_backward(in_, weight, bias)
        ctx.s, ctx.d, ctx.p = s, d, p
        _mcorrlcorr(in_, weight, out, s, d, p, (1, 1))
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        in_, weight, bias = ctx.saved_tensors
        grad_in = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # backward op for mcorr1d's grad_in is
            # mconvlconv(weight, grad_out.tranpose(2, 3), 1, s, p, d).transpose(2, 3)
            # weight (W1, W2, C_in, C_out) -> in_' of (X1', X2', N', C_in')
            # grad_out.tranpose(2, 3) (Y1, Y2, C_out, N) ->
            #                                    weight' of (W1', W2', C_in', C_out')
            # out' (Y1', Y2', N', C_out') -> grad_in.transpose(2, 3) (X1, X2, C_in, N)
            grad_in = torch.zeros_like(in_)
            _mconvlconv(
                weight,
                grad_out.transpose(2, 3),
                grad_in.transpose(2, 3),
                (1, 1),
                ctx.s,
                ctx.p,
                ctx.d,
            )
        if ctx.needs_input_grad[1]:
            # backward op for mcorr1d's grad_weight is
            # mcorrlcorr(in_.transpose(2, 3), grad_out, d, s, p, 1)
            # in_.transpose(2, 3) (X1, X2, C_in, N) -> in_' (X1', X2', N', C_in')
            # grad_out (Y1, Y2, N, C_out) -> weight' (W1', W2', C_in', C_out')
            # out' (Y1', Y2', N', C_out') -> grad_weight (W1, W2, C_in, C_out)
            grad_weight = torch.zeros_like(weight)
            _mcorrlcorr(
                in_.transpose(2, 3), grad_out, grad_weight, ctx.d, ctx.s, ctx.p, (1, 1)
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_out.flatten(0, 2).sum(0)
        return grad_in, grad_weight, grad_bias, None, None, None, None


mcorrlcorr = _MCorrLCorrOp.apply


# Copied from
# https://github.com/pytorch/pytorch/blob/2503028ff5bab90a6c93687ce4e294815f3e243a/torch/nn/modules/utils.py#L7
# (not public API)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)


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
        s: Tuple[int, ...] = 1,
        d: Tuple[int, ...] = 1,
        p: Tuple[int, ...] = 0,
        r: Tuple[int, ...] = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.s, self.d, self.p, self.r = s, d, p, r
        self.weight = torch.nn.Parameter(
            torch.empty(*kernel_size, in_channels, out_channels)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return (
            f"{self.weight.size(-2)}, {self.weight.size(-1)}, "
            f"kernel_size={self.weight.shape[:-2]}, "
            f"s={self.s}, d={self.d}, p={self.p}, r={self.r}"
        )

    def reset_parameters(self) -> None:
        # follow strategy from conv documentation (1.8.1)
        sqrt_k = (np.product(self.weight.shape[:-1])) ** -0.5
        torch.nn.init.uniform_(self.weight, -sqrt_k, sqrt_k)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -sqrt_k, sqrt_k)


class MCorr1d(_CorrNd):
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
        kernel_size, s, d, p, r = (_single(x) for x in (kernel_size, s, d, p, r))
        super().__init__(in_channels, out_channels, kernel_size, s, d, p, r, bias)

    def forward(self, in_: torch.Tensor) -> torch.Tensor:
        return mcorr1d(
            in_, self.weight, self.bias, self.s[0], self.d[0], self.p[0], self.r[0]
        )


class MCorrLCorr(_CorrNd):
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
        kernel_size, s, d, p, r = (_pair(x) for x in (kernel_size, s, d, p, r))
        super().__init__(in_channels, out_channels, kernel_size, s, d, p, r, bias)

    def forward(self, in_: torch.Tensor) -> torch.Tensor:
        return mcorrlcorr(in_, self.weight, self.bias, self.s, self.d, self.p, self.r)


class DilationLift(torch.nn.Module):
    r"""Perform the dilation lifting function

    For some positive :math:`\tau`, applies the map

    .. math::

        f[t - 1] = t^\tau f[t - 1]

    to the input :math:`f` along the dimension specified by `dim`. :math:`\tau` is
    learnable

    Parameters
    ----------
    dim : int, optional
        The dimension of the input to apply the lifting function to

    Attributes
    ----------
    log_tau : torch.nn.Parameter
        :math:`\log \tau`
    dim : int
    """

    def __init__(self, dim=2):
        super(DilationLift, self).__init__()
        self.log_tau = torch.nn.Parameter(torch.tensor(0.0))
        self.dim = dim
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def check_input(self, f: torch.Tensor) -> None:
        dim_ = self.dim + f.dim() if self.dim < 0 else self.dim
        if dim_ >= f.dim() or dim_ < 0:
            raise RuntimeError(
                f"{self.dim} not a valid dimension for a tensor if dimension {f.dim()}"
            )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        self.check_input(f)
        return dilation_lift(f, self.log_tau.exp(), self.dim)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.log_tau)


class LogCompression(torch.nn.Module):
    """Apply pointwise log-compression to a tensor

    For input tensor `x` with multi-index ``i``, pointwise log-compression is defined
    as

        x'[i] = log(eps + abs(x[i]))

    For some learnable epsilon > 0
    """

    def __init__(self):
        super().__init__()
        self.log_eps = torch.nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.log_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.abs() + self.log_eps.exp()).log()


class ScaledGaussianNoise(torch.nn.Module):
    r"""Add Gaussian noise to a signal as some fraction of the magnitude of a signal

    Let ``t`` index values of `x` along the dim `dim` and ``i`` index the remaining
    dimensions. During training, this layer applies noise of the form

    .. math::

        x'[i,t] = x[i,t] + \mathcal{N}(0, (\max_{t'} |x[i,t']|) \eps)

    During testing, the layer is a no-op.

    Parameters
    ----------
    dim : int, optional
        Which dimension to take the max along
    eps : float, optional
        The scale to multiply the maximum with
    """

    def __init__(self, dim: int = 1, eps: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def check_input(self, x: torch.Tensor) -> None:
        dim_ = self.dim + x.dim() if self.dim < 0 else self.dim
        if dim_ >= x.dim() or dim_ < 0:
            raise RuntimeError(
                f"{self.dim} not a valid dimension for a tensor if dimension {x.dim()}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check_input(x)
        if self.training and self.eps:
            return scaled_gaussian_noise(x, self.eps, self.dim)
        else:
            return x


@torch.jit.script
def my_dropout_2d(in_: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    # same as torch.nn.functional.dropout_2d, but on the last 2 indices instead of
    # the first
    if p < 0.0 or p > 1.0:
        raise RuntimeError(f"dropout probability has to be between 0 and 1, got {p}")
    if in_.dim() < 2:
        raise RuntimeError(f"in_ must be at least dimension 2 for 2d dropout")
    if not training or p == 0.0:
        return in_
    elif p == 1.0:
        return torch.zeros_like(in_)
    noise = in_.new_full((in_.size(-2), in_.size(-1)), 1 - p)
    noise.bernoulli_(1 - p)
    return in_ * noise
