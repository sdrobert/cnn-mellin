"""Pytorch implementations of Mellin-related ops using only built-in ops"""

from typing import Optional, Tuple
import torch

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
    weights: torch.Tensor,
    out: torch.Tensor,
    s: int,
    d: int,
    p: int,
    u: int,
) -> None:
    X = in_.size(0)
    W = weights.size(0)
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
            out[y] += torch.mm(in_[x], weights[w])


@torch.jit.script
def _mconv1d(
    in_: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    s: int,
    d: int,
    p: int,
    u: int,
) -> None:
    X = in_.size(0)
    W = weights.size(0)
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
            out[y] += torch.mm(in_[x], weights[w])


@torch.jit.script
def _lcorr1d(
    in_: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    s: int,
    d: int,
    p: int,
    u: int,
) -> None:
    X = in_.size(0)
    W = weights.size(0)
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
            out[y] += torch.mm(in_[x], weights[w])


@torch.jit.script
def _lconv1d(
    in_: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    s: int,
    d: int,
    p: int,
    u: int,
) -> None:
    X = in_.size(0)
    W = weights.size(0)
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
            out[y] += torch.mm(in_[x], weights[w])


@torch.jit.script
def _mcorrlcorr(
    in_: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
) -> None:
    X = in_.size(0)
    W = weights.size(0)
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
            _lcorr1d(in_[x], weights[w], out[y], s2, d2, p2, u2)


@torch.jit.script
def _mconvlconv(
    in_: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    s: Tuple[int, int],
    d: Tuple[int, int],
    p: Tuple[int, int],
    u: Tuple[int, int],
) -> None:
    X = in_.size(0)
    W = weights.size(0)
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
            _lconv1d(in_[x], weights[w], out[y], s2, d2, p2, u2)


def _mcorr1d_output_size(X: int, W: int, s: int, d: int, p: int, u: int, r: int) -> int:
    return ((p + 1) * (X + u - 1) - 1) // (W + d - 1) - s + 2 + r


def _lcorr1d_output_size(X: int, W: int, s: int, d: int, p: int, u: int, r: int) -> int:
    return (u * (X - 1) + p - d * (W - 1)) // s + 1 + r


class _MCorr1dOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        in_: torch.Tensor,
        weights: torch.Tensor,
        bias: Optional[torch.Tensor],
        s: int,
        d: int,
        p: int,
        r: int,
    ) -> torch.Tensor:
        if in_.dim() != 3:
            raise RuntimeError("in_ must be 3 dimensional")
        if weights.dim() != 3:
            raise RuntimeError("weights must be 3 dimensional")
        X, N, C_in = in_.shape
        if weights.size(1) != C_in:
            raise RuntimeError(
                f"Number of input channels differ between in_ ({C_in}) and weights "
                f"({weights.size(1)})"
            )
        W = weights.size(0)
        C_out = weights.size(2)
        Y = _mcorr1d_output_size(X, W, s, d, p, 1, r)
        if bias is not None:
            if bias.dim() != 1:
                raise RuntimeError("bias must be 1 dimensional")
            if bias.size(0) != C_out:
                raise RuntimeError(
                    f"Number of output channels differ between weights ({C_out}) "
                    f"and bias ({bias.size(0)})"
                )
            out = bias.view(1, 1, C_out).repeat(Y, N, 1)
        else:
            out = in_.zeros(Y, N, C_out)
        ctx.save_for_backward(in_, weights, bias)
        ctx.s, ctx.d, ctx.p = s, d, p
        _mcorr1d(in_, weights, out, s, d, p, 1)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        in_, weights, bias = ctx.saved_tensors
        grad_in = grad_weights = grad_bias = None
        if ctx.needs_input_grad[0]:
            # backward op for mcorr1d's grad_in is
            # mconv1d(weights, grad_out.tranpose(1, 2), 1, s, p, d)
            # weights (W, C_in, C_out) -> in_' of (X', N', C_in')
            # grad_out.tranpose(1, 2) (Y, C_out, N) -> weights' of (W', C_in', C_out')
            # out' (Y', N', C_out') -> grad_in.transpose(1, 2) (X, C_in, N)
            grad_in = torch.zeros_like(in_)
            _mconv1d(
                weights,
                grad_out.transpose(1, 2),
                grad_in.transpose(1, 2),
                1,
                ctx.s,
                ctx.p,
                ctx.d,
            )
        if ctx.needs_input_grad[1]:
            # backward op for mcorr1d's grad_weights is
            # mcorr1d(in_, grad_out, d, s, p, 1)
            # in_.transpose(1, 2) (X, C_in, N) -> in_' (X', N', C_in')
            # grad_out (Y, N, C_out) -> weights' (W', C_in', C_out')
            # out' (Y', N', C_out') -> grad_weights (W, C_in, C_out)
            grad_weights = torch.zeros_like(weights)
            _mcorr1d(
                in_.transpose(1, 2), grad_out, grad_weights, ctx.d, ctx.s, ctx.p, 1
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_out.flatten(0, 1).sum(0)
        return grad_in, grad_weights, grad_bias, None, None, None, None


mcorr1d = _MCorr1dOp.apply


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
