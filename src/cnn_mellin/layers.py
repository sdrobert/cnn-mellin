"""Pytorch implementations of Mellin-related ops using only built-in ops"""

import torch

__all__ = ["dilation_lift", "DilationLift", "LogCompression", "ScaledGaussianNoise"]


@torch.jit.script
def dilation_lift(input: torch.Tensor, tau: torch.Tensor, dim: int) -> torch.Tensor:
    dim_ = dim + input.dim() if dim < 0 else dim
    if dim_ >= input.dim() or dim_ < 0:
        raise RuntimeError(
            f"{dim} not a valid dimension for a tensor if dimension {input.dim()}"
        )
    old_shape = input.shape

    input = input.unsqueeze(-1).flatten(dim_ + 1)
    exp_ = torch.arange(
        1, old_shape[dim] + 1, dtype=input.dtype, device=input.device
    ) ** tau.to(input.dtype)
    return (input * exp_.unsqueeze(-1)).reshape(old_shape)


@torch.jit.script
def scaled_gaussian_noise(
    x: torch.Tensor, eps: float, dim: int, training: bool = True
) -> torch.Tensor:
    dim_ = dim + x.dim() if dim < 0 else dim
    if dim_ >= x.dim() or dim_ < 0:
        raise RuntimeError(
            f"{dim} not a valid dimension for a tensor if dimension {x.dim()}"
        )
    if training is False:
        return x
    mu, _ = x.abs().max(dim, keepdim=True)
    sigma = (mu * eps) * torch.randn_like(x)
    return x + sigma


class DilationLift(torch.nn.Module):
    r"""Perform the dilation lifting function

    For some positive :math:`\tau`, applies the map

    .. math::

        output[t - 1] = t^\tau input[t - 1]

    to the input :math:`input` along the dimension specified by `dim`. :math:`\tau` is
    learnable

    Parameters
    ----------
    dim : int, optional
        The dimension of the input to apply the lifting function to
    """

    __constants__ = ["dim"]

    dim: int
    log_tau: torch.Tensor

    def __init__(self, dim=2):
        super(DilationLift, self).__init__()
        self.log_tau = torch.nn.Parameter(torch.tensor(0.0))
        self.dim = dim
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return dilation_lift(input, self.log_tau.exp(), self.dim)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.log_tau)


class LogCompression(torch.nn.Module):
    """Apply pointwise log-compression to a tensor

    For input tensor `input` with multi-index ``i``, pointwise log-compression is
    defined as

        out[i] = log(eps + abs(input[i]))

    For some learnable epsilon > 0
    """

    log_eps: torch.Tensor

    def __init__(self):
        super().__init__()
        self.log_eps = torch.nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.log_eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input.abs() + self.log_eps.exp()).log()


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

    __constants__ = ["dim", "eps"]
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return scaled_gaussian_noise(x, self.eps, self.dim, self.training)
