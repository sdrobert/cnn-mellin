"""Pytorch implementations of Mellin-related ops using only built-in ops"""

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
