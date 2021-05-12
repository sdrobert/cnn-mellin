"""Pytorch implementations of Mellin-related ops using only built-in ops"""

import torch

__all__ = ["dilation_lift", "DilationLift"]


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
