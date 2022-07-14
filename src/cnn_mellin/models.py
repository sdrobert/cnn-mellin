"""Acoustic models"""

import warnings

from itertools import chain
from typing import Tuple

import param
import torch

from cnn_mellin.mconv import lcorr_valid_size, mcorr_valid_size, MCorr1d, MCorrLCorr
from cnn_mellin.layers import DilationLift, LogCompression

__all__ = [
    "AcousticModelParams",
    "AcousticModel",
]


class AcousticModelParams(param.Parameterized):
    use_log_compression = param.Boolean(
        True,
        doc="Whether to add a pointwise log(1 + eps) layer to the input with "
        "learnable eps",
    )
    use_lift = param.Boolean(
        True,
        doc="Whether to add the learnable lift operation to the network. Applied "
        "after log compression (if enabled)",
    )
    window_size = param.Integer(
        5,
        bounds=(1, None),
        softbounds=(2, 10),
        doc="The total number of audio elements per window in time",
    )
    window_stride = param.Integer(
        3,
        bounds=(1, None),
        softbounds=(1, 5),
        doc="The number of audio elements over to shift for subsequent windows",
    )
    convolutional_kernel_time = param.Integer(
        3,
        bounds=(1, None),
        softbounds=(1, 10),
        doc="The width of convolutional kernels along the time dimension",
    )
    convolutional_kernel_freq = param.Integer(
        3,
        bounds=(1, None),
        softbounds=(1, 20),
        doc="The width of convolutional kernels along the frequency dimension",
    )
    convolutional_initial_channels = param.Integer(
        64,
        bounds=(1, None),
        softbounds=(16, 64),
        doc="The number of channels in the initial convolutional layer",
    )
    convolutional_layers = param.Integer(
        5,
        bounds=(0, None),
        softbounds=(1, 6),
        doc="The number of layers in the convolutional part of the network",
    )
    convolutional_mellin = param.Boolean(
        False,
        doc="Whether the convolutional layers are mellin-linear (versus just linear)",
    )
    convolutional_time_factor = param.Integer(
        2,
        bounds=(1, None),
        softbounds=(1, 4),
        doc="The factor by which to reduce the size of the input along the "
        "time dimension between convolutional layers",
    )
    convolutional_freq_factor = param.Integer(
        1,
        bounds=(1, None),
        softbounds=(1, 2),
        doc="The factor by which to reduce the size of the input along the "
        "frequency dimension after convolutional_factor_schedule layers",
    )
    convolutional_channel_factor = param.Integer(
        1,
        bounds=(1, None),
        softbounds=(1, 2),
        doc="The factor by which to increase the size of the channel dimension after "
        "convolutional_factor_schedule layers",
    )
    convolutional_factor_schedule = param.Integer(
        2,
        bounds=(1, None),
        doc="The number of convolutional layers after the first layer before "
        "we modify the size of the time and frequency dimensions",
    )
    convolutional_nonlinearity = param.ObjectSelector(
        torch.nn.functional.relu,
        objects={
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        },
        doc="The pointwise convolutional_nonlinearity between convolutional layers",
    )
    recurrent_size = param.Integer(
        128,
        bounds=(1, None),
        softbounds=(64, 1024),
        doc="The size of each recurrent layer",
    )
    recurrent_layers = param.Integer(
        2,
        bounds=(0, None),
        softbounds=(1, 10),
        doc="The number of recurrent layers in the recurrent part of the network",
    )
    recurrent_type = param.ObjectSelector(
        torch.nn.LSTM,
        objects={"LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU, "RNN": torch.nn.RNN},
        doc="The type of recurrent cell in the recurrent part of the network ",
    )
    recurrent_bidirectional = param.Boolean(
        True, doc="Whether the recurrent layers are bidirectional"
    )
    seed = param.Integer(
        None,
        doc="Seed used for weight initialization. If unset, does not change "
        "the torch stream",
    )

    @classmethod
    def get_tunable(cls):
        return {
            "convolutional_channel_factor",
            "convolutional_factor_schedule",
            "convolutional_freq_factor",
            "convolutional_initial_channels",
            "convolutional_kernel_freq",
            "convolutional_kernel_time",
            "convolutional_layers",
            "convolutional_mellin",
            "convolutional_nonlinearity",
            "convolutional_time_factor",
            "recurrent_bidirectional",
            "recurrent_layers",
            "recurrent_size",
            "recurrent_type",
            "use_lift",
            "use_log_compression",
            "window_size",
            "window_stride",
        }

    @classmethod
    def suggest_params(cls, trial, base=None, only=None, prefix=""):
        if only is None:
            only = cls.get_tunable()
        only = set(only)  # copy to modify
        params = cls() if base is None else base
        pdict = params.param.params()
        softbound_scale_factor = 1
        if "raw" in trial.user_attrs or (
            hasattr(trial, "study") and "raw" in trial.study.user_attrs
        ):
            params.convolutional_kernel_freq = params.convolutional_freq_factor = 1
            for tunable in {"convolutional_kernel_freq", "convolutional_freq_factor"}:
                if tunable in only:
                    warnings.warn(
                        f"Removing '{tunable}' from list of tunable hyperparameters "
                        "because input is raw"
                    )
                    only.remove(tunable)
            # defaults are based on 10ms shift; with 16000 samps/sec, the same length
            # is 100 times
            softbound_scale_factor = 100

        def check_and_set(
            name, use_scale_factor=False, use_log=False, low=None, high=None
        ):
            if name not in only:
                return
            entry = pdict[name]
            deft = getattr(params, name)
            if isinstance(entry, param.Number):
                if low is None:
                    low = entry.get_soft_bounds()[0]
                    assert low is not None
                if high is None:
                    high = entry.get_soft_bounds()[1]
                if use_scale_factor:
                    low *= softbound_scale_factor
                    high *= softbound_scale_factor
                if isinstance(deft, int):
                    val = trial.suggest_int(prefix + name, low, high, log=use_log)
                else:
                    val = trial.suggest_float(prefix + name, low, high, log=use_log)
            elif isinstance(entry, param.Boolean):
                val = trial.suggest_categorical(prefix + name, (True, False))
            elif isinstance(entry, param.ObjectSelector):
                range_ = entry.get_range()
                key = trial.suggest_categorical(prefix + name, tuple(range_))
                val = range_[key]
            else:
                assert False
            setattr(params, name, val)

        check_and_set("window_size", True)
        check_and_set("window_stride", True)
        check_and_set("use_log_compression")
        check_and_set("use_lift")
        check_and_set("convolutional_layers")
        check_and_set("recurrent_layers")
        if params.convolutional_layers:
            check_and_set("convolutional_mellin")
            check_and_set("convolutional_kernel_time")
            check_and_set("convolutional_kernel_freq")
            check_and_set("convolutional_initial_channels", False, True)
            check_and_set(
                "convolutional_factor_schedule", high=params.convolutional_layers + 1
            )
            check_and_set("convolutional_nonlinearity")
            if params.convolutional_factor_schedule <= params.convolutional_layers:
                check_and_set("convolutional_time_factor", False, True)
                check_and_set("convolutional_freq_factor", False, True)
                check_and_set("convolutional_channel_factor", False, True)
        if params.recurrent_layers:
            check_and_set("recurrent_size", False, True)
            check_and_set("recurrent_bidirectional")
            check_and_set("recurrent_type")

        return params


class AcousticModel(torch.nn.Module):
    """The acoustic model

    The acoustic model has a call signature

        model(x, lens[, dropout_prob=float][, convolutional_dropout_is_2d=bool])

    `x` is of shape ``(T, N, freq_dim)`` representing the batched audio signals and
    `lens` of shape ``(N,)`` of the lengths of each audio sequence in the batch. Returns
    `y` of shape ``(T', N, target_dim)`` and ``lens_`` of shape ``N``, where

        lens_[n] = (lens[n] - 1) // params.window_stride + 1

    Represents the new lengths of audio sequences based on the number of windows that
    were extracted. `dropout_prob` is

    Parameters
    ----------
    freq_dim : int
        The number of coefficients per element in the audio sequence
    target_dim : int
        The number of types in the output vocabulary (including blanks)
    params : AcousticModelParams
    """

    def __init__(self, freq_dim: int, target_dim: int, params: AcousticModelParams):
        super(AcousticModel, self).__init__()
        self.params = params
        self.freq_dim = freq_dim
        self.target_dim = target_dim

        if params.use_log_compression:
            self.log_compression = LogCompression()
        else:
            self.add_module("log_compression", None)

        if params.use_lift:
            self.lift = DilationLift(2)
        else:
            self.add_module("lift", None)

        self.convs = torch.nn.ModuleList([])

        self.raw = freq_dim == 1

        x = params.window_size
        kx = params.convolutional_kernel_time
        ky = 1 if self.raw else params.convolutional_kernel_freq
        ci = 1
        y = freq_dim
        co = params.convolutional_initial_channels
        up_c = params.convolutional_channel_factor
        decim_x = params.convolutional_time_factor
        on_sy = 1 if self.raw else params.convolutional_freq_factor
        px, py = (kx - 1) // 2, (ky - 1) // 2
        off_s = dy = off_dx = 1
        if params.convolutional_mellin:
            on_sx = 1
            on_dx = (decim_x - 1) * kx + 1
        else:
            on_sx = decim_x
            on_dx = 1
        for layer_no in range(1, params.convolutional_layers + 1):
            if (
                layer_no % params.convolutional_factor_schedule
            ):  # "off:" not adjusting output size
                dx, sx, sy, Dx, Dy = off_dx, off_s, off_s, 1, 1
            else:  # "on:" adjusting output size
                dx, sx, sy, co, Dx, Dy = on_dx, on_sx, on_sy, up_c * co, decim_x, on_sy
            if params.convolutional_mellin:
                new_x = mcorr_valid_size(kx, x, sx, dx, px)
                exp_x = max(x // Dx, 1)
                rx = exp_x - new_x
                if self.raw:
                    self.convs.append(MCorr1d(ci, co, kx, sx, dx, px, rx))
                else:
                    new_y = lcorr_valid_size(ky, y, sy, dy, py)
                    exp_y = max(y // Dy, 1)
                    ry = exp_y - new_y
                    self.convs.append(
                        MCorrLCorr(
                            ci,
                            co,
                            (kx, ky),
                            s=(sx, sy),
                            d=(dx, dy),
                            p=(px, py),
                            r=(rx, ry),
                        )
                    )
                    y = new_y + ry
                x = new_x + rx
            else:
                px_ = px if (x + 2 * px - kx) // sx + 1 > 0 else px + 1
                if self.raw:
                    self.convs.append(torch.nn.Conv1d(ci, co, kx, sx, px_))
                else:
                    py_ = py if (y + 2 * py - ky) // sy + 1 > 0 else py + 1
                    self.convs.append(
                        torch.nn.Conv2d(ci, co, (kx, ky), (sx, sy), (px_, py_))
                    )
                    y = (y + 2 * py_ - ky) // sy + 1
                x = (x + 2 * px_ - kx) // sx + 1
            ci = co
        assert x > 0 and y > 0
        # sum the window and flatten the channel and frequency dimension
        prev_size = ci * y
        if params.recurrent_layers:
            self.rnn = params.recurrent_type(
                input_size=prev_size,
                hidden_size=params.recurrent_size,
                num_layers=params.recurrent_layers,
                dropout=0.0,
                bidirectional=params.recurrent_bidirectional,
                batch_first=False,
            )
            prev_size = params.recurrent_size * (
                2 if params.recurrent_bidirectional else 1
            )
        else:
            self.rnn = None
        self.out = torch.nn.Linear(prev_size, target_dim)
        if params.seed is not None:
            self.reset_parameters()

    def check_input(self, x: torch.Tensor, lens: torch.Tensor):
        if x.dim() != 3:
            raise RuntimeError("x must be 3-dimensional")
        if lens.dim() != 1:
            raise RuntimeError("lens must be 1-dimensional")
        if x.size(2) != self.freq_dim:
            raise RuntimeError(
                f"Final dimension size of x {x.size(2)} != {self.freq_dim}"
            )
        if x.size(0) != lens.size(0):
            str_ = f"Batch size of x ({x.size(1)}) != lens size ({lens.size(0)})"
            if x.size(1) == lens.size(0):
                str_ += ". (batch dimension of x should be 0, not 1)"
            raise RuntimeError(str_)
        if ((lens <= 0) | (lens > x.size(1))).any():
            raise RuntimeError(f"lens must all be between [1, {lens.size(2)}]")

    def forward(
        self,
        x: torch.Tensor,
        lens: torch.Tensor,
        dropout_prob: float = 0.0,
        dropout_is_2d: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.check_input(x, lens)
        # zero the input past lens to ensure no weirdness
        len_mask = (
            torch.arange(x.size(1), device=lens.device) >= lens.unsqueeze(1)
        ).unsqueeze(2)
        x = x.masked_fill(len_mask, 0)
        del len_mask
        # we always pad by one less than the window length. This will have the effect
        # of taking the final window if it's incomplete
        # N.B. this has to be zero-padding b/c shorter sequences will be zero-padded.
        if self.params.window_size > 1:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.params.window_size - 1))
        x = x.unfold(1, self.params.window_size, self.params.window_stride).transpose(
            2, 3
        )  # (N, T', w, F)
        # all lengths are positive, so trunc = floor. Assuming trunc is more efficient
        lens_ = (
            torch.div(lens - 1, self.params.window_stride, rounding_mode="trunc") + 1
        )

        # fuse the N and T' dimension together for now. No sense in performing
        # convolutions on windows of entirely padding
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lens_.cpu(), batch_first=True, enforce_sorted=False
        )

        x, bs, si, ui = x.data, x.batch_sizes, x.sorted_indices, x.unsorted_indices

        # perform optional log compression, lift
        if self.log_compression is not None:
            x = self.log_compression(x)
        if self.lift is not None:
            x = self.lift(x)

        if self.raw:
            x = x.flatten(1)  # (N', w)
        x = x.unsqueeze(1)  # (N', 1, w[, F])

        # convolutions
        for conv in self.convs:
            x = self.params.convolutional_nonlinearity(conv(x))
            if dropout_is_2d:
                if self.raw:
                    x = torch.nn.functional.dropout2d(
                        x.unsqueeze(-1), dropout_prob, self.training
                    ).squeeze(-1)
                else:
                    x = torch.nn.functional.dropout2d(x, dropout_prob, self.training)
            else:
                x = torch.nn.functional.dropout(x, dropout_prob, self.training)

        # sum out the window dimension and reshape for rnn input
        x = x.sum(2).view(x.size(0), -1)  # (N', co * F')

        # rnns
        if self.rnn is not None:
            x = self.rnn(
                torch.nn.utils.rnn.PackedSequence(
                    x, batch_sizes=bs, sorted_indices=si, unsorted_indices=ui
                )
            )[0].data

        # feed-forward
        x = self.out(x)  # (N', target_dim)

        # unpack and return
        x = torch.nn.utils.rnn.PackedSequence(
            x, batch_sizes=bs, sorted_indices=si, unsorted_indices=ui
        )
        return torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=False)[0], lens_

    def reset_parameters(self):
        if self.params.seed is not None:
            torch.manual_seed(self.params.seed)
        for layer in chain(
            (self.lift, self.log_compression, self.out, self.rnn), self.convs
        ):
            if layer is not None:
                layer.reset_parameters()
