"""Acoustic models"""

import warnings

from itertools import chain
from typing import Tuple

import param
import torch

from pydrobert.mellin.torch import MellinLinearCorrelation

from cnn_mellin.layers import DilationLift

__all__ = [
    "AcousticModelParams",
    "AcousticModel",
]


class AcousticModelParams(param.Parameterized):
    window_size = param.Integer(
        10,
        bounds=(1, None),
        softbounds=(2, 20),
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
        softbounds=(1, 10),
        doc="The width of convolutional kernels along the frequency dimension",
    )
    initial_channels = param.Integer(
        64,
        bounds=(1, None),
        softbounds=(1, 256),
        doc="The number of channels in the initial convolutional layer",
    )
    convolutional_layers = param.Integer(
        5,
        bounds=(0, None),
        softbounds=(0, 20),
        doc="The number of layers in the convolutional part of the network",
    )
    recurrent_size = param.Integer(
        128,
        bounds=(1, None),
        softbounds=(64, 512),
        doc="The size of each recurrent layer",
    )
    recurrent_layers = param.Integer(
        2,
        bounds=(0, None),
        softbounds=(0, 20),
        doc="The number of recurrent layers in the recurrent part of the network",
    )
    recurrent_type = param.ObjectSelector(
        torch.nn.LSTM,
        objects={"LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU, "RNN": torch.nn.RNN},
        doc="The type of recurrent cell in the recurrent part of the network ",
    )
    bidirectional = param.Boolean(
        True, doc="Whether the recurrent layers are bidirectional"
    )
    mellin = param.Boolean(
        False,
        doc="Whether the convolutional layers are mellin-linear (versus just linear)",
    )
    time_factor = param.Integer(
        2,
        bounds=(1, None),
        softbounds=(1, 5),
        doc="The factor by which to reduce the size of the input along the "
        "time dimension between convolutional layers",
    )
    freq_factor = param.Integer(
        1,
        bounds=(1, None),
        softbounds=(1, 5),
        doc="The factor by which to reduce the size of the input along the "
        "frequency dimension after factor_sched layers",
    )
    channel_factor = param.Integer(
        1,
        bounds=(1, None),
        softbounds=(1, 5),
        doc="The factor by which to increase the size of the channel dimension after "
        "factor_sched layers",
    )
    factor_sched = param.Integer(
        2,
        bounds=(1, None),
        doc="The number of convolutional layers after the first layer before "
        "we modify the size of the time and frequency dimensions",
    )
    convolutional_nonlinearity = param.ObjectSelector(
        torch.nn.functional.relu,
        objects={
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.nn.functional.sigmoid,
            "tanh": torch.nn.functional.tanh,
        },
        doc="The pointwise convolutional_nonlinearity between convolutional layers",
    )
    seed = param.Integer(
        None,
        doc="Seed used for weight initialization. If unset, does not change "
        "the torch stream",
    )
    convolutional_dropout_2d = param.Boolean(
        True, doc="If true, zero out channels instead of individual coefficients"
    )

    @classmethod
    def get_tunable(cls):
        return {
            "window_size",
            "window_stride",
            "convolutional_kernel_time",
            "convolutional_kernel_freq",
            "initial_channels",
            "convolutional_layers",
            "recurrent_size",
            "recurrent_layers",
            "recurrent_type",
            "bidirectional",
            "mellin",
            "time_factor",
            "freq_factor",
            "channel_factor",
            "factor_sched",
            "convolutional_nonlinearity",
            "convolutional_dropout_2d",
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
            params.convolutional_kernel_freq = params.freq_factor = 1
            for tunable in {"convolutional_kernel_freq", "freq_factor"}:
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
        check_and_set("convolutional_layers", False)
        check_and_set("recurrent_layers", False)
        if params.convolutional_layers:
            check_and_set("mellin")
            check_and_set("convolutional_kernel_time", True)
            check_and_set("convolutional_kernel_freq", True)
            check_and_set("initial_channels", False, True)
            check_and_set("factor_sched", high=params.convolutional_layers + 1)
            check_and_set("convolutional_dropout_2d")
            check_and_set("convolutional_nonlinearity")
            if params.factor_sched <= params.convolutional_layers:
                check_and_set("time_factor", False, True)
                check_and_set("freq_factor", False, True)
                check_and_set("channel_factor", False, True)
        if params.recurrent_layers:
            check_and_set("recurrent_size", False, True)
            check_and_set("bidirectional")
            check_and_set("recurrent_type")

        return params


class AcousticModel(torch.nn.Module):
    """The acoustic model

    Expects input `x` of shape ``(T, N, audio_dim)`` representing the batched audio
    signals and `lens` of shape ``(N,)`` of the lengths of each audio sequence in the
    batch. Returns `y` of shape ``(T', N, target_dim)`` and ``lens_`` of shape ``N``,
    where

        lens_[n] = (lens[n] - 1) // params.window_stride + 1

    Represents the new lengths of audio sequences based on the number of windows that
    were extracted.

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
        self.lift = DilationLift(2)
        self.params = params
        self.freq_dim = freq_dim
        self.target_dim = target_dim

        self.convs = torch.nn.ModuleList([])
        self.drops = torch.nn.ModuleList([])
        self._drop_p = 0.0

        if params.convolutional_dropout_2d:
            Dropout = torch.nn.Dropout2d
        else:
            Dropout = torch.nn.Dropout

        kx = params.convolutional_kernel_time
        ky = params.convolutional_kernel_freq
        ci = 1
        y = freq_dim
        co = params.initial_channels
        up_c = params.channel_factor
        decim_x = params.time_factor
        on_sy = params.freq_factor
        py = (ky - 1) // 2
        off_s = dy = off_dx = 1
        if params.mellin:
            # the equation governing the size of the output on a mellin dimension is
            # x' = ceil((p + 1)(x' + u - 1)/(k + d - 1)) - (s - 1) + r
            # to reproduce the same size output, set p = k - 1, d = 1, s = 1, r = 0,
            # u = 1.
            # There are a few ways to decimate the input to an appropriate width, but
            # earlier experimentation preferred higher values of p to do so. Setting
            # p = k - 1, d = (decim_x - 1) * k + 1, u = 1, s = 1, and r = 0,
            # x' = ceil((k * x) / (kx + (decim_x - 1) * k + 1 - 1))
            #    = ceil((k * x) / (decim_x * k)) = ceil(x / decim_x)
            on_sx = 1
            px = kx - 1
            on_dx = (decim_x - 1) * kx + 1
        else:
            # the equation governing the size of the output on a linear dimension is
            # x' = (x + 2p - d * (k - 1) - 1) // s + 1
            # If k % 2 = 1, (k - 1) // 2 * 2 == k - 1. If k % 2 == 0,
            # (k - 1) // 2 * 2 == k - 2, which will shrink our input by 1 on the
            # off-factors and by 0 or 1 on the on-factors, depending on whether
            # the decimation factor is > 1.
            on_sx = decim_x
            px = (kx - 1) // 2
            on_dx = 1
        for layer_no in range(1, params.convolutional_layers + 1):
            if layer_no % params.factor_sched:  # "off:" not adjusting output size
                dx, sx, sy = off_dx, off_s, off_s
            else:  # "on:" adjusting output size
                dx, sx, sy, co = on_dx, on_sx, on_sy, up_c * co
            if params.mellin:
                self.convs.append(
                    MellinLinearCorrelation(
                        ci,
                        co,
                        (kx, ky),
                        s=(sx, sy),
                        d=(dx, dy),
                        p=(px, py),
                        r=(0, py // sy),
                    )
                )
                y = (y + py - ky) // sy + py // sy + 1
            else:
                self.convs.append(
                    torch.nn.Conv2d(ci, co, (kx, ky), padding=(px, py), stride=(sx, sy))
                )
                y = (y + 2 * py - ky) // sy + 1
            ci = co
            self.drops.append(Dropout(p=0.0))
        # flatten the channel dimension (ci) into the frequency dimension (y)
        prev_size = ci * y
        if params.recurrent_layers:
            self.rnn = params.recurrent_type(
                input_size=prev_size,
                hidden_size=params.recurrent_size,
                num_layers=params.recurrent_layers,
                dropout=0.0,
                bidirectional=params.bidirectional,
                batch_first=False,
            )
            prev_size = params.recurrent_size * (2 if params.bidirectional else 1)
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

    def forward(
        self, x: torch.Tensor, lens: torch.Tensor
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
        lens_ = (lens - 1) // self.params.window_stride + 1
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lens_, batch_first=True, enforce_sorted=False
        )
        # fuse the N and T' dimension together for now. No sense in performing
        # convolutions on windows of entirely padding
        x, bs, si, ui = x.data, x.batch_sizes, x.sorted_indices, x.unsorted_indices
        x = self.lift(x).unsqueeze(1)  # (N', 1, w, F)
        for conv, drop in zip(self.convs, self.drops):
            x = drop(self.params.convolutional_nonlinearity(conv(x)))  # (N', co, w',F')
        x = x.sum(2).view(x.size(0), -1)  # (N', co * F')
        if self.rnn is not None:
            x = self.rnn(
                torch.nn.utils.rnn.PackedSequence(
                    x, batch_sizes=bs, sorted_indices=si, unsorted_indices=ui
                )
            )[0].data
        x = self.out(x)  # (N', target_dim)
        x = torch.nn.utils.rnn.PackedSequence(
            x, batch_sizes=bs, sorted_indices=si, unsorted_indices=ui
        )
        return torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=False)

    def reset_parameters(self):
        if self.params.seed is not None:
            torch.manual_seed(self.params.seed)
        for layer in chain(
            (self.lift, self.out, self.rnn)
            if self.rnn is not None
            else (self.lift, self.out),
            self.convs,
        ):
            layer.reset_parameters()

    @property
    def dropout(self):
        # there's a possibility of no dropout layers at all, which is why
        # we keep track of it as a simple variable
        return self._drop_p

    @dropout.setter
    def dropout(self, p):
        for layer in self.drops:
            layer.p = p
        self.rnn.dropout = p
        self._drop_p = p
