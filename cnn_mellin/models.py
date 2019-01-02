'''Acoustic models'''

from itertools import chain

import param
import torch

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


class AcousticModelParams(param.Parameterized):
    freq_dim = param.Integer(
        None, bounds=(1, None),
        doc='The number of log-frequency filters per frame of input'
    )
    target_dim = param.Integer(
        None, bounds=(1, None),
        doc='The target (output) dimension per frame'
    )
    num_conv = param.Integer(
        2, bounds=(0, None),
        doc='The number of convolutional layers in the network'
    )
    num_fc = param.Integer(
        3, bounds=(1, None),
        doc='The number of fully connected layers in the network'
    )
    flatten_style = param.ObjectSelector(
        'keep_filts', objects=['keep_filts', 'keep_chans', 'keep_both'],
        doc='How to remove a dimension of the input after convoultions. '
        '"keep_filts" keeps the filter dimension but makes the last '
        'convoultional layer have one channel.'
        '"keep_chans" keeps the channel dimension and sums out the filters. '
        '"keep_both" keeps all coefficients by flattening filters and channels'
    )
    dropout2d_on_conv = param.Boolean(
        False,
        doc='If True, will zero out full channels instead of individual '
        'elements'
    )
    mellin = param.Boolean(
        False,
        doc='Whether the convolutional layers are mellin-linear (versus just '
        'linear)'
    )
    init_num_channels = param.Integer(
        256, bounds=(1, None),
        doc='How many channels to introduce into the first convolutional '
        'layer before factor_sched layers'
    )
    channels_factor = param.Integer(
        None, bounds=(1, None),
        doc='The factor by which to multiply the number of channels after '
        'factor_sched layers. If None, will be calculated as time_factor *'
        'filt_factor'
    )
    time_factor = param.Integer(
        2, bounds=(1, None),
        doc='The factor by which to reduce the size of the input along the '
        'time dimension between convolutional layers'
    )
    freq_factor = param.Integer(
        1, bounds=(1, None),
        doc='The factor by which to reduce the size of the input along the '
        'frequency dimension after factor_sched layers'
    )
    factor_sched = param.Integer(
        None, bounds=(1, None),
        doc='The number of convolutional layers after the first layer before '
        'we modify the size of the time, frequency, and channel dimensions, '
        'then again after that many layers again. For example, factor_sched = '
        '2 implies convolutional input sizes will be modified after the third '
        'layer, fifth layer, and so on.'
    )
    hidden_size = param.Integer(
        1024, bounds=(1, None),
        doc='The size of hidden states (output of fully-connected layers), '
        'except the final one'
    )
    kernel_time = param.Integer(
        3, bounds=(1, None),
        doc='The length of convolutional kernels in time'
    )
    kernel_freq = param.Integer(
        3, bounds=(1, None),
        doc='The length of convolutional kernels in log-frequency'
    )
    nonlinearity = param.ObjectSelector(
        'relu', objects=['relu', 'sigmoid', 'tanh'],
        doc='The pointwise nonlinearity between non-final layers'
    )
    mconv_decimation_strategy = param.ObjectSelector(
        'pad-to-dec-time-floor',
        objects=[
            'pad-then-dec', 'pad-to-dec-time-floor', 'pad-to-dec-time-ceil'],
        doc='How to decimate with a Mellin convolution (if applicable). '
        '"pad-then-dec" = set p to prior length, then set d high to decimate. '
        '"pad-to-dec-time-floor" = set p to decimate, flooring, then pad the '
        'missing length using r. "pad-to-dec-time-ceil" = set p to decimate, '
        'taking the ceiling, then cut the excess samples with d'
    )
    seed = param.Integer(
        None,
        doc='Seed used for weight initialization. If unset, does not change '
        'the torch stream'
    )


class AcousticModel(torch.nn.Module):

    def __init__(self, params, window):
        super(AcousticModel, self).__init__()
        self.params = params
        self.window = window
        self.convs = torch.nn.ModuleList([])
        self.fcs = torch.nn.ModuleList([])
        self.nonlins = torch.nn.ModuleList([])
        self.drops = torch.nn.ModuleList([])
        self._drop_p = 0.0
        prev_w = window
        prev_h = params.freq_dim
        if params.nonlinearity == 'relu':
            Nonlin = torch.nn.ReLU
        elif params.nonlinearity == 'sigmoid':
            Nonlin = torch.nn.Sigmoid
        else:
            Nonlin = torch.nn.Tanh
        if params.num_conv:
            prev_chan = 1
            decim_w = params.time_factor
            decim_h = params.freq_factor
            if params.channels_factor is not None:
                mult_chan = params.channels_factor
            else:
                mult_chan = decim_w * decim_h
            kw = params.kernel_time
            kh = params.kernel_freq
            dh = 1
            ph = kh // 2
            if params.factor_sched is None:
                sched = float('inf')
            else:
                sched = params.factor_sched + 1
            if params.dropout2d_on_conv:
                Dropout = torch.nn.Dropout2D
            else:
                Dropout = torch.nn.Dropout
            if params.mellin:
                from pydrobert.mellin.torch import (
                    MellinLinearCorrelation as Conv)
                off_pw = kw - 1
                off_dw = 1
                if params.mconv_decimation_strategy == 'pad-to-dec-time-ceil':
                    on_pw = (kw - 1) // decim_w
                    on_dw = (on_pw + 1) * decim_w - kw + 1
                elif params.mconv_decimation_strategy == 'pad-then-dec':
                    on_pw = kw - 1
                    on_dw = (decim_w - 1) * kw + 1
                else:
                    on_pw = max((kw - decim_w) // decim_w, 0)
                    on_dw = max((on_pw + 1) * decim_w - kw + 1, 1)

                def _update_size_and_kwargs(on):
                    if on:
                        pw, dw = on_pw, on_dw
                        cur_w = (prev_w + decim_w - 1) // decim_w
                        cur_h = (prev_h + decim_h - 1) // decim_h
                    else:
                        pw, dw = off_pw, off_dw
                        cur_w = prev_w
                        cur_h = prev_h
                    sw = 1
                    rw = cur_w - ((pw + 1) * prev_w - 1) // (kw + dw - 1) - 1
                    sh = (prev_h + - (dh - 1) * (kh - 1) - 1) // cur_h + 1
                    rh = cur_h - (prev_h + ph - dh * (kh - 1) - 1) // sh - 1
                    return cur_w, cur_h, {
                        'p': (pw, ph),
                        's': (sw, sh),
                        'd': (dw, dh),
                        'r': (rw, rh),
                    }
            else:
                Conv = torch.nn.Conv2d
                pw = kw // 2
                dw = 1

                def _update_size_and_kwargs(on):
                    if on:
                        cur_w = (prev_w + decim_w - 1) // decim_w
                        cur_h = (prev_h + decim_h - 1) // decim_h
                    else:
                        cur_w = prev_w
                        cur_h = prev_h
                    sh = (prev_h + 2 * ph - dh * (kh - 1) - 1) // cur_h + 1
                    cur_h = (prev_h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
                    sw = (prev_w + 2 * pw - dw * (kw - 1) - 1) // cur_w + 1
                    cur_w = (prev_w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
                    return cur_w, cur_h, {
                        'padding': (pw, ph),
                        'stride': (sw, sh),
                        'dilation': (dw, dh),
                    }
            for layer_idx in range(params.num_conv):
                sched -= 1
                if not sched:
                    sched = params.factor_sched
                    on = True
                    cur_chan = mult_chan * prev_chan
                else:
                    on = False
                    cur_chan = prev_chan
                if not layer_idx:
                    cur_chan = params.init_num_channels
                if (
                        layer_idx == params.num_conv - 1 and
                        params.flatten_style == 'keep_filts'):
                    cur_chan = 1
                cur_w, cur_h, kwargs = _update_size_and_kwargs(on)
                self.convs.append(Conv(
                    prev_chan, cur_chan, (kw, kh),
                    **kwargs
                ))
                self.nonlins.append(Nonlin())
                self.drops.append(Dropout(p=0.0))
                prev_chan, prev_w, prev_h = cur_chan, cur_w, cur_h
            if params.flatten_style == 'keep_chans':
                prev_h = prev_chan
            elif params.flatten_style == 'keep_both':
                prev_h *= prev_chan
        prev_size = prev_w * prev_h
        for layer_idx in range(params.num_fc):
            if layer_idx == params.num_fc - 1:
                self.fcs.append(torch.nn.Linear(prev_size, params.target_dim))
            else:
                self.fcs.append(torch.nn.Linear(prev_size, params.hidden_size))
                self.nonlins.append(Nonlin())
                self.drops.append(torch.nn.Dropout(p=0.0))
                prev_size = params.hidden_size
        self.reset_parameters()

    def forward(self, x):
        if self.params.num_conv:
            x = x.unsqueeze(1)
            for conv, nonlin, drop in zip(
                    self.convs, self.nonlins, self.drops):
                x = drop(nonlin(conv(x)))
            if self.params.flatten_style == 'keep_chans':
                x = x.sum(-1)
        x = x.view(-1, self.fcs[0].in_features)
        for layer_idx, fc in enumerate(self.fcs):
            x = fc(x)
            if layer_idx < self.params.num_fc - 1:
                x = self.nonlins[self.params.num_conv + layer_idx](x)
                x = self.drops[self.params.num_conv + layer_idx](x)
        return x

    def reset_parameters(self):
        if self.params.seed is not None:
            torch.manual_seed(self.params.seed)
        for layer in chain(self.convs, self.fcs):
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
        self._drop_p = p
