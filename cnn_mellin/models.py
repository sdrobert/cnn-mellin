'''Acoustic models'''

from itertools import chain

import param
import torch

from cnn_mellin import TupleList

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"
__all__ = [
    'AcousticModelParams',
    'AcousticModel',
    'estimate_total_size_bytes',
]


class AcousticModelParams(param.Parameterized):
    freq_dim = param.Integer(
        None, bounds=(1, None),
        doc='The number of log-frequency filters per frame of input'
    )
    target_dim = param.Integer(
        None, bounds=(1, None),
        doc='The target (output) dimension per frame'
    )
    kernel_sizes = TupleList(
        [(3, 3, 128)], class_=int,
        doc='A list of kernel dimensions for convolutional layers. The kernel '
        'dimensions should be (T, F, C_out), where T is the dimension in time,'
        'F the dimension in frequency, and C_out is the number of output '
        'channels. '
    )
    hidden_sizes = param.List(
        [512, 512], class_=int,
        doc='Intermediate fully-connected layer sizes. Note that one '
        'additional fully-connected layer will be incuded after these layers '
        'to fit the target_dim'
    )
    flatten_style = param.ObjectSelector(
        'keep_chans', objects=['keep_chans', 'keep_both'],
        doc='How to remove a dimension of the input after convoultions. '
        '"keep_chans" keeps the channel dimension and sums out the filters. '
        '"keep_both" keeps all coefficients by flattening filters and '
        'channels. To keep only filters, set the number of channels in the '
        'last listed conv_kernels element to 1'
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
        'we modify the size of the time and frequency dimensions'
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
    '''The acoustic model

    Parameters
    ----------
    params : AcousticModelParams
    window : int
        The total size of the window in time (context_left + context_right + 1)
    '''

    def __init__(self, params, window):
        super(AcousticModel, self).__init__()
        self.params = params
        self.window = window
        self.convs = torch.nn.ModuleList([])
        self.fcs = torch.nn.ModuleList([])
        self.nonlins = torch.nn.ModuleList([])
        self.drops = torch.nn.ModuleList([])
        self._drop_p = 0.0
        if params.target_dim is None:
            raise ValueError('target_dim must be set!')
        if params.nonlinearity == 'relu':
            Nonlin = torch.nn.ReLU
        elif params.nonlinearity == 'sigmoid':
            Nonlin = torch.nn.Sigmoid
        else:
            Nonlin = torch.nn.Tanh
        if params.dropout2d_on_conv:
            Dropout = torch.nn.Dropout2d
        else:
            Dropout = torch.nn.Dropout
        conv_config = _get_conv_config(params, window)
        for layer_config in conv_config['layers']:
            self.convs.append(conv_config['class'](
                layer_config['in_chan'],
                layer_config['out_chan'],
                (layer_config['kw'], layer_config['kh']),
                **layer_config['kwargs']
            ))
            self.nonlins.append(Nonlin())
            self.drops.append(Dropout(p=0.0))
        prev_size = conv_config['out_size']
        for hidden_size in params.hidden_sizes:
            self.fcs.append(torch.nn.Linear(prev_size, hidden_size))
            self.nonlins.append(Nonlin())
            self.drops.append(torch.nn.Dropout(p=0.0))
            prev_size = hidden_size
        self.fcs.append(torch.nn.Linear(prev_size, params.target_dim))
        self.reset_parameters()

    def forward(self, x):
        num_convs = len(self.convs)
        if num_convs:
            x = x.unsqueeze(1)
            for conv, nonlin, drop in zip(
                    self.convs, self.nonlins, self.drops):
                x = drop(nonlin(conv(x)))
            if self.params.flatten_style == 'keep_chans':
                x = x.sum(-1)
        x = x.view(-1, self.fcs[0].in_features)
        for layer_idx, fc in enumerate(self.fcs[:-1]):
            x = fc(x)
            x = self.nonlins[num_convs + layer_idx](x)
            x = self.drops[num_convs + layer_idx](x)
        x = self.fcs[-1](x)
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


def estimate_total_size_bytes(params, num_windows, window, bytes_per_scalar=4):
    '''Estimate the size of model + forward + backward in bytes

    Parameters
    ----------
    params : AcousticModelParams
    num_windows : int
        The number of windows in a given batch
    window : int
        The total size of the context  window in time
        ``(context_left + context_right + 1)``

    Returns
    -------
    total_bytes : int

    Warning
    -------
    This function is only meant to provide an estimate. It is untested and
    probably not very accurate
    '''
    # FIXME(sdrobert):
    # https://discuss.pytorch.org/t/rethinking-memory-estimates-for-training/36080
    total_output_scalars = 0
    total_weight_scalars = 0
    conv_config = _get_conv_config(params, window)
    total_input_scalars = num_windows * window * params.freq_dim
    for layer_config in conv_config['layers']:
        total_weight_scalars += (
            layer_config['in_chan'] * layer_config['out_chan'] *
            layer_config['kw'] * layer_config['kh']
        ) + layer_config['out_chan']
        total_output_scalars += (
            layer_config['out_chan'] *
            layer_config['out_w'] * layer_config['out_h']
        )
    prev_size = conv_config['out_size']
    for cur_size in params.hidden_sizes + [params.target_dim]:
        total_weight_scalars += prev_size * cur_size + cur_size
        total_output_scalars += cur_size
        prev_size = cur_size
    total_output_scalars *= 2 * num_windows  # backward pass and batch
    total_scalars = (
        total_weight_scalars + total_output_scalars + total_input_scalars)
    total_bytes = total_scalars * bytes_per_scalar
    return total_bytes


def _get_conv_config(params, window):
    config = dict()
    config['layers'] = layers = []
    if not len(params.kernel_sizes):
        config['out_size'] = window * params.freq_dim
        return config
    decim_w = params.time_factor
    decim_h = params.freq_factor

    def kw(layer_idx):
        return params.kernel_sizes[layer_idx][0]

    def kh(layer_idx):
        return params.kernel_sizes[layer_idx][1]

    def dh(layer_idx):
        return 1

    def ph(layer_idx):
        return kh(layer_idx) // 2

    def ci(layer_idx):
        return params.kernel_sizes[layer_idx - 1][2] if layer_idx else 1

    def co(layer_idx):
        return params.kernel_sizes[layer_idx][2]
    if params.mellin:
        from pydrobert.mellin.torch import MellinLinearCorrelation
        config['class'] = MellinLinearCorrelation

        def off_pw(layer_idx):
            return kw(layer_idx) - 1

        def off_dw(layer_idx):
            return 1
        if params.mconv_decimation_strategy == 'pad-to-dec-time-ceil':
            def on_pw(layer_idx):
                return (kw(layer_idx) - 1) // decim_w

            def on_dw(layer_idx):
                return (on_pw(layer_idx) + 1) * decim_w - kw(layer_idx) + 1
        elif params.mconv_decimation_strategy == 'pad-then-dec':
            def on_pw(layer_idx):
                return kw(layer_idx) - 1

            def on_dw(layer_idx):
                return (decim_w - 1) * kw(layer_idx) + 1
        else:
            def on_pw(layer_idx):
                return max((kw(layer_idx) - decim_w) // decim_w, 0)

            def on_dw(layer_idx):
                return max(
                    (on_pw(layer_idx) + 1) * decim_w - kw(layer_idx) + 1, 1)

        def update_size_and_kwargs(on, prev_w, prev_h, layer_idx):
            if on:
                pw, dw = on_pw, on_dw
                cur_w = (prev_w + decim_w - 1) // decim_w
                cur_h = (prev_h + decim_h - 1) // decim_h
            else:
                pw, dw = off_pw, off_dw
                cur_w = prev_w
                cur_h = prev_h
            sw = 1
            rw = (pw(layer_idx) + 1) * prev_w - 1
            rw //= kw(layer_idx) + dw(layer_idx) - 1
            rw = cur_w - rw - 1
            # we don't have rh before calculating sh. Below assumes rh is going
            # to pad however much is necessary to get (kh - 1) in the
            # numerator, hence ph - dh * (kh - 1) => -(dh - 1) * (kh - 1)
            sh = prev_h - (dh(layer_idx) - 1) * (kh(layer_idx) - 1) - 1
            sh //= cur_h
            sh += 1
            rh = prev_h + ph(layer_idx) - dh(layer_idx) * (kh(layer_idx) - 1)
            rh -= 1
            rh //= sh
            rh = cur_h - rh - 1
            return cur_w, cur_h, {
                'p': (pw(layer_idx), ph(layer_idx)),
                's': (sw, sh),
                'd': (dw(layer_idx), dh(layer_idx)),
                'r': (rw, rh),
            }
    else:
        config['class'] = torch.nn.Conv2d

        def pw(layer_idx):
            return kw(layer_idx) // 2

        def dw(layer_idx):
            return 1

        def update_size_and_kwargs(on, prev_w, prev_h, layer_idx):
            if on:
                cur_w = (prev_w + decim_w - 1) // decim_w
                cur_h = (prev_h + decim_h - 1) // decim_h
            else:
                cur_w = prev_w
                cur_h = prev_h
            sh = prev_h + 2 * ph(layer_idx) - 1
            sh -= dh(layer_idx) * (kh(layer_idx) - 1)
            sh //= cur_h
            sh += 1
            cur_h = prev_h + 2 * ph(layer_idx) - 1
            cur_h -= dh(layer_idx) * (kh(layer_idx) - 1)
            cur_h //= sh
            cur_h += 1
            sw = prev_w + 2 * pw(layer_idx) - 1
            sw -= dw(layer_idx) * (kw(layer_idx) - 1)
            sw //= cur_w
            sw += 1
            cur_w = prev_w + 2 * pw(layer_idx) - 1
            cur_w -= dw(layer_idx) * (kw(layer_idx) - 1)
            cur_w //= sw
            cur_w += 1
            return cur_w, cur_h, {
                'padding': (pw(layer_idx), ph(layer_idx)),
                'stride': (sw, sh),
                'dilation': (dw(layer_idx), dh(layer_idx)),
            }
    prev_w = window
    prev_h = params.freq_dim
    if params.factor_sched is None:
        sched = float('inf')
    else:
        sched = params.factor_sched + 1
    for layer_idx in range(len(params.kernel_sizes)):
        sched -= 1
        if not sched:
            sched = params.factor_sched
            on = True
        else:
            on = False
        cur_w, cur_h, kwargs = update_size_and_kwargs(
            on, prev_w, prev_h, layer_idx)
        entry = {
            'in_chan': ci(layer_idx),
            'out_chan': co(layer_idx),
            'kw': kw(layer_idx),
            'kh': kh(layer_idx),
            'in_w': prev_w,
            'in_h': prev_h,
            'out_w': cur_w,
            'out_h': cur_h,
            'kwargs': kwargs
        }
        layers.append(entry)
        prev_w, prev_h = cur_w, cur_h
    if params.flatten_style == 'keep_chans':
        prev_h = co(layer_idx)
    elif params.flatten_style == 'keep_both':
        prev_h *= co(layer_idx)
    config['out_size'] = prev_w * prev_h
    return config
