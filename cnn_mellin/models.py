'''Acoustic models'''

import param

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


class AcousticModelParams(param.Parameterized):
    filt_dim = param.Integer(
        None, bounds=(1, None),
        doc='The number of filters per frame of input'
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
    mellin = param.Boolean(
        False, bounds=(1, None),
        doc='Whether the convolutional layers are mellin-linear (versus just '
        'linear)'
    )
    init_num_channels = param.Integer(
        256, bounds=(1, None),
        doc='How many channels to introduce into the first convolutional layer'
    )
    channels_factor = param.Integer(
        1, bounds=(1, None),
        doc='The factor by which to multiply the number of channels between '
        'convolutional layers'
    )
    decimation_factor = param.Integer(
        2, bounds=(1, None),
        doc='The factor by which to reduce the size of the input along the '
        'time dimension between convolutional layers'
    )
    hidden_size = param.Integer(
        1024, bounds=(1, None),
        doc='The size of hidden states (output of fully-connected layers), '
        'except the final one'
    )
    kernel_width = param.Integer(
        3, bounds=(1, None),
        doc='The length of convolutional kernels in time'
    )
    kernel_height = param.Integer(
        3, bounds=(1, None),
        doc='The length of convolutional kernels in log-frequency'
    )
    nonlinearity = param.ObjectSelector(
        'relu', objects=['relu', 'sigmoid', 'tanh'],
        doc='The pointwise nonlinearity between non-final layers'
    )
    mconv_decimation_strategy = param.ObjectSelector(
        'pad-to-dec-width-floor',
        objects=[
            'pad-then-dec', 'pad-to-dec-width-floor', 'pad-to-dec-width-ceil'],
        doc='How to decimate with a Mellin convolution (if applicable). '
        '"pad-then-dec" = set p to prior width, then set d high to decimate. '
        '"pad-to-dec-width-floor" = set p to decimate, flooring, then pad the '
        'missing width using r. "pad-to-dec-width-ceil" = set p to decimate, '
        'taking the ceiling, then cut the excess samples with d'
    )
    seed = param.Integer(
        None,
        doc='Seed used for weight initialization. If unset, does not change '
        'the torch stream'
    )
