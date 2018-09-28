'''Parameters of various cnn_mellin operations'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import param

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


class SpectDataParams(param.Parameterized):
    # context windows are more model parameters than data parameters, but
    # we're going to extract them as part of the data loading process, which
    # is easily parallelized by the DataLoader
    context_left = param.Integer(
        4, bounds=(0, None), softbounds=(3, 8),
        doc='How many frames to the left of (before) the current frame are '
        'included when determining the class of the current frame'
    )
    context_right = param.Integer(
        4, bounds=(0, None), softbounds=(3, 8),
        doc='How many frames to the right of (after) the current frame are '
        'included when determining the class of the current frame'
    )


class DataSetParams(param.Parameterized):
    batch_size = param.Integer(
        10, bounds=(1, None),
        doc='Number of elements in a batch. For training, this is the total '
        'number of context windows in a batch. For validation/testing, this '
        'is the number of utterances to process at once'
    )
    seed = param.Integer(
        None,
        doc='The seed used to shuffle data. The seed is incremented at every '
        'epoch'
    )
    drop_last = param.Boolean(
        False,
        doc='Whether to drop the last batch if it does reach batch_size'
    )


class SpectDataSetParams(SpectDataParams, DataSetParams):
    pass


class AcousticModelParams(param.Parameterized):
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


class TrainingParams(param.Parameterized):
    num_epochs = param.Integer(
        None, bounds=(1, None),
        doc='Total number of epochs to run for. If unspecified, runs '
        'until the early stopping criterion (or infinitely if disabled) '
    )
    optimizer = param.ObjectSelector(
        'adam', objects=['adam', 'rmsprop', 'sgd'],
        doc='The name of the optimizer to train with'
    )
    log10_learning_rate = param.Number(
        None, softbounds=(-10, -2),
        doc='Optimizer log-learning rate. If unspecified, uses the '
        'built-in rate'
    )
    log10_weight_decay = param.Number(
        None, softbounds=(-10, -2),
        doc='The log of parameter\'s L2 weight penalty. If unspecified, '
        'no penalty is used'
    )
    early_stopping_threshold = param.Number(
        0., bounds=(0, None), softbounds=(0, 1.),
        doc='Minimum improvement in xent from the last best that resets the '
        'early stopping clock. If zero, early stopping will not be performed'
    )
    early_stopping_patience = param.Integer(
        1, bounds=(1, None), softbounds=(1, 30),
        doc='Number of epochs where, if the classifier has failed to '
        'improve it\'s error, training is halted'
    )
    early_stopping_burnin = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs before the early stopping criterion kicks in'
    )
    reduce_lr_threshold = param.Number(
        0., bounds=(0, None), softbounds=(0, 1.),
        doc='Minimum improvement in xent from the last best that resets the '
        'clock for reducing the learning rate. If zero, the learning rate '
        'will not be reduced during training. Se'
    )
    reduce_lr_factor = param.Number(
        None, bounds=(0, 1), softbounds=(0, .5),
        inclusive_bounds=(False, False),
        doc='Factor by which to multiply the learning rate if there has '
        'been no improvement in the error after "reduce_lr_patience" '
        'epochs. If unset, uses the pytorch defaults'
    )
    reduce_lr_patience = param.Integer(
        1, bounds=(1, None), softbounds=(1, 30),
        doc='Number of epochs where, if the classifier has failed to '
        'improve it\'s error, the learning rate is reduced'
    )
    reduce_lr_cooldown = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs after reducing the learning rate before we '
        'resume checking improvements'
    )
    reduce_lr_log10_epsilon = param.Integer(
        -8, bounds=(None, 0),
        doc='The log10 absolute difference between error rates that, below '
        'which, reducing the error rate is considered meaningless'
    )
    reduce_lr_burnin = param.Integer(
        0, bounds=(0, None), softbounds=(0, 10),
        doc='Number of epochs before the criterion for reducing the learning '
        'rate kicks in'
    )
    seed = param.Integer(
        None,
        doc='Seed used for training procedures (e.g. dropout). If '
        'unset, will not touch torch\'s seeding'
    )
    dropout_prob = param.Magnitude(
        0.,
        doc='The probability of dropping a hidden unit during training'
    )
    keep_last_and_best_only = param.Boolean(
        True,
        doc='If the model is being saved, keep only the model and optimizer '
        'parameters for the last and best epoch (in terms of validation loss).'
        ' If False, save every epoch. See also "saved_model_fmt" and '
        '"saved_optimizer_fmt"'
    )
    saved_model_fmt = param.String(
        'model_{epoch:03d}.pt',
        doc='The file name format string used to save model state information.'
        ' Entries from the state csv are used to format this string (see '
        'TrainingStateController)'
    )
    saved_optimizer_fmt = param.String(
        'optim_{epoch:03d}.pt',
        doc='The file name format string used to save optimizer state '
        'information. Entries from the state csv are used to format this '
        'string (see TrainingStateController)'
    )
