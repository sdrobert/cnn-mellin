'''Command line hooks for cnn_mellin'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from collections import OrderedDict

import param
import pydrobert.param.serialization as serial
import pydrobert.torch.data as data
import pydrobert.torch.training as training
import cnn_mellin.models as models
import cnn_mellin.running as running

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def _construct_default_param_dict():
    class TrainingParams(
            running.TrainingEpochParams, training.TrainingStateParams):
        weight_tensor_file = param.Filename(
            None,
            doc='Path to a stored tensor containing class weights. '
            'If None, training will be uniform'
        )
    return OrderedDict((
        ('model', models.AcousticModelParams(name='model')),
        ('training', TrainingParams(name='training')),
        ('data', data.SpectDataParams(name='data')),
        ('train_data', data.DataSetParams(name='train_data')),
        ('val_data', data.DataSetParams(name='val_data')),
    ))


def _print_parameters_as_ini_parse_args(args):
    parser = argparse.ArgumentParser(
        description=print_parameters_as_ini.__doc__
    )
    parser.add_argument(
        'in_configs', nargs='*', type=argparse.FileType('r'),
        help='One or more INI files to populate parameters before printing '
        'all of them. Later config files clobber earlier set parameters'
    )
    return parser.parse_args(args)


def print_parameters_as_ini(args=None):
    '''Print parameters of cnn-mellin experiment in INI format

    By default, this command prints to stdout the default values of parameters
    used in this experiment. However, those values can be modified by
    deserializing the contents of an arbitrary number of paths to INI
    configuration files.
    '''
    try:
        options = _print_parameters_as_ini_parse_args(args)
    except SystemExit as ex:
        return ex.code
    param_dict = _construct_default_param_dict()
    for in_config in options.in_configs:
        serial.serialze_from_ini(in_config, param_dict)
    serial.serialize_to_ini(sys.stdout, param_dict)
