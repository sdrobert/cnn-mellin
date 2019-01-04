'''Command line hooks for cnn_mellin'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import warnings

from collections import OrderedDict

import param
import pydrobert.param.serialization as serial
import pydrobert.param.argparse as pargparse
import pydrobert.torch.data as data
import pydrobert.torch.training as training
import cnn_mellin.models as models
import cnn_mellin.running as running
import torch

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
        optimizer = param.ObjectSelector(
            torch.optim.Adam, objects={
                'adam': torch.optim.Adam,
                'adadelta': torch.optim.Adadelta,
                'adagrad': torch.optim.Adagrad,
                'sgd': torch.optim.SGD,
            },
            doc='The optimizer to train with'
        )
        weight_decay = param.Number(
            0, bounds=(0, None),
            doc='The L2 penalty to apply to weights'
        )
    return OrderedDict((
        ('model', models.AcousticModelParams(name='model')),
        ('training', TrainingParams(name='training')),
        ('data', data.ContextWindowDataParams(name='data')),
        ('train_data', data.DataSetParams(name='train_data')),
        ('val_data', data.DataSetParams(name='val_data')),
        ('pdfs_data', data.DataSetParams(name='pdfs_data')),
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
    parser.add_argument(
        '--add-help-string', action='store_true', default=False,
        help='If set, will include parameter help strings at the start of the '
        'printout'
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
        serial.deserialize_from_ini(in_config, param_dict)
    serial.serialize_to_ini(
        sys.stdout, param_dict, include_help=options.add_help_string)
    return 0


def _acoustic_model_forward_pdfs_parse_args(args):
    parser = argparse.ArgumentParser(
        description=acoustic_model_forward_pdfs.__doc__
    )
    param_dict = _construct_default_param_dict()
    parser.add_argument(
        '--config',
        action=pargparse.ParameterizedIniReadAction,
        help='Read in a INI (config) file for settings',
        parameterized=param_dict,
    )
    parser.add_argument(
        '--device',
        type=torch.device,
        default=torch.device('cuda'),
        help='The torch device to run the forward step on'
    )
    parser.add_argument(
        '--pdfs-dir', default=None,
        help='Directory to save pdfs to. If unset, will write to data_dir + '
        "'/pdfs'"
    )
    parser.add_argument(
        'log_prior_file',
        help='A file containing a FloatTensor of log-prior distribution over '
        'targets'
    )
    parser.add_argument('data_dir', help='Path to feat data directory')
    subparser = parser.add_subparsers(
        title='model acquisition', dest='model',
        description='How to get a hold of the model parameters we use'
    )
    path_parser = subparser.add_parser(
        'path', help='Load model parameters from a simple path'
    )
    path_parser.add_argument('model_path')
    history_parser = subparser.add_parser(
        'history',
        help='Load the model from the history of a model\'s training. By '
        'default, loads the best model (in terms of validation loss)'
    )
    history_parser.add_argument(
        'state_dir',
        help='The directory where the training states, including model '
        'parameters, is located'
    )
    history_parser.add_argument(
        'state_csv',
        help='A path to the CSV file where the training history is listed'
    )
    history_parser.add_argument(
        '--last', default=False, action='store_true',
        help='If set, will load the last model (in terms of epochs) instead '
        'of the best model'
    )
    return parser.parse_args(args)


def acoustic_model_forward_pdfs(args=None):
    '''Write emission probabilities from cnn-mellin'''
    try:
        options = _acoustic_model_forward_pdfs_parse_args(args)
    except SystemExit as ex:
        return ex.code
    model = models.AcousticModel(
        options.config['model'],
        1 + options.config['data'].context_left +
        options.config['data'].context_right,
    )
    if options.model == 'path':
        model_path = options.model_path
    else:
        controller = training.TrainingStateController(
            options.config['training'],
            state_csv_path=options.state_csv,
            state_dir=options.state_dir
        )
        if options.last:
            epoch = controller.get_last_epoch()
        else:
            epoch = controller.get_best_epoch()
        if epoch:
            info = controller.get_info(epoch)
            model_path = os.path.join(
                options.state_dir,
                controller.params.saved_model_fmt.format(**info)
            )
        else:
            warnings.warn(
                'There are no training logs listed in "{}"! pdfs will be '
                'random'
            )
            model_path = None
        del controller
    if model_path is not None:
        model_device = next(model.parameters()).device
        state_dict = torch.load(model_path, map_location=model_device)
        model.load_state_dict(state_dict)
        del state_dict
    log_prior = torch.load(options.log_prior_file, map_location=options.device)
    pdfs_data = data.ContextWindowEvaluationDataLoader(
        options.data_dir,
        data.ContextWindowDataSetParams(
            name='pdfs_data_set',
            **param.param_union(
                options.config['data'],
                options.config['pdfs_data'],
            )
        ),
    )
    running.write_am_pdfs(
        model,
        pdfs_data,
        log_prior,
        device=options.device,
        pdfs_dir=options.pdfs_dir,
    )
    return 0


def _train_acoustic_model_parse_args(args):
    parser = argparse.ArgumentParser(
        description=train_acoustic_model.__doc__
    )
    param_dict = _construct_default_param_dict()
    parser.add_argument(
        '--config',
        action=pargparse.ParameterizedIniReadAction,
        help='Read in a INI (config) file for settings',
        parameterized=param_dict,
    )
    parser.add_argument(
        '--device',
        type=torch.device,
        default=torch.device('cuda'),
        help='The torch device to run training on'
    )
    parser.add_argument(
        '--train-num-data-workers',
        type=int,
        default=os.cpu_count() - 1,
        help='The number of subprocesses to spawn to serve training data. 0 '
        'means data are served on the main thread. The default is one fewer '
        'than the number of CPUs available'
    )
    parser.add_argument(
        '--state-csv',
        default=None,
        help='Where to store the training history CSV file. Set this if you '
        'expect training to halt or want to keep track of the loss'
    )
    parser.add_argument(
        'state_dir',
        help='Path to where model and optimizer states are stored'
    )
    parser.add_argument('train_dir', help='Path to training data')
    parser.add_argument('val_dir', help='Path to evaluation data')
    return parser.parse_args(args)


def train_acoustic_model(args=None):
    '''Train an acoustic model from cnn-mellin'''
    try:
        options = _train_acoustic_model_parse_args(args)
    except SystemExit as ex:
        return ex.code
    model = models.AcousticModel(
        options.config['model'],
        1 + options.config['data'].context_left +
        options.config['data'].context_right,
    )
    if options.config['training'].weight_tensor_file is not None:
        weight_tensor = torch.load(
            options.config['training'].weight_tensor_file,
            map_location=options.device,
        )
    else:
        weight_tensor = None
    train_data = data.ContextWindowTrainingDataLoader(
        options.train_dir,
        data.ContextWindowDataSetParams(
            name='train_data_set',
            **param.param_union(
                options.config['data'],
                options.config['train_data'],
            )
        ),
        pin_memory=(options.device.type == 'cuda'),
        num_workers=options.train_num_data_workers,
    )
    val_data = data.ContextWindowEvaluationDataLoader(
        options.val_dir,
        data.ContextWindowDataSetParams(
            name='val_data_set',
            **param.param_union(
                options.config['data'],
                options.config['val_data'],
            )
        ),
    )
    controller = training.TrainingStateController(
        options.config['training'],
        state_csv_path=options.state_csv,
        state_dir=options.state_dir
    )
    optimizer = options.config['training'].optimizer(
        model.parameters(),
        weight_decay=options.config['training'].weight_decay,
    )
    controller.load_model_and_optimizer_for_epoch(
        model, optimizer, controller.get_last_epoch())
    for epoch in range(
            controller.get_last_epoch() + 1,
            options.config['training'].num_epochs + 1):
        train_loss = running.train_am_for_epoch(
            model,
            train_data,
            optimizer,
            options.config['training'],
            epoch=epoch,
            device=options.device,
            weight=weight_tensor
        )
        val_loss = running.get_am_alignment_cross_entropy(
            model,
            val_data,
            device=options.device,
            weight=weight_tensor
        )
        if not controller.update_for_epoch(
                model, optimizer, train_loss, val_loss):
            break
    return 0
