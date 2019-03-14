'''Command line hooks for cnn_mellin'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import warnings

from collections import OrderedDict
from itertools import count as icount

import param
import pydrobert.param.serialization as serial
import pydrobert.param.argparse as pargparse
import pydrobert.torch.data as data
import pydrobert.torch.training as training
import cnn_mellin.models as models
import cnn_mellin.running as running
import torch

from cnn_mellin import TupleList

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def _construct_default_param_dict():
    dict_ = OrderedDict((
        ('model', models.AcousticModelParams(name='model')),
        ('training', running.TrainingParams(name='training')),
        ('data', data.ContextWindowDataParams(name='data')),
        ('train_data', data.DataSetParams(name='train_data')),
        ('val_data', data.DataSetParams(name='val_data')),
        ('pdfs_data', data.DataSetParams(name='pdfs_data')),
    ))
    try:
        from cnn_mellin.optim import CNNMellinOptimParams
        dict_['optim'] = CNNMellinOptimParams(name='optim')
    except ImportError as e:  # gpyopt missing
        pass
    return dict_


class CommaListSerializer(serial.DefaultSerializer):
    def help_string(self, name, parameterized):
        return 'Elements separated by commas'

    def serialize(self, name, parameterized):
        val = super(CommaListSerializer, self).serialize(name, parameterized)
        return ', '.join(str(x) for x in val)


class CommaTupleListSerializer(serial.DefaultSerializer):
    def help_string(self, name, parameterized):
        return (
            'elements are separated by commas. An element is a tuple, which '
            'is a bracketed, comma-delimited list'
        )

    def serialize(self, name, parameterized):
        val = super(CommaTupleListSerializer, self).serialize(
            name, parameterized)
        return ', '.join(
            '(' + ', '.join(str(xx) for xx in x) + ')'
            for x in val
        )


class CommaListDeserializer(serial.DefaultListDeserializer):
    def deserialize(self, name, block, parameterized):
        if block.strip() == '':
            block = []
        else:
            p = parameterized.params()[name]
            try:
                if p.class_:
                    block = [
                        p.class_(x.strip(), *self.args, **self.kwargs)
                        for x in block.split(',')
                    ]
                else:
                    block = [x.strip() for x in block.split(',')]
            except (TypeError, ValueError) as e:
                raise_from(ParamConfigTypeError(parameterized, name), e)
        parameterized.param.set_param(name, block)


class CommaTupleListDeserializer(serial.DefaultListDeserializer):
    def deserialize(self, name, block, parameterized):
        if block.strip() == '':
            block = []
        else:
            p = parameterized.params()[name]
            try:
                new_block = []
                new_tuple = None
                for x in block.split(','):
                    x = x.strip()
                    append = False
                    while x.startswith('('):
                        if new_tuple is not None:
                            raise ValueError('nested parentheses not allowed')
                        x = x[1:].strip()
                        new_tuple = []
                    while x.endswith(')'):
                        if new_tuple is None or append:
                            raise ValueError(
                                'closing parenthesis without open')
                        x = x[:-1].strip()
                        append = True
                    if new_tuple is None:
                        raise ValueError(
                            'Found element {} , but not in tuple'.format(x))
                    if p.class_:
                        x = p.class_(x, *self.args, **self.kwargs)
                    new_tuple.append(x)
                    if append:
                        new_block.append(tuple(new_tuple))
                        new_tuple = None
                block = new_block
            except (TypeError, ValueError) as e:
                raise_from(ParamConfigTypeError(parameterized, name), e)
        parameterized.param.set_param(name, block)


class CommaListSelectorSerializer(serial.DefaultListSelectorSerializer):
    def help_string(self, name, parameterized):
        choices_help_string = super(
            CommaListSelectorSerializer, self).help_string(name, parameterized)
        return 'Elements separated by commas. ' + choices_help_string

    def serialize(self, name, parameterized):
        val = super(CommaListSelectorSerializer, self).serialize(
            name, parameterized)
        return ', '.join(str(x) for x in val)


class CommaListSelectorDeserializer(serial.DefaultListSelectorDeserializer):
    def deserialize(self, name, block, parameterized):
        if block.strip() == '':
            block = []
        else:
            block = [x.strip() for x in block.split(',')]
        super(CommaListSelectorDeserializer, self).deserialize(
            name, block, parameterized)


def _print_parameters_as_ini_parse_args(args):
    parser = argparse.ArgumentParser(
        description=print_parameters_as_ini.__doc__
    )
    parser.add_argument(
        'in_configs', nargs='*', type=argparse.FileType('r'),
        help='One or more INI files to populate parameters before printing '
        'all of them. Later config files clobber earlier set parameters'
    )
    try:
        import cnn_mellin.optim
        parser.add_argument(
            '--history-url', default=None,
            help='If set, will clobber parameters with the best values found '
            'from this optimization database'
        )
        parser.add_argument(
            '--data-set-config-section',
            choices=['train_data', 'val_data', 'pdfs_data'],
            default='train_data',
            help='What section to write data-set related parameters to if '
            '--history-url has been set'
        )
    except ImportError:
        pass
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
        serial.deserialize_from_ini(
            in_config, param_dict,
            deserializer_type_dict={
                param.ListSelector: CommaListSelectorDeserializer(),
                param.List: CommaListDeserializer(),
                TupleList: CommaTupleListDeserializer(),
            },
        )
    if hasattr(options, 'history_url') and options.history_url is not None:
        import optuna
        study = optuna.study.Study(
            study_name=param_dict['optim'].study_name,
            storage=options.history_url
        )
        data_set_params = data.ContextWindowDataSetParams(
            name='data_set',
            **param.param_union(
                options.config['data'],
                options.config[options.data_set_config_section],
            )
        )
        optimization_candidates = set(
            options.config['optim'].params()['to_optimize'].objects)
        write_trial_params_to_parameterizeds(
            study.best_params,
            options.config['model'],
            options.config['training'],
            data_set_params,
        )
        for config_params, optimized_params in (
                (options.config['model'], optins.config['model']),
                (options.config['data'], data_set_params),
                (
                    options.config[options.data_set_config_section],
                    data_set_params),
                (options.config['training'], options.config['training'])):
            for param_name in config_params.param.params().keys():
                if param_name in optimization_candidates:
                    config_params.param.set_param(
                        **{param_name: getattr(optimized_params, param_name)})
    serial.serialize_to_ini(
        sys.stdout, param_dict,
        serializer_type_dict={
            param.ListSelector: CommaListSelectorSerializer(),
            param.List: CommaListSerializer(),
            TupleList: CommaTupleListSerializer(),
        },
        include_help=options.add_help_string
    )
    return 0


def _target_count_info_to_tensor(args):
    parser = argparse.ArgumentParser(
        description=target_count_info_to_tensor.__doc__
    )
    parser.add_argument(
        '--num-targets', type=int, default=None,
        help='The total number of targets. If num-targets is fewer than the '
        'maximum count index + 1 from the in_file, this command will error. '
        'If unset, num-targets will be inferred to be the maximum count index '
        '+ 1 from the in_file'
    )
    parser.add_argument(
        '--min-count', type=int, default=1,
        help='If a count is less than this value, it will be increased to this'
        'value'
    )
    parser.add_argument(
        'in_file', type=argparse.FileType('r'),
        help='The info file'
    )
    parser.add_argument(
        'tensor_type', choices=['inv_weight', 'log_prior'],
        help='What to convert the counts to. inv_weight = an inverse '
        'weight vector. Targets are weighed according to how frequently they '
        '*do not* show up in data ((total - count) / total). log_prior = the '
        'frequentist estimate of the probability of seeing a target, in the '
        'log domain (log(count / total))'
    )
    parser.add_argument(
        'out_file', type=argparse.FileType('wb'),
        help='The file path to write the float tensor to'
    )
    return parser.parse_args(args)


# FIXME(sdrobert): should this be in pydrobert-pytorch?
def target_count_info_to_tensor(args=None):
    '''Convert SpectData info file target counts to a FloatTensor

    Given a file produced by get-torch-spect-data-dir-info on a torch data
    directory with an ``ali`` directory, this function converts target counts
    to a ``torch.FloatTensor``, which can be used in the training or decoding
    processes
    '''
    try:
        options = _target_count_info_to_tensor(args)
    except SystemExit as ex:
        return ex.code
    count_dict = dict()
    for line in options.in_file:
        key, count = line.strip().split(' ')
        if key.startswith('count_'):
            target = int(key[6:])
            count_dict[target] = int(count)
    if not len(count_dict):
        if options.num_targets is not None:
            warnings.warn(
                'No counts were found in "{}". Are you sure the directory it '
                'came from included alignments?'.format(options.in_file.name)
            )
        else:
            print(
                'No counts were found in "{}" and the number of targets is '
                'unknown. Are you sure the directory it came from included '
                'aligments?'.format(options.in_file.name),
                file=sys.stderr
            )
            return 1
    max_target = max(count_dict.keys(), default=-1)
    if options.num_targets is None:
        num_targets = max_target + 1
    else:
        num_targets = options.num_targets
        if max_target >= num_targets:
            print(
                'Saw count with ID={}, but --num-targets={}'.format(
                    max_target, num_targets),
                file=sys.stderr
            )
            return 1
    counts = []
    total = 0
    for target in range(num_targets):
        count = max(count_dict.get(target, 0), options.min_count)
        total += count
        counts.append(count)
    if not total:
        print('Total count was 0', file=sys.stderr)
        return 1
    counts = torch.FloatTensor(counts)
    if options.tensor_type == 'inv_weight':
        t = (total - counts) / total
    else:
        t = torch.log(counts / total)
    torch.save(t, options.out_file)
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
        deserializer_type_dict={
            param.ListSelector: CommaListSelectorDeserializer(),
            param.List: CommaListDeserializer(),
            TupleList: CommaTupleListDeserializer(),
        },
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
        pin_memory=(options.device.type == 'cuda'),
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
        deserializer_type_dict={
            param.ListSelector: CommaListSelectorDeserializer(),
            param.List: CommaListDeserializer(),
            TupleList: CommaTupleListDeserializer(),
        },
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
        '--weight-tensor-file', type=argparse.FileType('rb'), default=None,
        help='Path to a stored tensor containing class weights. If unset, '
        'training will be uniform. If set, it will only be used if'
        'weigh_training_samples = True in the [training] configuration.'
    )
    parser.add_argument(
        '--state-csv',
        default=None,
        help='Where to store the training history CSV file. Set this if you '
        'expect training to halt or want to keep track of the loss'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=False,
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
    if options.weight_tensor_file is not None:
        weight_tensor = torch.load(
            options.weight_tensor_file,
            map_location=options.device,
        )
    else:
        weight_tensor = None
    train_params = data.ContextWindowDataSetParams(
        name='train_data_set',
        **param.param_union(
            options.config['data'],
            options.config['train_data'],
        )
    )
    val_params = data.ContextWindowDataSetParams(
        name='val_data_set',
        **param.param_union(
            options.config['data'],
            options.config['train_data'],
        )
    )
    running.train_am(
        options.config['model'],
        options.config['training'],
        options.train_dir,
        train_params,
        options.val_dir,
        val_params,
        state_dir=options.state_dir,
        state_csv=options.state_csv,
        weight=weight_tensor,
        device=options.device,
        train_num_data_workers=options.train_num_data_workers,
        print_epochs=options.verbose,
    )
    return 0


def _optimize_acoutic_model_parse_args(args):
    parser = argparse.ArgumentParser(
        description=optimize_acoustic_model.__doc__
    )
    param_dict = _construct_default_param_dict()
    parser.add_argument(
        '--config',
        action=pargparse.ParameterizedIniReadAction,
        help='Read in a INI (config) file for settings',
        parameterized=param_dict,
        deserializer_type_dict={
            param.ListSelector: CommaListSelectorDeserializer(),
            param.List: CommaListDeserializer(),
            TupleList: CommaTupleListDeserializer(),
        },
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
        '--weight-tensor-file', type=argparse.FileType('rb'), default=None,
        help='Path to a stored tensor containing class weights. If unset, '
        'training will be uniform.'
    )
    parser.add_argument(
        '--no-val-partition', dest='val_partition', action='store_false',
        default=True,
        help='If this flag is set, no validation partition will be used for '
        'lr decay, early stopping, etc. Instead, test data will be evaluated'
    )
    parser.add_argument(
        '--history-url', default=None,
        help='What database to store intermediate results to. If not set, '
        'will be in-memory'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=False,
    )
    parser.add_argument(
        '--data-set-config-section',
        choices=['train_data', 'val_data', 'pdfs_data'], default='train_data',
        help='What section of the configuration to combine with [data] to '
        'make a data set configuration. This section will dictate, for '
        'example, the base batch_size and the subset_ids that the '
        'optimization will be restricted to. Optimized settings will be '
        'written back to this section'
    )
    parser.add_argument(
        '--agent-name', default=None,
        help='A name assigned to the caller. If restarting after a SIGINT, '
        'trial(s) will exist in the database with state RUNNING that are not '
        'actually running anymore. If `agent_name` is set, trials whose '
        '`agent_name` properties match `agent_name` and whose states are '
        'RUNNING will bediscounted from seed and evalution partition '
        'calculations, improving reproducibility (in the non-distributed case)'
    )
    parser.add_argument(
        '--add-help-string', action='store_true', default=False,
        help='If set, will include parameter help strings at the start of the '
        'printout'
    )
    parser.add_argument('data_dir', help='Path to optimization data')
    parser.add_argument(
        'partitions', nargs='+',
        help='Either an integer that will be used to dynamically partition '
        '`data_dir` into that many partitions (usually must be >3, but >2 if '
        '--no-val-partition), or some number of files (> 3) containing '
        'utterance ids, one per line, of each partition'
    )
    parser.add_argument(
        'out_ini',
        help='Path to write the optimize config (INI) file to'
    )
    return parser.parse_args(args)


def optimize_acoustic_model(args=None):
    '''Optimize an acoustic model from cnn-mellin'''
    try:
        options = _optimize_acoutic_model_parse_args(args)
    except SystemExit as ex:
        return ex.code
    if options.weight_tensor_file is not None:
        weight_tensor = torch.load(
            options.weight_tensor_file,
            map_location=options.device,
        )
    else:
        weight_tensor = None
    try:
        if len(options.partitions) == 1:
            partitions = int(options.partitions[0])
        else:
            partitions = tuple(
                tuple(x.strip() for x in open(f))
                for f in options.partitions
            )
    except ValueError:
        print(
            'partitions must be an integer or sequence of file names',
            file=sys.stderr
        )
        return 1
    import cnn_mellin.optim as optim
    data_set_params = data.ContextWindowDataSetParams(
        name='data_set',
        **param.param_union(
            options.config['data'],
            options.config[options.data_set_config_section],
        )
    )
    model_params, training_params, data_set_params = optim.optimize_am(
        options.data_dir, partitions, options.config['optim'],
        options.config['model'], options.config['training'],
        data_set_params,
        val_partition=options.val_partition,
        weight=weight_tensor,
        device=options.device,
        train_num_data_workers=options.train_num_data_workers,
        history_url=options.history_url,
        verbose=options.verbose,
        agent_name=options.agent_name,
    )
    # only set the values that could have been optimized. This avoids weirdness
    # like setting the partition for individual data sets
    optimization_candidates = set(
        options.config['optim'].params()['to_optimize'].objects)
    for config_params, optimized_params in (
            (options.config['model'], model_params),
            (options.config['data'], data_set_params),
            (options.config[options.data_set_config_section], data_set_params),
            (options.config['training'], training_params)):
        for param_name in config_params.param.params().keys():
            if param_name in optimization_candidates:
                config_params.param.set_param(
                    **{param_name: getattr(optimized_params, param_name)})
    serial.serialize_to_ini(
        options.out_ini, options.config,
        serializer_type_dict={
            param.ListSelector: CommaListSelectorSerializer(),
            param.List: CommaListSerializer(),
            TupleList: CommaTupleListSerializer(),
        },
        include_help=options.add_help_string
    )
    return 0
