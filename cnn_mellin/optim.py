'''Utilities for hyperparameter optimization'''

import os

from itertools import chain

import param
import pydrobert.gpyopt as gpyopt
import pydrobert.torch.data as data
import numpy as np
import cnn_mellin.models as models
import cnn_mellin.running as running

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

OPTIM_DICT = {
    'nonlinearity': ('model', 'categorical', ('relu', 'sigmoid', 'tanh')),
    'flatten_style':
        ('model', 'categorical', ('keep_filts', 'keep_chans', 'keep_both')),
    'init_num_channels': ('model', 'categorical', (1, 128, 256, 512)),
    'mellin': ('model', 'categorical', (True, False)),
    'kernel_freq': ('model', 'discrete', (1, 9)),
    'factor_sched': ('model', 'discrete', (1, 3)),
    'kernel_time': ('model', 'discrete', (1, 9)),
    'freq_factor': ('model', 'categorical', (1, 2, 4)),
    'mconv_decimation_strategy':
        ('model', 'categorical', (
            "pad-then-dec", "pad-to-dec-time-floor", "pad-to-dec-time-ceil")),
    'channels_factor': ('model', 'categorical', (None, 1, 2, 4)),
    'num_fc': ('model', 'discrete', (1, 4)),
    'num_conv': ('model', 'discrete', (1, 4)),
    'hidden_size': ('model', 'categorical', (512, 1024, 2048)),
    'dropout2d_on_conv': ('model', 'categorical', (True, False)),
    'time_factor': ('model', 'categorical', (1, 2, 4)),
    'early_stopping_threshold': ('training', 'continuous', (0.0, 0.5)),
    'early_stopping_patience': ('training', 'discrete', (1, 10)),
    'early_stopping_burnin': ('training', 'discrete', (0, 10)),
    'reduce_lr_threshold': ('training', 'continuous', (0.0, 0.5)),
    'reduce_lr_factor': ('training', 'continuous', (0.1, 0.5)),
    'reduce_lr_patience': ('training', 'discrete', (1, 10)),
    'reduce_lr_cooldown': ('training', 'discrete', (0, 10)),
    'reduce_lr_log10_epsilon': ('training', 'discrete', (-10, -2)),
    'reduce_lr_burnin': ('training', 'discrete', (0, 10)),
    'dropout_prob': ('training', 'continuous', (0.0, 0.5)),
    'weight_decay': ('training', 'continuous', (0.0, 1.0)),
    'num_epochs': ('training', 'discrete', (1, 100)),
    'log10_learning_rate': ('training', 'discrete', (-10, -1)),
    'optimizer':
        ('training', 'categorical', ('adam', 'adadelta', 'adagrad', 'sgd')),
    'weigh_training_samples': ('training', 'categorical', (True, False)),
    'context_left': ('data_set', 'discrete', (0, 5)),
    'context_right': ('data_set', 'discrete', (0, 5)),
    'batch_size': ('data_set', 'discrete', (1, 20)),
    'reverse': ('data_set', 'categorical', (True, False)),
}


class CNNMellinOptimParams(gpyopt.BayesianOptimizationParams):
    to_optimize = param.ListSelector(
        [], objects=list(OPTIM_DICT),
        doc='A list of hyperparameters to optimize with a call to '
        'optimize-acoustic-model'
    )
    partition_style = param.ObjectSelector(
        'round-robin', objects=['round-robin', 'average', 'last'],
        doc='This determines how to come up with the value to optimize. '
        '"average" trains models with a leave-one-partition-out strategy, '
        'then averages them. "last" evaluates with the last listed partition. '
        '"round-robin" rotates the evaluation partition for each successive '
        'sample of the objective.'
    )


def optimize_am(
        data_dir, partitions, optim_params, base_model_params,
        base_training_params, base_data_set_params, val_partition=False,
        weight=None, device='cpu', train_num_data_workers=os.cpu_count() - 1,
        history_csv=None):
    '''Optimize acoustic model hyperparameters

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the unpartitioned data to optimize
        with
    partitions : int or sequence
        Either an integer (>1) specifying the number of partitions to split
        `data_dir` into. If an integer, `data_dir` will be split randomly and
        roughly evenly into this number of partitions. Otherwise, can be a
        sequence of sequences, each subsequence representing a partition that
        contains all the utterance ids in the partition
    optim_params : CNNMellinOptimParams
        Specify how the optimization is to be performed, including what
        hyperparameters to optimize
    base_model_params : cnn_mellin.models.AcousticModelParams
        Default hyperparameters for the acoustic model. The values for any
        hyperparameters that are being optimized will be ignored
    base_training_params : cnn_mellin.running.TrainingParams
        Default hyperparameters used in training. The values for any
        hyperparameters that are being optimized will be ignored
    base_data_set_params : pydrobert.torch.data.ContextWindowDataSetParams
        Default hyperparameters used to load data. The values for any
        hyperparameters that are being optimized will be ignored
    val_partition : bool, optional
        Specify whether a validation partition should be used. If ``True``, the
        number of partitions must be greater than 2. If partition ``i`` is the
        evaluation partition (or partition ``i`` is currently being evaluated
        as part of the "average" scheme in `optim_params.partition_style`),
        partition ``(i - 1) % |partitions|`` will be treated as the validation
        partition.
    weight : FloatTensor, optional
        A weight tensor to be used in the loss function. If unset, but
        ``weigh_training_samples`` is listed in ``optim_params.to_optimize``,
        ``weigh_training_samples`` will be removed from ``to_optimize``
    device : torch.device or str, optional
        The device to perform training and evaluation on
    train_num_data_workers : int, optional
        The number of worker threads to spawn to serve training data. 0 means
        data are served on the main thread. The default is one fewer than the
        number of CPUs available
    history_csv : str, optional
        If set, the optimization history will be loaded and stored from a CSV
        file at this path

    Returns
    -------
    model_params, training_params, data_set_params
        Upon completion, returns the parameters with the best objective values

    Notes
    -----
    This function unsets 'seed' in ``base_*_params`` when performing
    optimization, though those seeds are retained in the returned
    parameters. Seeds are dynamically incremented for each retraining.
    '''
    subset_ids = set(base_data_set_params.subset_ids)
    if isinstance(partitions, int):
        utt_ids = data.SpectDataSet(data_dir).utt_ids
        if subset_ids:
            utt_ids &= subset_ids
        utt_ids = sorted(utt_ids)
        rng = np.random.RandomState(seed=optim_params.seed)
        rng.shuffle(utt_ids)
        new_partitions = []
        partition_size = len(utt_ids) // partitions
        for _ in range(partitions - 1):
            new_partitions.append(utt_ids[:partition_size])
            utt_ids = utt_ids[partition_size:]
        new_partitions.append(utt_ids)
        partitions = new_partitions
    num_partitions = len(partitions)
    if num_partitions < 2 + (1 if val_partition else 0):
        raise ValueError('Too few partitions')
    if not all(len(x) for x in partitions):
        raise ValueError('One or more partitions have no entries')
    if weight is None and 'weigh_training_samples' in optim_params.to_optimize:
        warnings.warn(
            'weigh_training_samples was set in to_optimize, but we do not '
            'have a weight tensor. Removing from to_optimize'
        )
        optim_params.to_optimize.remove('weigh_training_samples')
    partitions_to_average = 1
    eval_idx = [num_partitions - 1]
    seed = [0]
    model_param_dict = param.param_union(base_model_params)
    training_param_dict = param.param_union(base_training_params)
    data_param_dict = param.param_union(base_data_set_params)

    def to_next(x):
        return (x + 1) % num_partitions
    if optim_params.partition_style == 'last':

        def to_next(x):
            return x
    elif optim_params.partition_style == 'average':
        partitions_to_average = num_partitions

    def objective(**kwargs):
        model_params = models.AcousticModelParams(**model_param_dict)
        model_params.seed = seed[0]
        training_params = running.TrainingParams(**training_param_dict)
        training_params.seed = seed[0] + 1
        train_params = data.ContextWindowDataSetParams(**data_param_dict)
        train_params.seed = seed[0] + 2
        val_params = data.ContextWindowDataSetParams(**data_param_dict)
        eval_params = data.ContextWindowDataSetParams(**data_param_dict)
        seed[0] += 3
        for key, value in kwargs.items():
            params_name = OPTIM_DICT[key][0]
            if params_name == 'model':
                model_params.param.set_param(**{key: value})
            elif params_name == 'training':
                training_params.param.set_param(**{key: value})
            else:
                train_params.param.set_param(**{key: value})
                val_params.param.set_param(**{key: value})
                eval_params.param.set_param(**{key: value})
        objectives = []
        for _ in range(partitions_to_average):
            eval_subset_ids = set(partitions[eval_idx[0]])
            if val_partition:
                val_idx = (eval_idx[0] - 1) % num_partitions
                train_subset_ids = set(chain(*(
                    partitions[i] for i in range(num_partitions)
                    if i not in {val_idx, eval_idx[0]}
                )))
                val_subset_ids = set(partitions[val_idx])
            else:
                train_subset_ids = set(chain(*(
                    partitions[i] for i in range(num_partitions)
                    if i != eval_idx[0]
                )))
                val_subset_ids = train_subset_ids
            if subset_ids:
                train_subset_ids &= subset_ids
                val_subset_ids &= subset_ids
                eval_subset_ids &= subset_ids
            train_params.subset_ids = list(train_subset_ids)
            val_params.subset_ids = list(val_subset_ids)
            eval_params.subset_ids = list(eval_subset_ids)
            model = running.train_am(
                model_params, training_params, data_dir, train_params,
                data_dir, val_params, weight=weight, device=device,
                train_num_data_workers=train_num_data_workers,
                print_epochs=False
            )
            eval_data = data.ContextWindowEvaluationDataLoader(
                data_dir, eval_params)
            objectives.append(running.get_am_alignment_cross_entropy(
                model, eval_data, device=device))
            eval_idx[0] = to_next(eval_idx[0])
        return np.mean(objectives)
    wrapped = gpyopt.GPyOptObjectiveWrapper(objective)
    if history_csv:
        try:
            _, Y = wrapped.read_history_to_X_Y(history_csv)
            seed[0] = len(Y) * 3
            for _ in range(len(Y)):
                for _ in range(partitions_to_average):
                    eval_idx[0] = to_next(eval_idx[0])
        except IOError:
            pass
    for param_name in optim_params.to_optimize:
        param_type, param_constr = OPTIM_DICT[param_name][1:]
        wrapped.set_variable_parameter(param_name, param_type, param_constr)
    best = gpyopt.bayesopt(wrapped, optim_params, history_csv)
    model_params = models.AcousticModelParams(**model_param_dict)
    training_params = running.TrainingParams(**training_param_dict)
    data_params = data.ContextWindowDataSetParams(**data_param_dict)
    weigh_training_samples = None
    for key, value in best.items():
        params_name = OPTIM_DICT[key][0]
        if params_name == 'model':
            model_params.param.set_param(**{key: value})
        elif params_name == 'training':
            training_params.param.set_param(**{key: value})
        else:
            data_params.param.set_param(**{key: value})
    return model_params, training_params, data_params
