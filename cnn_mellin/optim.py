'''Utilities for hyperparameter optimization'''

import os
import heapq
import functools
import traceback
import sys

from itertools import chain

import param
import pydrobert.gpyopt as gpyopt
import pydrobert.torch.data as data
import numpy as np
import cnn_mellin.models as models
import cnn_mellin.running as running
import torch

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

OPTIM_DICT = {
    'nonlinearity': ('model', 'categorical', ('relu', 'sigmoid', 'tanh')),
    'flatten_style':
        ('model', 'categorical', ('keep_filts', 'keep_chans', 'keep_both')),
    'init_num_channels': ('model', 'discrete', (1, 16, 32, 64)),
    'mellin': ('model', 'categorical', (True, False)),
    'kernel_freq': ('model', 'discrete', tuple(range(1, 10))),
    'factor_sched': ('model', 'discrete', tuple(range(1, 3))),
    'kernel_time': ('model', 'discrete', tuple(range(1, 10))),
    'freq_factor': ('model', 'discrete', (1, 2, 4)),
    'mconv_decimation_strategy':
        ('model', 'categorical', (
            "pad-then-dec", "pad-to-dec-time-floor", "pad-to-dec-time-ceil")),
    'channels_factor': ('model', 'discrete', (1, 2, 4)),
    'num_fc': ('model', 'discrete', tuple(range(1, 5))),
    'num_conv': ('model', 'discrete', tuple(range(1, 11))),
    'hidden_size': ('model', 'discrete', (512, 1024, 2048)),
    'dropout2d_on_conv': ('model', 'categorical', (True, False)),
    'time_factor': ('model', 'discrete', (1, 2, 4)),
    'early_stopping_threshold': ('training', 'continuous', (0.0, 0.5)),
    'early_stopping_patience': ('training', 'discrete', tuple(range(1, 11))),
    'early_stopping_burnin': ('training', 'discrete', tuple(range(0, 11))),
    'reduce_lr_threshold': ('training', 'continuous', (0.0, 0.5)),
    'reduce_lr_factor': ('training', 'continuous', (0.1, 0.5)),
    'reduce_lr_patience': ('training', 'discrete', tuple(range(1, 11))),
    'reduce_lr_cooldown': ('training', 'discrete', tuple(range(0, 11))),
    'reduce_lr_log10_epsilon': ('training', 'discrete', tuple(range(-10, -1))),
    'reduce_lr_burnin': ('training', 'discrete', tuple(range(0, 11))),
    'dropout_prob': ('training', 'continuous', (0.0, 0.5)),
    'weight_decay': ('training', 'continuous', (0.0, 1.0)),
    'num_epochs': ('training', 'discrete', tuple(range(1, 31))),
    'log10_learning_rate': ('training', 'discrete', tuple(range(-10, -3))),
    'optimizer':
        ('training', 'categorical', ('adam', 'adadelta', 'adagrad', 'sgd')),
    'weigh_training_samples': ('training', 'categorical', (True, False)),
    'context_left': ('data_set', 'discrete', tuple(range(0, 7))),
    'context_right': ('data_set', 'discrete', tuple(range(0, 7))),
    'batch_size': ('data_set', 'discrete', tuple(range(1, 21))),
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
    model_estimate_memory_limit_bytes = param.Integer(
        10 * (1024 ** 3), allow_None=True, bounds=(1, None),
        doc='How many bytes to limit the *estimated* memory requirements for '
        'training to. This should not be set to the full size of the device, '
        'in case the model size is underestimated'
    )
    nan_or_failure_loss = param.Number(
        10.0, bounds=(0., None),
        doc='If an exception occurs (e.g. out of memory) or the loss is NaN, '
        'the loss will be substituted for this value. If you set this too '
        'high, it will mess with the variance of the underlying Gaussian '
        'process. Too low, and it could become an optimum. Suggested value is '
        'twice the average loss you see in random samples'
    )


# modified from
# https://docs.fast.ai/troubleshoot.html#memory-leakage-on-exception
# we can use this decorator to clean up after ourselves on badness
def _gpu_mem_restore(sub_loss):
    "Reclaim GPU RAM if CUDA out of memory happened"
    def _gpu_mem_restore_inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            loss = float('nan')
            try:
                loss = func(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except Exception:
                type, val, tb = sys.exc_info()
                print('Got exception: ', val, '. Handling', file=sys.stderr)
                traceback.clear_frames(tb)
            if loss != loss:
                loss = sub_loss
            return loss
        return wrapper
    return _gpu_mem_restore_inner


def optimize_am(
        data_dir, partitions, optim_params, base_model_params,
        base_training_params, base_data_set_params, val_partition=False,
        weight=None, device='cpu', train_num_data_workers=os.cpu_count() - 1,
        history_csv=None, verbose=False):
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
    verbose : bool, optional
        Print intermediate results, like epoch loss and current settings

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
        utt_ids = data.SpectDataSet(
            data_dir, subset_ids=subset_ids if subset_ids else None).utt_ids
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
    # gpyopt has been very buggy about constraints. Instead, we'll raise in
    # the objective function whenever the constraint hasn't been met, which
    # will give us a substitute value
    if optim_params.model_estimate_memory_limit_bytes is not None:
        # data_params.batch_size refers to the number of utterances, whereas
        # our estimates are based on the number of windows. We'll iterate
        # through the partition to determine the maximum utterance lengths,
        # then use those to max bound the number of windows
        if 'batch_size' in optim_params.to_optimize:
            max_queue_size = OPTIM_DICT['batch_size'][2][1]
        else:
            max_queue_size = data_param_dict['batch_size']
        sds = data.SpectDataSet(
            data_dir, subset_ids=subset_ids if subset_ids else None)
        queue = []
        for feats, _ in sds:
            if len(queue) < max_queue_size:
                heapq.heappush(queue, len(feats))
            else:
                heapq.heappushpop(queue, len(feats))
        max_num_windows = sum(queue)
        del queue, sds, max_queue_size
    else:
        max_num_windows = None

    def objective(**kwargs):
        if verbose:
            print('Evaluating objective @ {}'.format(kwargs))
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
        if max_num_windows:
            bytes_estimate = models.estimate_total_size_bytes(
                model_params, max_num_windows,
                1 + train_params.context_left + train_params.context_right,
            )
            too_big = (
                bytes_estimate >
                optim_params.model_estimate_memory_limit_bytes)
        else:
            too_big = False
        objectives = []
        for _ in range(partitions_to_average):
            if verbose:
                print('Testing on partition {}'.format(eval_idx[0]))
            if device.type == 'cuda':
                torch.cuda.empty_cache()
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
            model = eval_data = xent = float('nan')
            try:
                if too_big:
                    raise ValueError("Model passed memory limit estimate")
                model = running.train_am(
                    model_params, training_params, data_dir, train_params,
                    data_dir, val_params, weight=weight, device=device,
                    train_num_data_workers=train_num_data_workers,
                    print_epochs=verbose
                )
                eval_data = data.ContextWindowEvaluationDataLoader(
                    data_dir, eval_params)
                xent = running.get_am_alignment_cross_entropy(
                    model, eval_data, device=device)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print('Got exception: {}. Assigning bad loss.'.format(e))
            finally:
                if xent != xent:
                    xent = optim_params.nan_or_failure_loss
                objectives.append(xent)
                del model, eval_data, xent
            eval_idx[0] = to_next(eval_idx[0])
        mean = np.mean(objectives)
        if verbose:
            print('Mean loss: {:.02f}'.format(mean))
        return mean
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
    # we add all possible optimizable parameters as fixed values. This way,
    # those values will be recorded in the history csv AND constraints can
    # use them
    for key in OPTIM_DICT:
        params_name = OPTIM_DICT[key][0]
        if params_name == 'model':
            wrapped.set_fixed_parameter(key, model_param_dict[key])
        elif params_name == 'training':
            wrapped.set_fixed_parameter(key, training_param_dict[key])
        else:
            wrapped.set_fixed_parameter(key, data_param_dict[key])
    for param_name in optim_params.to_optimize:
        param_type, param_constr = OPTIM_DICT[param_name][1:]
        wrapped.set_variable_parameter(param_name, param_type, param_constr)
    constraints = None
    # if optim_params.model_estimate_memory_limit_bytes is not None:
    #     # data_params.batch_size refers to the number of utterances, whereas
    #     # our estimates are based on the number of windows. We'll iterate
    #     # through the partition to determine the maximum utterance lengths,
    #     # then use those to max bound the number of windows
    #     if 'batch_size' in optim_params.to_optimize:
    #         max_queue_size = OPTIM_DICT['batch_size'][2][1]
    #     else:
    #         max_queue_size = data_param_dict['batch_size']
    #     sds = data.SpectDataSet(
    #         data_dir, subset_ids=subset_ids if subset_ids else None)
    #     queue = []
    #     for feats, _ in sds:
    #         if len(queue) < max_queue_size:
    #             heapq.heappush(queue, len(feats))
    #         else:
    #             heapq.heappushpop(queue, len(feats))
    #     max_num_windows = sum(queue)
    #     del queue, sds, max_queue_size
    #
    #     def _memory_estimate_constraint(**kwargs):
    #         model_params = models.AcousticModelParams(**model_param_dict)
    #         training_params = running.TrainingParams(**training_param_dict)
    #         data_params = data.ContextWindowDataSetParams(**data_param_dict)
    #         for key, value in kwargs.items():
    #             params_name = OPTIM_DICT[key][0]
    #             if params_name == 'model':
    #                 model_params.param.set_param(**{key: value})
    #             elif params_name == 'training':
    #                 training_params.param.set_param(**{key: value})
    #             else:
    #                 data_params.param.set_param(**{key: value})
    #         bytes_estimate = models.estimate_total_size_bytes(
    #             model_params, max_num_windows,
    #             1 + data_params.context_left + data_params.context_right,
    #         )
    #         return (
    #             bytes_estimate <=
    #             optim_params.model_estimate_memory_limit_bytes
    #         )
    #     constraints = [_memory_estimate_constraint]
    best = gpyopt.bayesopt(
        wrapped, optim_params, history_csv, constraints=constraints)
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
