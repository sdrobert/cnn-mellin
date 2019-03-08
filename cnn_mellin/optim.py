'''Utilities for hyperparameter optimization'''

import os
import heapq
import functools
import traceback
import sys

from itertools import chain
from time import time

import optuna
import param
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
    'init_num_channels': ('model', 'int', (1, 64)),
    'mellin': ('model', 'categorical', (True, False)),
    'kernel_freq': ('model', 'int', (1, 10)),
    'factor_sched': ('model', 'int', (1, 2)),
    'kernel_time': ('model', 'int', (1, 10)),
    'freq_factor': ('model', 'int', (1, 4)),
    'mconv_decimation_strategy':
        ('model', 'categorical', (
            "pad-then-dec", "pad-to-dec-time-floor", "pad-to-dec-time-ceil")),
    'channels_factor': ('model', 'int', (1, 4)),
    'num_fc': ('model', 'int', (1, 3)),
    'num_conv': ('model', 'int', (0, 3)),
    'hidden_size': ('model', 'categorical', (512, 1024, 2048)),
    'dropout2d_on_conv': ('model', 'categorical', (True, False)),
    'time_factor': ('model', 'int', (1, 4)),
    'early_stopping_threshold': ('training', 'uniform', (0.0, 0.5)),
    'early_stopping_patience': ('training', 'int', (1, 10)),
    'early_stopping_burnin': ('training', 'int', (0, 10)),
    'reduce_lr_threshold': ('training', 'uniform', (0.0, 0.5)),
    'reduce_lr_factor': ('training', 'log_uniform', (0.1, 0.5)),
    'reduce_lr_patience': ('training', 'int', (1, 10)),
    'reduce_lr_cooldown': ('training', 'int', (0, 10)),
    'reduce_lr_log10_epsilon': ('training', 'uniform', (-10, -2)),
    'reduce_lr_burnin': ('training', 'discrete', (0, 10)),
    'dropout_prob': ('training', 'log_uniform', (1e-5, 0.5)),
    'weight_decay': ('training', 'log_uniform', (1e-5, 0.5)),
    'num_epochs': ('training', 'int', (5, 30)),
    'log10_learning_rate': ('training', 'uniform', (-10, -3)),
    'optimizer':
        ('training', 'categorical', ('adam', 'adadelta', 'adagrad', 'sgd')),
    'weigh_training_samples': ('training', 'categorical', (True, False)),
    'context_left': ('data_set', 'int', (0, 10)),
    'context_right': ('data_set', 'int', (0, 10)),
    'batch_size': ('data_set', 'int', (5, 20)),
    'reverse': ('data_set', 'categorical', (True, False)),
}


# this is inspired by Yusuke's class here:
# https://github.com/Minyus/optkeras/blob/1ccd3ba89cfa9bdf973f73045111eeae1060c3c5/optkeras/optkeras.py#L318
# We also prune repeats when they've failed or been pruned, as well as the
# "running" ones that were interrupted
class RepeatPruner(optuna.pruners.BasePruner):
    '''Prune if we've already tried those parameters'''
    def prune(self, storage, study_id, trial_id, step):
        cur_trial = storage.get_trial(trial_id)
        agent_name = cur_trial.user_attrs.get('agent_name', float('nan'))
        trials = storage.get_all_trials(study_id)
        for trial in trials:
            if trial.trial_id == trial_id:
                continue
            if (
                    trial.state == optuna.structs.TrialState.RUNNING and
                    (
                        trial.user_attrs.get('agent_name', float('nan')) ==
                        agent_name)):
                continue
            if trial.params == cur_trial.params:
                return True
        return False


class TimerPruner(optuna.pruners.BasePruner):
    '''Prune if a step exceeds some wall clock time threshold'''

    def __init__(self, seconds):
        super(TimerPruner, self).__init__()
        self.seconds = seconds

    def prune(self, storage, study_id, trial_id, step):
        stamp = time()
        trial = storage.get_trial(trial_id)
        stamps = trial.user_attrs.get('intermediate_timestamps', [])
        stamps.append(stamp)
        storage.set_trial_user_attr(
            trial_id, 'intermediate_timestamps', stamps)
        if len(stamps) == 1:
            return False
        prev_stamp = stamps[-2]
        return abs(stamp - prev_stamp) > self.seconds


class ChainOrPruner(optuna.pruners.BasePruner):
    '''Prune if one of any passed pruners tells us to'''

    def __init__(self, pruners):
        super(ChainOrPruner, self).__init__()
        self.pruners = list(pruners)

    def prune(self, storage, study_id, trial_id, step):
        for pruner in self.pruners:
            if pruner.prune(storage, study_id, trial_id, step):
                return True
        return False


class CNNMellinOptimParams(param.Parameterized):
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
    seed = param.Integer(
        None,
        doc='Seed used for bayesian optimization'
    )
    max_samples = param.Integer(
        None, bounds=(0, None),
        doc='If set, the max number of samples. history and '
        'initial_design_samples both count towards this. E.g. max_samples=5, '
        'initial_design_numdata=4, and 3 samples have been loaded from a '
        'history file. 1 sample will be taken to complete the initial design, '
        ' 1 according to the acquisition function over the initial design\'s '
        'samples. If running in distributed mode, there is no guarantee that '
        'this number will be resepected'
    )
    initial_design_samples = param.Integer(
        5, bounds=(0, None),
        doc='The number of initial samples before optimization. History counts'
        'towards this. This number does not matter if random sampling'
    )
    sampler = param.ObjectSelector(
        'tpe', objects=['tpe', 'random'],
        doc='What performs the optimization. tpe = Tree of Parzen Estimators '
        'with EI; random = random sampling'
    )
    study_name = param.String(
        'optim_am', allow_None=False,
        doc='What the study is called in the history database, if it exists'
    )
    max_time_per_epoch = param.Number(
        None, bounds=(1., None),
        doc='The maximum time, in seconds, allowed to train a model for an '
        'epoch. If exceeded, once the epoch is done, the trial will be pruned.'
        ' If unset, no trial will time out'
    )
    median_pruner_epoch_warmup = param.Integer(
        None, bounds=(0, None),
        doc='If set, median pruning will be enabled, and will begin after '
        'these many epochs. If unset, median pruning will be disabled'
    )
    random_after_n_unsuccessful_trials = param.Integer(
        5, bounds=(1, None), allow_None=True,
        doc='If set, and there have been this many or more sequential '
        'failed or pruned trials, sample the next trial randomly. This helps '
        'avoid a non-random sampler getting caught in a situation where it '
        'repeatedly samples infeasible points'
    )


def optimize_am(
        data_dir, partitions, optim_params, base_model_params,
        base_training_params, base_data_set_params, val_partition=False,
        weight=None, device='cpu', train_num_data_workers=os.cpu_count() - 1,
        history_url=None, verbose=False, agent_name=None):
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
    history_url : str, optional
        If set, the optimization history will be loaded and stored from a
        relational database according to SQLAlchemy
    verbose : bool, optional
        Print intermediate results, like epoch loss and current settings
    agent_name : str, optional
        A name assigned to the caller. If restarting after a SIGINT, trial(s)
        will exist in the database with state RUNNING that are not actually
        running anymore. If `agent_name` is set, trials whose `agent_name`
        properties match `agent_name` and whose states are RUNNING will be
        discounted from seed and evalution partition calculations, improving
        reproducibility (in the non-distributed case)

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
    if optim_params.seed is None:
        optim_seed = np.random.randint(1000000000)
    else:
        optim_seed = optim_params.seed
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = torch.device(device)
    subset_ids = set(base_data_set_params.subset_ids)
    if isinstance(partitions, int):
        utt_ids = data.SpectDataSet(
            data_dir, subset_ids=subset_ids if subset_ids else None).utt_ids
        utt_ids = sorted(utt_ids)
        rng = np.random.RandomState(seed=optim_seed)
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
    model_param_dict = param.param_union(base_model_params)
    training_param_dict = param.param_union(base_training_params)
    data_param_dict = param.param_union(base_data_set_params)
    if optim_params.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(
            consider_endpoints=True,
            n_startup_trials=optim_params.initial_design_samples,
            seed=optim_seed,
        )
    else:
        sampler = optuna.samplers.RandomSampler(seed=optim_seed)
    study = optuna.create_study(
        storage=history_url,
        study_name=optim_params.study_name,
        sampler=sampler,
        load_if_exists=True,
        pruner=ChainOrPruner([
            RepeatPruner(),
            # we always include the TimerPruner, even if disabled, so we can
            # record the epoch-wise time to completion
            TimerPruner(
                float('inf') if optim_params.max_time_per_epoch is None
                else optim_params.max_time_per_epoch
            ),
        ])
    )
    if optim_params.median_pruner_epoch_warmup is not None:
        study.pruner.pruners.append(optuna.pruners.MedianPruner(
            n_startup_trials=optim_params.initial_design_samples,
            n_warmup_steps=optim_params.median_pruner_epoch_warmup,
        ))
    running_discount = sum(
        1 for trial in study.trials
        if trial.state == optuna.structs.TrialState.RUNNING and
        trial.user_attrs.get('agent_name', float('nan')) == agent_name
    )

    def to_next_seed(trial):
        return ((trial.trial_id - running_discount) * 44893) % 49811

    def to_next_partition(trial):
        return num_partitions - 1
    if optim_params.partition_style == 'round-robin':

        def to_next_partition(trial):
            return (trial.trial_id - running_discount) % num_partitions
    elif optim_params.partition_style == 'average':
        partitions_to_average = num_partitions
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
        max_num_windows = dict()
        for batch_size in range(len(queue), 0, -1):
            max_num_windows[batch_size] = sum(queue)
            heapq.heappop(queue)
        del queue, sds, max_queue_size
    else:
        max_num_windows = None

    def objective(trial):
        trial.set_user_attr('agent_name', agent_name)
        model_params = models.AcousticModelParams(**model_param_dict)
        model_params.seed = to_next_seed(trial)
        training_params = running.TrainingParams(**training_param_dict)
        training_params.seed = model_params.seed + 1
        train_params = data.ContextWindowDataSetParams(**data_param_dict)
        train_params.seed = model_params.seed + 2
        val_params = data.ContextWindowDataSetParams(**data_param_dict)
        eval_params = data.ContextWindowDataSetParams(**data_param_dict)
        if verbose:
            print('Beginning trial:', end='')
        for key in optim_params.to_optimize:
            param_name, param_type, param_arg = OPTIM_DICT[key]
            if param_type == 'uniform':
                value = trial.suggest_uniform(key, *param_arg)
            elif param_type == 'log_uniform':
                value = trial.suggest_loguniform(key, *param_arg)
            elif param_type == 'categorical':
                value = trial.suggest_categorical(key, param_arg)
            elif param_type == 'discrete_uniform':
                value = trial.suggest_discrete_uniform(key, *param_arg)
            else:
                value = trial.suggest_int(key, *param_arg)
            if verbose:
                print('{}={},'.format(key, value), end='')
            if param_name == 'model':
                model_params.param.set_param(**{key: value})
            elif param_name == 'training':
                training_params.param.set_param(**{key: value})
            else:
                train_params.param.set_param(**{key: value})
                val_params.param.set_param(**{key: value})
                eval_params.param.set_param(**{key: value})
        if verbose:
            print('')
        if trial.should_prune(0):
            if verbose:
                print('Pruning trial - Already seen')
            raise optuna.structs.TrialPruned()
        if max_num_windows:
            bytes_estimate = models.estimate_total_size_bytes(
                model_params, max_num_windows[train_params.batch_size],
                1 + train_params.context_left + train_params.context_right,
            )
            if bytes_estimate > optim_params.model_estimate_memory_limit_bytes:
                raise ValueError("Estimate size {}B exceeds limit {}B".format(
                    bytes_estimate,
                    optim_params.model_estimate_memory_limit_bytes))
        objectives = []
        for p_shift in range(partitions_to_average):
            eval_partition = to_next_partition(trial) + p_shift
            eval_partition %= num_partitions
            if verbose:
                print('Evaluating on partition {}'.format(eval_partition))
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            eval_subset_ids = set(partitions[eval_partition])
            if val_partition:
                val_idx = (eval_partition - 1) % num_partitions
                train_subset_ids = set(chain(*(
                    partitions[i] for i in range(num_partitions)
                    if i not in {val_idx, eval_partition}
                )))
                val_subset_ids = set(partitions[val_idx])
            else:
                train_subset_ids = set(chain(*(
                    partitions[i] for i in range(num_partitions)
                    if i != eval_partition
                )))
                val_subset_ids = train_subset_ids
            if subset_ids:
                train_subset_ids &= subset_ids
                val_subset_ids &= subset_ids
                eval_subset_ids &= subset_ids
            train_params.subset_ids = list(train_subset_ids)
            val_params.subset_ids = list(val_subset_ids)
            eval_params.subset_ids = list(eval_subset_ids)

            if not p_shift:
                # only record intermediate results for the first partition
                callbacks = tuple()
            else:

                def report_and_prune(dict_):
                    trial.report(dict_['val_loss'], dict_['epoch'])
                    if trial.should_prune(dict_['epoch']):
                        raise optuna.structs.TrialPruned()
                callbacks = (report_and_prune,)
            model = running.train_am(
                model_params, training_params, data_dir, train_params,
                data_dir, val_params, weight=weight, device=device,
                train_num_data_workers=train_num_data_workers,
                print_epochs=verbose,
                callbacks=callbacks
            )
            eval_data = data.ContextWindowEvaluationDataLoader(
                data_dir, eval_params)
            xent = running.get_am_alignment_cross_entropy(
                model, eval_data, device=device)
            objectives.append(xent)
            del model, eval_data, xent
        mean = np.mean(objectives)
        if verbose:
            print('Mean loss: {:.02f}'.format(mean))
        return mean
    if optim_params.max_samples is not None:
        n_trials = optim_params.max_samples
    else:
        n_trials = float('inf')
    # this strange loop manages the sampler's seeds and keeps the number of
    # trials in the ballpark of max_samples when distributed
    completed_trials = sum(
        1 for trial in study.trials if trial.state.is_finished())
    while completed_trials < n_trials:
        all_trials = study.trials
        # other_trials will only keep the seed consistent when not distributed
        other_trials = len(all_trials) - running_discount
        sampler.rng = np.random.RandomState(
            optim_seed + other_trials)
        if hasattr(sampler, 'random_sampler'):
            sampler.random_sampler.rng = np.random.RandomState(
                (optim_seed + other_trials) * 16319)
            if optim_params.random_after_n_unsuccessful_trials:
                all_trials.sort(key=lambda trial: -trial.trial_id)
                n_unsuccessful = 0
                for trial in all_trials:
                    if trial.state == optuna.structs.TrialState.RUNNING:
                        continue
                    if trial.state == optuna.structs.TrialState.COMPLETE:
                        break
                    n_unsuccessful += 1
                if (
                        n_unsuccessful >=
                        optim_params.random_after_n_unsuccessful_trials):
                    study.sampler = sampler.random_sampler
        del all_trials
        study.optimize(objective, n_trials=1)
        study.sampler = sampler
        completed_trials = sum(
            1 for trial in study.trials if trial.state.is_finished())
    best = study.best_params
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
