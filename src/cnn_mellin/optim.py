import os
from typing import List, Optional, Union
import warnings
import gc
import time

from copy import deepcopy

import torch
import optuna
import pydrobert.param.optuna as poptuna
import pydrobert.param.serialization as serialization
import pydrobert.torch.data as data
import cnn_mellin.running as running
import cnn_mellin.models as models
import sqlalchemy.engine.url

from cnn_mellin import construct_default_param_dict


def init_study(
    train_dir: str,
    param_dict: dict,
    out_url: Union[str, sqlalchemy.engine.url.URL],
    only: set,
    study_name: Optional[str] = None,
    device: Union[torch.device, str] = "cpu",
    dev_prop: float = 0.1,
    mem_limit: int = 6 * (1024 ** 3),
) -> optuna.Study:
    only = set(only)  # non-destructive

    out_url = sqlalchemy.engine.url.make_url(out_url)
    if out_url.database is None:
        raise ValueError(f"'{out_url}' database path is invalid or empty")
    if study_name is None:
        study_name = os.path.basename(out_url.database).split(".")[0]
    train_dir = os.path.abspath(train_dir)
    num_filts, num_classes = running.get_filts_and_classes(train_dir)
    raw = num_filts == 1

    if (
        param_dict["training"].num_epochs is None
        and not param_dict["training"].early_stopping_threshold
        and "training.num_epochs" not in only
        and "training.early_stopping_threshold" not in only
    ):
        raise ValueError(
            "A trial has no way to stop. Set either training.num_epochs or "
            "training.early_stopping_threshold"
        )

    # determine things that won't make sense to train
    if raw:
        _chuck_set_from_only(
            only,
            {
                "model.convolutional_kernel_freq",
                "model.convolutional_freq_factor",
                "training.max_time_warp",
                "training.max_freq_warp",
                "training.max_freq_mask",
                "training.num_freq_mask",
            },
            "features are likely raw (only one coefficient)",
        )
    elif (
        not param_dict["training"].num_freq_mask
        and "training.num_freq_mask" not in only
    ):
        _chuck_set_from_only(
            only,
            {"training.max_freq_mask"},
            "training.num_freq_mask is 0 and not optimized",
        )
    if (
        not param_dict["training"].num_time_mask
        and "training.num_time_mask" not in only
    ):
        _chuck_set_from_only(
            only,
            {
                "training.num_time_mask_proportion",
                "training.max_time_mask",
                "training.max_time_mask_proportion",
            },
            "training.num_time_mask is 0 and not optimized",
        )
    if (
        not param_dict["model"].convolutional_layers
        and "model.convolutional_layers" not in only
    ):
        _chuck_set_from_only(
            only,
            {
                "model.convolutional_mellin",
                "model.convolutional_kernel_time",
                "model.convolutional_kernel_freq",
                "model.convolutional_initial_channels",
                "model.convolutional_factor_sched",
                "model.convolutional_time_factor",
                "model.convolutional_freq_factor",
                "model.convolutional_channel_factor",
                "training.convolutional_dropout_2d",
                "model.convolutional_nonlinearity",
            },
            "model.convolutional_layers is 0 and not optmized",
        )
    elif (
        not param_dict["training"].dropout_prob and "training.dropout_prob" not in only
    ):
        _chuck_set_from_only(
            only,
            {"training.convolutional_dropout_2d"},
            "training.dropout_prob is 0 and not optimized",
        )
    if (
        not param_dict["model"].recurrent_layers
        and "model.recurrent_layers" not in only
    ):
        _chuck_set_from_only(
            only,
            {
                "model.recurrent_size",
                "model.recurrent_bidirectional",
                "model.recurrent_type",
            },
            "model.recurrent_layers is 0 and not optimized",
        )
    if (
        not param_dict["training"].reduce_lr_threshold
        and "training.reduce_lr_threshold" not in only
    ):
        _chuck_set_from_only(
            only,
            {
                "training.reduce_lr_factor",
                "training.reduce_lr_patience",
                "training.reduce_lr_cooldown",
                "training.reduce_lr_burnin",
            },
            "training.reduce_lr_threshold is 0 and not optimized",
        )
    if (
        not param_dict["training"].early_stopping_threshold
        and "training.early_stopping_threshold" not in only
    ):
        _chuck_set_from_only(
            only,
            {"training.early_stopping_burnin", "training.early_stopping_patience"},
            "training.early_stopping_threshold is 0 and not optimized",
        )

    # check that we have anything remaining
    if not only:
        raise ValueError("set 'only' cannot be empty (and cannot be reduced to empty)")
    remainder = only - poptuna.get_param_dict_tunable(param_dict)
    if remainder:
        raise ValueError(
            f"set 'only' contains elements {remainder} which are not valid parameters"
        )

    # determine what values to optimize given our restrictions
    optimized_dicts = set()
    for name_ in only:
        optimized_dicts.add(name_.split(".")[0])
    directions = ["minimize", "minimize"]
    optimized_values = ["er", "wall_time"]
    if "model" in optimized_dicts or "data" in optimized_dicts:
        directions.append("minimize")
        optimized_values.append("memory")
    else:
        warnings.warn(
            "No model or data hyperparameters detected. Will not optimize memory usage"
        )

    # get some other user restrictions
    ds = data.SpectDataSet(
        train_dir,
        subset_ids=param_dict["data"].subset_ids
        if param_dict["data"].subset_ids
        else None,
    )
    total_utts = len(ds)
    num_train = int(total_utts * (1 - dev_prop))
    num_dev = total_utts - num_train
    if num_train < 1 or num_dev < 1:
        raise ValueError(
            f"Could not split {total_utts} utterances into {1 - dev_prop:%} "
            f"({num_train}) training utterances and {dev_prop:%} ({num_dev}) dev "
            "utterances"
        )
    train_ids = ds.utt_ids[:num_train]
    dev_ids = ds.utt_ids[num_train:]
    max_frames = max(x[0].size(0) for x in ds)
    serialized = serialization.serialize_to_dict(param_dict)
    del ds

    study = optuna.create_study(
        storage=str(out_url), study_name=study_name, directions=directions
    )
    if raw:
        study.set_user_attr("raw", True)
    study.set_user_attr("device", str(device))
    study.set_user_attr("data_dir", train_dir)
    study.set_user_attr("train_ids", train_ids)
    study.set_user_attr("dev_ids", dev_ids)
    study.set_user_attr("num_classes", num_classes)
    study.set_user_attr("max_frames", max_frames)
    study.set_user_attr("global_dict", serialized)
    study.set_user_attr("num_filts", num_filts)
    study.set_user_attr("optimized_values", optimized_values)
    study.set_user_attr("mem_limit", mem_limit)
    study.set_user_attr("only", sorted(only))
    return study


def _chuck_set_from_only(only: set, set_: set, reason: str):
    for tunable in only & set_:
        warnings.warn(
            f"Removing {tunable} from list of optimized hyperparameters since {reason}"
        )
        only.remove(tunable)


def get_forward_backward_memory(
    param_dict: dict, num_filts: int, num_classes: int, max_frames: int
) -> int:

    # do our best to stop deallocs from outside this scope being counted
    gc.collect()

    with torch.autograd.profiler.profile(profile_memory=True) as prof:
        model = models.AcousticModel(num_filts, num_classes + 1, param_dict["model"])
        optim = param_dict["training"].optimizer(model.parameters(), lr=1e-4)
        x = torch.empty(param_dict["data"].batch_size, max_frames, num_filts)
        len_ = torch.full((param_dict["data"].batch_size,), max_frames)
        optim.zero_grad()
        logits, out_len = model(
            x,
            len_,
            param_dict["training"].dropout_prob,
            param_dict["training"].convolutional_dropout_2d,
        )
        z = logits.sum()
        z.backward()
        optim.step()
    del model, optim, out_len, logits, x, len_, z

    return prof.total_average().cpu_memory_usage


def objective(trial: optuna.Trial) -> List[float]:
    user_attrs = trial.study.user_attrs
    device = torch.device(user_attrs["device"])
    # try to make a tensor on the device. Raises a runtime error if it can't
    torch.empty(1, device=device)
    global_dict = construct_default_param_dict()
    serialization.deserialize_from_dict(user_attrs["global_dict"], global_dict)
    param_dict = poptuna.suggest_param_dict(trial, global_dict, set(user_attrs["only"]))
    # deserialize a copy of the data params into a parameter set for the dev partition
    dev_params = data.SpectDataSetParams(name="dev")
    serialization.deserialize_from_dict(user_attrs["global_dict"]["data"], dev_params)
    param_dict["data"].subset_ids = user_attrs["train_ids"]
    dev_params = deepcopy(param_dict["data"])
    dev_params.subset_ids = user_attrs["dev_ids"]

    # def pruner_callback(epoch: int, train_loss: float, dev_err: float):
    #     trial.report(dev_err, epoch)
    #     if trial.should_prune():
    #         raise optuna.exceptions.TrialPruned()

    try:

        results = [None, None]
        if "memory" in user_attrs["optimized_values"]:
            val = get_forward_backward_memory(
                param_dict,
                user_attrs["num_filts"],
                user_attrs["num_classes"],
                user_attrs["max_frames"],
            )
            if val > user_attrs["mem_limit"]:
                raise optuna.exceptions.TrialPruned()
            results.append(None)
            results[user_attrs["optimized_values"].index("memory")] = float(val)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        start = time.time()
        _, er = running.train_am(
            param_dict["model"],
            param_dict["training"],
            param_dict["data"],
            user_attrs["data_dir"],
            user_attrs["data_dir"],
            None,
            device,
            dev_data_params=dev_params,
        )
        wall_time = time.time() - start

        results[user_attrs["optimized_values"].index("wall_time")] = wall_time
        results[user_attrs["optimized_values"].index("er")] = er

    except RuntimeError:  # probably an OOM
        raise optuna.exceptions.TrialPruned()

    return results


# class ChainOrPruner(optuna.pruners.BasePruner):
#     """Prune if one of any passed pruners tells us to"""

#     def __init__(self, pruners):
#         super(ChainOrPruner, self).__init__()
#         self.pruners = list(pruners)

#     def prune(self, storage, study_id, trial_id, step):
#         for pruner in self.pruners:
#             if pruner.prune(storage, study_id, trial_id, step):
#                 return True
#         return False


# OPTIM_BOUNDS = {
#     "convolutional_factor_sched": (1, 100),
#     "convolutional_freq_factor": (1, 4),
#     "num_fc": (0, 5),
#     "num_conv": (0, 10),
#     "convolutional_time_factor": (1, 4),
#     "early_stopping_threshold": (1e-5, 0.5),
#     "early_stopping_patience": (1, 10),
#     "early_stopping_burnin": (0, 10),
#     "reduce_lr_threshold": (1e-5, 0.5),
#     "reduce_lr_patience": (1, 10),
#     "reduce_lr_factor": (0.1, 0.5),
#     "reduce_lr_cooldown": (0, 10),
#     "reduce_lr_log10_epsilon": (-10, -2),
#     "reduce_lr_burnin": (0, 10),
#     "dropout_prob": (0.0, 0.5),
#     "weight_decay": (1e-5, 0.5),
#     "num_epochs": (5, 30),
#     "log10_learning_rate": (-10, -3),
#     "context_left": (0, 20),
#     "context_right": (0, 20),
#     "batch_size": (5, 20),
#     "kernel_time": (1, 10),
#     "kernel_freq": (1, 10),
#     "kernel_cout": (1, 512),
#     "hidden_size": (1, 2048),
# }

# OPTIM_SOURCES = {
#     "nonlinearity": "model",
#     "flatten_style": "model",
#     "mellin": "model",
#     "convolutional_factor_sched": "model",
#     "convolutional_freq_factor": "model",
#     "mconv_decimation_strategy": "model",
#     "dropout2d_on_conv": "model",
#     "convolutional_time_factor": "model",
#     "kernel_sizes": "model",
#     "hidden_sizes": "model",
#     "early_stopping_threshold": "training",
#     "early_stopping_patience": "training",
#     "early_stopping_burnin": "training",
#     "reduce_lr_threshold": "training",
#     "reduce_lr_factor": "training",
#     "reduce_lr_patience": "training",
#     "reduce_lr_cooldown": "training",
#     "reduce_lr_log10_epsilon": "training",
#     "reduce_lr_burnin": "training",
#     "dropout_prob": "training",
#     "weight_decay": "training",
#     "num_epochs": "training",
#     "log10_learning_rate": "training",
#     "optimizer": "training",
#     "weigh_training_samples": "training",
#     "context_left": "data_set",
#     "context_right": "data_set",
#     "batch_size": "data_set",
#     "reverse": "data_set",
#     "num_conv": "model",
#     "num_fc": "model",
# }


# class CNNMellinOptimParams(param.Parameterized):
#     to_optimize = param.ListSelector(
#         [],
#         objects=list(OPTIM_SOURCES),
#         doc="A list of hyperparameters to optimize with a call to "
#         "optimize-acoustic-model",
#     )
#     partition_style = param.ObjectSelector(
#         "round-robin",
#         objects=["round-robin", "average", "last"],
#         doc="This determines how to come up with the value to optimize. "
#         '"average" trains models with a leave-one-partition-out strategy, '
#         'then averages them. "last" evaluates with the last listed partition. '
#         '"round-robin" rotates the evaluation partition for each successive '
#         "sample of the objective.",
#     )
#     model_estimate_memory_limit_bytes = param.Integer(
#         10 * (1024 ** 3),
#         allow_None=True,
#         bounds=(1, None),
#         doc="How many bytes to limit the *estimated* memory requirements for "
#         "training to. This should not be set to the full size of the device, "
#         "in case the model size is underestimated",
#     )
#     seed = param.Integer(None, doc="Seed used for bayesian optimization")
#     max_samples = param.Integer(
#         None,
#         bounds=(0, None),
#         doc="If set, the max number of samples. history and "
#         "initial_design_samples both count towards this. E.g. max_samples=5, "
#         "initial_design_numdata=4, and 3 samples have been loaded from a "
#         "history file. 1 sample will be taken to complete the initial design, "
#         " 1 according to the acquisition function over the initial design's "
#         "samples. If running in distributed mode, there is no guarantee that "
#         "this number will be resepected",
#     )
#     initial_design_samples = param.Integer(
#         5,
#         bounds=(0, None),
#         doc="The number of initial samples before optimization. History counts"
#         "towards this. This number does not matter if random sampling",
#     )
#     sampler = param.ObjectSelector(
#         "tpe",
#         objects=["tpe", "random"],
#         doc="What performs the optimization. tpe = Tree of Parzen Estimators "
#         "with EI; random = random sampling",
#     )
#     study_name = param.String(
#         "optim_am",
#         allow_None=False,
#         doc="What the study is called in the history database, if it exists",
#     )
#     max_time_per_epoch = param.Number(
#         None,
#         bounds=(1.0, None),
#         doc="The maximum time, in seconds, allowed to train a model for an "
#         "epoch. If exceeded, once the epoch is done, the trial will be pruned."
#         " If unset, no trial will time out",
#     )
#     median_pruner_epoch_warmup = param.Integer(
#         None,
#         bounds=(0, None),
#         doc="If set, median pruning will be enabled, and will begin after "
#         "these many epochs. If unset, median pruning will be disabled",
#     )
#     random_after_n_unsuccessful_trials = param.Integer(
#         5,
#         bounds=(1, None),
#         allow_None=True,
#         doc="If set, and there have been this many or more sequential "
#         "failed or pruned trials, sample the next trial randomly. This helps "
#         "avoid a non-random sampler getting caught in a situation where it "
#         "repeatedly samples infeasible points",
#     )


# def optimize_am(
#     data_dir,
#     partitions,
#     optim_params,
#     base_model_params,
#     base_training_params,
#     base_data_set_params,
#     val_partition=False,
#     weight=None,
#     device="cpu",
#     train_num_data_workers=os.cpu_count() - 1,
#     history_url=None,
#     verbose=False,
#     agent_name=None,
# ):
#     """Optimize acoustic model hyperparameters

#     Parameters
#     ----------
#     data_dir : str
#         Path to the directory containing the unpartitioned data to optimize
#         with
#     partitions : int or sequence
#         Either an integer (>1) specifying the number of partitions to split
#         `data_dir` into. If an integer, `data_dir` will be split randomly and
#         roughly evenly into this number of partitions. Otherwise, can be a
#         sequence of sequences, each subsequence representing a partition that
#         contains all the utterance ids in the partition
#     optim_params : CNNMellinOptimParams
#         Specify how the optimization is to be performed, including what
#         hyperparameters to optimize
#     base_model_params : cnn_mellin.models.AcousticModelParams
#         Default hyperparameters for the acoustic model. The values for any
#         hyperparameters that are being optimized will be ignored
#     base_training_params : cnn_mellin.running.TrainingParams
#         Default hyperparameters used in training. The values for any
#         hyperparameters that are being optimized will be ignored
#     base_data_set_params : pydrobert.torch.data.ContextWindowDataSetParams
#         Default hyperparameters used to load data. The values for any
#         hyperparameters that are being optimized will be ignored
#     val_partition : bool, optional
#         Specify whether a validation partition should be used. If ``True``, the
#         number of partitions must be greater than 2. If partition ``i`` is the
#         evaluation partition (or partition ``i`` is currently being evaluated
#         as part of the "average" scheme in `optim_params.partition_style`),
#         partition ``(i - 1) % |partitions|`` will be treated as the validation
#         partition.
#     weight : FloatTensor, optional
#         A weight tensor to be used in the loss function. If unset, but
#         ``weigh_training_samples`` is listed in ``optim_params.to_optimize``,
#         ``weigh_training_samples`` will be removed from ``to_optimize``
#     device : torch.device or str, optional
#         The device to perform training and evaluation on
#     train_num_data_workers : int, optional
#         The number of worker threads to spawn to serve training data. 0 means
#         data are served on the main thread. The default is one fewer than the
#         number of CPUs available
#     history_url : str, optional
#         If set, the optimization history will be loaded and stored from a
#         relational database according to SQLAlchemy
#     verbose : bool, optional
#         Print intermediate results, like epoch loss and current settings
#     agent_name : str, optional
#         A name assigned to the caller. If restarting after a SIGINT, trial(s)
#         will exist in the database with state RUNNING that are not actually
#         running anymore. If `agent_name` is set, trials whose `agent_name`
#         properties match `agent_name` and whose states are RUNNING will be
#         discounted from seed and evalution partition calculations, improving
#         reproducibility (in the non-distributed case)

#     Returns
#     -------
#     model_params, training_params, data_set_params
#         Upon completion, returns the parameters with the best objective values

#     Notes
#     -----
#     This function unsets 'seed' in ``base_*_params`` when performing
#     optimization, though those seeds are retained in the returned
#     parameters. Seeds are dynamically incremented for each retraining.
#     """
#     if optim_params.seed is None:
#         # maximum is 2 ** 32 - 1, but we do some prime number wiggling
#         optim_seed = np.random.randint(2 ** 16 - 1)
#     else:
#         optim_seed = optim_params.seed
#     if verbose:
#         optuna.logging.set_verbosity(optuna.logging.INFO)
#     else:
#         optuna.logging.set_verbosity(optuna.logging.WARNING)
#     device = torch.device(device)
#     subset_ids = set(base_data_set_params.subset_ids)
#     if isinstance(partitions, int):
#         utt_ids = data.SpectDataSet(
#             data_dir, subset_ids=subset_ids if subset_ids else None
#         ).utt_ids
#         utt_ids = sorted(utt_ids)
#         rng = np.random.RandomState(seed=optim_seed)
#         rng.shuffle(utt_ids)
#         new_partitions = []
#         partition_size = len(utt_ids) // partitions
#         for _ in range(partitions - 1):
#             new_partitions.append(utt_ids[:partition_size])
#             utt_ids = utt_ids[partition_size:]
#         new_partitions.append(utt_ids)
#         partitions = new_partitions
#     num_partitions = len(partitions)
#     if num_partitions < 2 + (1 if val_partition else 0):
#         raise ValueError("Too few partitions")
#     if not all(len(x) for x in partitions):
#         raise ValueError("One or more partitions have no entries")
#     if weight is None and "weigh_training_samples" in optim_params.to_optimize:
#         warnings.warn(
#             "weigh_training_samples was set in to_optimize, but we do not "
#             "have a weight tensor. Removing from to_optimize"
#         )
#         optim_params.to_optimize.remove("weigh_training_samples")
#     partitions_to_average = 1
#     model_param_dict = param.param_union(base_model_params)
#     training_param_dict = param.param_union(base_training_params)
#     data_param_dict = param.param_union(base_data_set_params)
#     if optim_params.sampler == "tpe":
#         sampler = optuna.samplers.TPESampler(
#             consider_endpoints=True,
#             n_startup_trials=optim_params.initial_design_samples,
#             seed=optim_seed,
#         )
#     else:
#         sampler = optuna.samplers.RandomSampler(seed=optim_seed)
#     study = optuna.create_study(
#         storage=history_url,
#         study_name=optim_params.study_name,
#         sampler=sampler,
#         load_if_exists=True,
#         pruner=ChainOrPruner([]),
#     )
#     if optim_params.median_pruner_epoch_warmup is not None:
#         study.pruner.pruners.append(
#             optuna.pruners.MedianPruner(
#                 n_startup_trials=optim_params.initial_design_samples,
#                 n_warmup_steps=optim_params.median_pruner_epoch_warmup,
#             )
#         )
#     running_trials = [
#         trial
#         for trial in study.trials
#         if trial.state == optuna.structs.TrialState.RUNNING
#         and trial.user_attrs.get("agent_name", float("nan")) == agent_name
#     ]

#     def to_next_seed(trial):
#         return (trial.number * 44893) % 49811

#     def to_next_partition(trial):
#         return num_partitions - 1

#     if optim_params.partition_style == "round-robin":

#         def to_next_partition(trial):
#             return trial.number % num_partitions

#     elif optim_params.partition_style == "average":
#         partitions_to_average = num_partitions
#     if optim_params.model_estimate_memory_limit_bytes is not None:
#         # data_params.batch_size refers to the number of utterances, whereas
#         # our estimates are based on the number of windows. We'll iterate
#         # through the partition to determine the maximum utterance lengths,
#         # then use those to max bound the number of windows
#         if "batch_size" in optim_params.to_optimize:
#             max_queue_size = OPTIM_BOUNDS["batch_size"][1]
#         else:
#             max_queue_size = data_param_dict["batch_size"]
#         sds = data.SpectDataSet(data_dir, subset_ids=subset_ids if subset_ids else None)
#         queue = []
#         for feats, _ in sds:
#             if len(queue) < max_queue_size:
#                 heapq.heappush(queue, len(feats))
#             else:
#                 heapq.heappushpop(queue, len(feats))
#         max_num_windows = dict()
#         for batch_size in range(len(queue), 0, -1):
#             max_num_windows[batch_size] = sum(queue)
#             heapq.heappop(queue)
#         del queue, sds, max_queue_size
#     else:
#         max_num_windows = None

#     def objective(trial):
#         trial.set_user_attr("agent_name", agent_name)
#         # we do this in case we're restarting a trial after a sigint
#         trial_seed = trial.user_attrs.get("trial_seed", to_next_seed(trial))
#         trial.set_user_attr("trial_seed", trial_seed)
#         model_params = models.AcousticModelParams(**model_param_dict)
#         model_params.seed = trial_seed
#         training_params = running.TrainingParams(**training_param_dict)
#         training_params.seed = trial_seed + 1
#         train_params = data.ContextWindowDataSetParams(**data_param_dict)
#         train_params.seed = trial_seed + 2
#         val_params = data.ContextWindowDataSetParams(**data_param_dict)
#         eval_params = data.ContextWindowDataSetParams(**data_param_dict)
#         _write_params_from_objective_trial(
#             trial,
#             optim_params.to_optimize,
#             model_params,
#             training_params,
#             train_params,
#             val_params,
#             eval_params,
#             max_num_windows,
#             optim_params.model_estimate_memory_limit_bytes,
#         )
#         if verbose:
#             print("Beginning trial: ", trial.params)
#         for other_trial in study.trials:
#             if (
#                 other_trial.number != trial.number
#                 and other_trial.params == trial.params
#             ):
#                 raise ValueError("Trial params match previous")
#         objectives = []
#         for p_shift in range(partitions_to_average):
#             eval_partition = to_next_partition(trial) + p_shift
#             eval_partition %= num_partitions
#             if verbose:
#                 print("Evaluating on partition {}".format(eval_partition))
#             if device.type == "cuda":
#                 torch.cuda.empty_cache()
#             eval_subset_ids = set(partitions[eval_partition])
#             if val_partition:
#                 val_idx = (eval_partition - 1) % num_partitions
#                 train_subset_ids = set(
#                     chain(
#                         *(
#                             partitions[i]
#                             for i in range(num_partitions)
#                             if i not in {val_idx, eval_partition}
#                         )
#                     )
#                 )
#                 val_subset_ids = set(partitions[val_idx])
#             else:
#                 train_subset_ids = set(
#                     chain(
#                         *(
#                             partitions[i]
#                             for i in range(num_partitions)
#                             if i != eval_partition
#                         )
#                     )
#                 )
#                 val_subset_ids = train_subset_ids
#             if subset_ids:
#                 train_subset_ids &= subset_ids
#                 val_subset_ids &= subset_ids
#                 eval_subset_ids &= subset_ids
#             train_params.subset_ids = list(train_subset_ids)
#             val_params.subset_ids = list(val_subset_ids)
#             eval_params.subset_ids = list(eval_subset_ids)

#             if p_shift:
#                 # only record intermediate results for the first partition
#                 epoch_callbacks = batch_callbacks = tuple()
#             else:
#                 timeout = optim_params.max_time_per_epoch
#                 if timeout is None:
#                     timeout = float("inf")

#                 def check_no_timeout(batch_idx, loss):
#                     start = trial.user_attrs["epoch_start"]
#                     now = time()
#                     if abs(now - start) > timeout:
#                         raise ValueError(
#                             "Training time exceeded max of {:.2f}s" "".format(timeout)
#                         )

#                 def report_and_prune(dict_):
#                     trial.set_user_attr("epoch_start", time())
#                     trial.report(dict_["val_loss"], dict_["epoch"])
#                     if trial.should_prune(dict_["epoch"]):
#                         raise optuna.structs.TrialPruned()

#                 batch_callbacks = (check_no_timeout,)
#                 epoch_callbacks = (report_and_prune,)
#                 trial.set_user_attr("epoch_start", time())
#             model = running.train_am(
#                 model_params,
#                 training_params,
#                 data_dir,
#                 train_params,
#                 data_dir,
#                 val_params,
#                 weight=weight,
#                 device=device,
#                 train_num_data_workers=train_num_data_workers,
#                 print_epochs=verbose,
#                 batch_callbacks=batch_callbacks,
#                 epoch_callbacks=epoch_callbacks,
#             )
#             eval_data = data.ContextWindowEvaluationDataLoader(data_dir, eval_params)
#             xent = running.get_am_alignment_cross_entropy(
#                 model, eval_data, device=device
#             )
#             objectives.append(xent)
#             del model, eval_data, xent
#         mean = np.mean(objectives)
#         if verbose:
#             print("Mean loss: {:.02f}".format(mean))
#         return mean

#     if optim_params.max_samples is not None:
#         max_trials = optim_params.max_samples
#     else:
#         max_trials = float("inf")
#     # first finish all trials left running after sigint
#     for frozen_trial in running_trials:
#         trial_id = frozen_trial.trial_id
#         trial = optuna.trial.Trial(study, trial_id)
#         try:
#             result = objective(trial)
#         except optuna.structs.TrialPruned as e:
#             message = "Setting status of trial#{} as {}. {}".format(
#                 frozen_trial.number, optuna.structs.TrialState.PRUNED, str(e)
#             )
#             study.logger.info(message)
#             study.storage.set_trial_state(trial_id, optuna.structs.TrialState.PRUNED)
#             continue
#         except Exception as e:
#             message = (
#                 "Setting status of trial#{} as {} because of the following "
#                 "error: {}".format(
#                     frozen_trial.number, optuna.structs.TrialState.FAIL, repr(e)
#                 )
#             )
#             study.logger.warning(message, exc_info=True)
#             study.storage.set_trial_state(trial_id, optuna.structs.TrialState.FAIL)
#             study.storage.set_trial_system_attr(trial_id, "fail_reason", message)
#             continue
#         try:
#             result = float(result)
#         except (ValueError, TypeError):
#             message = (
#                 "Setting status of trial#{} as {} because the returned value "
#                 "from the objective function cannot be casted to float. "
#                 "Returned value is: {}".format(
#                     trial_id, optuna.structs.TrialState.FAIL, repr(result)
#                 )
#             )
#             study.logger.warning(message)
#             study.storage.set_trial_state(trial_id, optuna.structs.TrialState.FAIL)
#             study.storage.set_trial_system_attr(trial_id, "fail_reason", message)
#             continue
#         if result != result:
#             message = (
#                 "Setting status of trial#{} as {} because the objective "
#                 "function returned {}.".format(
#                     trial_id, optuna.structs.TrialState.FAIL, result
#                 )
#             )
#             study.logger.warning(message)
#             study.storage.set_trial_state(trial_id, optuna.structs.TrialState.FAIL)
#             study.storage.set_trial_system_attr(trial_id, "fail_reason", message)
#             continue
#         trial.report(result)
#         study.storage.set_trial_state(trial_id, optuna.structs.TrialState.COMPLETE)
#     del running_trials
#     # we assume that all other agents will likewise finish their running
#     # trials
#     n_trials = study.storage.get_n_trials(study.study_id)
#     completed_trials = n_trials - study.storage.get_n_trials(
#         study.study_id, state=optuna.structs.TrialState.FAIL
#     )
#     completed_trials -= study.storage.get_n_trials(
#         study.study_id, state=optuna.structs.TrialState.RUNNING
#     )
#     while completed_trials < max_trials:
#         # n_trials will only keep the seed consistent when not distributed
#         sampler.rng = np.random.RandomState(optim_seed + n_trials)
#         if hasattr(sampler, "random_sampler"):
#             sampler.random_sampler.rng = np.random.RandomState(
#                 (optim_seed + n_trials) * 16319
#             )
#             if optim_params.random_after_n_unsuccessful_trials:
#                 all_trials = study.trials
#                 all_trials.sort(key=lambda trial: -trial.trial_id)
#                 n_unsuccessful = 0
#                 for trial in all_trials:
#                     if trial.state == optuna.structs.TrialState.RUNNING:
#                         continue
#                     if trial.state == optuna.structs.TrialState.COMPLETE:
#                         break
#                     n_unsuccessful += 1
#                 if n_unsuccessful >= optim_params.random_after_n_unsuccessful_trials:
#                     study.sampler = sampler.random_sampler
#                 del all_trials
#         study.optimize(objective, n_trials=1)
#         study.sampler = sampler
#         n_trials = study.storage.get_n_trials(study.study_id)
#         completed_trials = n_trials - study.storage.get_n_trials(
#             study.study_id, state=optuna.structs.TrialState.FAIL
#         )
#         completed_trials -= study.storage.get_n_trials(
#             study.study_id, state=optuna.structs.TrialState.RUNNING
#         )
#     model_params = models.AcousticModelParams(**model_param_dict)
#     training_params = running.TrainingParams(**training_param_dict)
#     data_set_params = data.ContextWindowDataSetParams(**data_param_dict)
#     write_trial_params_to_parameterizeds(
#         study.best_params, model_params, training_params, data_set_params
#     )
#     return model_params, training_params, data_set_params


# def write_trial_params_to_parameterizeds(
#     trial_params, model_params, training_params, data_set_params
# ):
#     """Write optuna trial parameters to model/training parameterized configs

#     Parameters
#     ----------
#     trial_params : dict
#         A dictionary of parameters from an optimization Optuna study (a call to
#         ``optimize_am``). For example, given
#         ``isinstance(study, optuna.study.Study)``, ``study.best_params`` would
#         be suitable
#     model_params : AcousticModelParams
#         Where to store model parameters from `trial_params`
#     training_params : TrainingParams
#         Where to store training parameters from `trial_params`
#     data_set_params : pydrobert.torch.data.ContextWindowDataSetParams
#         Where to store data set parameters from `trial_params`
#     """
#     for key, value in trial_params.items():
#         if key in OPTIM_SOURCES:
#             source = OPTIM_SOURCES[key]
#             if key == "num_conv":
#                 kernel_sizes = []
#                 for layer_idx in range(value):
#                     kernel_sizes.append(
#                         (
#                             trial_params["kw_{}".format(layer_idx)],
#                             trial_params["kh_{}".format(layer_idx)],
#                             trial_params["co_{}".format(layer_idx)],
#                         )
#                     )
#                 model_params.kernel_sizes = kernel_sizes
#             elif key == "num_fc":
#                 hidden_sizes = []
#                 for layer_idx in range(value):
#                     hidden_sizes.append(trial_params["hidden_{}".format(layer_idx)])
#                 model_params.hidden_sizes = hidden_sizes
#             elif source == "model":
#                 model_params.param.set_param(key, value)
#             elif source == "training":
#                 training_params.param.set_param(key, value)
#             else:
#                 data_set_params.param.set_param(key, value)


# def _write_params_from_objective_trial(
#     trial,
#     to_optimize,
#     model_params,
#     training_params,
#     train_params,
#     val_params,
#     eval_params,
#     max_num_windows,
#     model_limit_bytes,
# ):
#     # this ugly piece of crap is called during optimization. There are so
#     # many conditionals because it isn't necessary to sample many parameters
#     # unless some other condition is met. Fortunately, optuna (i.e. TPE) allows
#     # for this
#     to_optimize = set(to_optimize)
#     if "early_stopping_threshold" in to_optimize:
#         training_params.early_stopping_threshold = trial.suggest_loguniform(
#             "early_stopping_threshold", *OPTIM_BOUNDS["early_stopping_threshold"]
#         )
#     if "reduce_lr_threshold" in to_optimize:
#         training_params.reduce_lr_threshold = trial.suggest_loguniform(
#             "reduce_lr_threshold", *OPTIM_BOUNDS["reduce_lr_threshold"]
#         )
#     if "weight_decay" in to_optimize:
#         training_params.weight_decay = trial.suggest_loguniform(
#             "weight_decay", *OPTIM_BOUNDS["weight_decay"]
#         )
#     if "num_epochs" in to_optimize:
#         training_params.num_epochs = trial.suggest_int(
#             "num_epochs", *OPTIM_BOUNDS["num_epochs"]
#         )
#     if "log10_learning_rate" in to_optimize:
#         training_params.log10_learning_rate = trial.suggest_uniform(
#             "log10_learning_rate", *OPTIM_BOUNDS["log10_learning_rate"]
#         )
#     if "weigh_training_samples" in to_optimize:
#         training_params.weigh_training_samples = trial.suggest_categorical(
#             "weigh_training_samples", (True, False)
#         )
#     if "context_left" in to_optimize:
#         train_params.context_left = (
#             val_params.context_left
#         ) = eval_params.context_left = trial.suggest_int(
#             "context_left", *OPTIM_BOUNDS["context_left"]
#         )
#     if "context_right" in to_optimize:
#         train_params.context_right = (
#             val_params.context_right
#         ) = eval_params.context_right = trial.suggest_int(
#             "context_right", *OPTIM_BOUNDS["context_right"]
#         )
#     if "batch_size" in to_optimize:
#         train_params.batch_size = (
#             val_params.batch_size
#         ) = eval_params.batch_size = trial.suggest_int(
#             "batch_size", *OPTIM_BOUNDS["batch_size"]
#         )
#     if "reverse" in to_optimize:
#         train_params.reverse = (
#             val_params.reverse
#         ) = eval_params.reverse = trial.suggest_categorical("reverse", (True, False))
#     if training_params.early_stopping_threshold:
#         # we want these parameters to sample from min_ to not occurring,
#         # but not have multiple values for not occurring
#         if "early_stopping_burnin" in to_optimize:
#             min_, max_ = OPTIM_BOUNDS["early_stopping_burnin"]
#             max_ = min(max_, training_params.num_epochs - 1)
#             if min_ <= max_:
#                 training_params.early_stopping_burnin = trial.suggest_int(
#                     "early_stopping_burnin", min_, max_
#                 )
#         if "early_stopping_patience" in to_optimize:
#             min_, max_ = OPTIM_BOUNDS["early_stopping_patience"]
#             max_ = min(
#                 max_,
#                 training_params.num_epochs - training_params.early_stopping_burnin - 1,
#             )
#             if min_ <= max_:
#                 training_params.early_stopping_patience = trial.suggest_int(
#                     "early_stopping_patience", min_, max_
#                 )
#     if training_params.reduce_lr_threshold:
#         if "reduce_lr_factor" in to_optimize:
#             training_params.reduce_lr_factor = trial.suggest_loguniform(
#                 "reduce_lr_factor", *OPTIM_BOUNDS["reduce_lr_factor"]
#             )
#         if "reduce_lr_log10_epsilon" in to_optimize:
#             training_params.reduce_lr_log10_epsilon = trial.suggest_uniform(
#                 "reduce_lr_log10_epsilon", *OPTIM_BOUNDS["reduce_lr_log10_epsilon"]
#             )
#         if "reduce_lr_burnin" in to_optimize:
#             min_, max_ = OPTIM_BOUNDS["reduce_lr_burnin"]
#             max_ = min(max_, training_params.num_epochs)
#             if min_ <= max_:
#                 training_params.reduce_lr_burnin = trial.suggest_int(
#                     "reduce_lr_burnin", min_, max_
#                 )
#         if "reduce_lr_patience" in to_optimize:
#             min_, max_ = OPTIM_BOUNDS["reduce_lr_patience"]
#             max_ = min(
#                 max_, training_params.num_epochs - training_params.reduce_lr_burnin
#             )
#             if min_ <= max_:
#                 training_params.reduce_lr_patience = trial.suggest_int(
#                     "reduce_lr_patience", min_, max_
#                 )
#         if "reduce_lr_cooldown" in to_optimize:
#             min_, max_ = OPTIM_BOUNDS["reduce_lr_cooldown"]
#             max_ = min(
#                 max_,
#                 training_params.num_epochs
#                 - training_params.reduce_lr_burnin
#                 - 2 * training_params.reduce_lr_patience,
#             )
#             if min_ <= max_:
#                 training_params.reduce_lr_cooldown = trial.suggest_int(
#                     "reduce_lr_cooldown", min_, max_
#                 )
#     if "kernel_sizes" in to_optimize:
#         num_conv = trial.suggest_int("num_conv", *OPTIM_BOUNDS["num_conv"])
#         model_params.kernel_sizes = [
#             (
#                 OPTIM_BOUNDS["kernel_time"][0],
#                 OPTIM_BOUNDS["kernel_freq"][0],
#                 OPTIM_BOUNDS["kernel_cout"][0],
#             )
#         ] * num_conv
#         optimize_kernel_sizes = True
#     else:
#         num_conv = len(model_params.kernel_sizes)
#         optimize_kernel_sizes = False
#     if "hidden_sizes" in to_optimize:
#         num_fc = trial.suggest_int("num_fc", *OPTIM_BOUNDS["num_fc"])
#         model_params.hidden_sizes = [OPTIM_BOUNDS["hidden_size"][0]] * num_fc
#         optimize_hidden_sizes = True
#     else:
#         num_fc = len(model_params.hidden_sizes)
#         optimize_hidden_sizes = False
#     if num_conv + num_fc:
#         if "nonlinearity" in to_optimize:
#             training_params.nonlinearity = trial.suggest_categorical(
#                 "nonlinearity", ("relu", "sigmoid", "tanh")
#             )
#         if "dropout_prob" in to_optimize:
#             training_params.dropout_prob = trial.suggest_uniform(
#                 "dropout_prob", *OPTIM_BOUNDS["dropout_prob"]
#             )
#     if num_conv:
#         if "flatten_style" in to_optimize:
#             model_params.flatten_style = trial.suggest_categorical(
#                 "flatten_style", ("keep_chans", "keep_both")
#             )
#         if "mellin" in to_optimize:
#             model_params.mellin = trial.suggest_categorical("mellin", (True, False))
#         if model_params.mellin and "mconv_decimation_strategy" in to_optimize:
#             model_params.mconv_decimation_strategy = trial.suggest_categorical(
#                 "mconv_decimation_strategy",
#                 ("pad-then-dec", "pad-to-dec-time-floor", "pad-to-dec-time-ceil",),
#             )
#         if "convolutional_factor_sched" in to_optimize:
#             min_, max_ = OPTIM_BOUNDS["convolutional_factor_sched"]
#             max_ = min(max_, num_conv)
#             if min_ <= max_:
#                 model_params.convolutional_factor_sched = trial.suggest_int(
#                     "convolutional_factor_sched", min_, max_
#                 )
#         if model_params.convolutional_factor_sched:
#             if "convolutional_freq_factor" in to_optimize:
#                 model_params.convolutional_freq_factor = trial.suggest_int(
#                     "convolutional_freq_factor", *OPTIM_BOUNDS["convolutional_freq_factor"]
#                 )
#             if "convolutional_time_factor" in to_optimize:
#                 model_params.convolutional_time_factor = trial.suggest_int(
#                     "convolutional_time_factor", *OPTIM_BOUNDS["convolutional_time_factor"]
#                 )
#         if "dropout2d_on_conv" in to_optimize:
#             model_params.dropout2d_on_conv = trial.suggest_categorical(
#                 "dropout2d_on_conv", *OPTIM_BOUNDS["dropout2d_on_conv"]
#             )
#     cw_size = 1 + train_params.context_left + train_params.context_right
#     if model_limit_bytes is not None:
#         bytes_estimate = models.estimate_total_size_bytes(
#             model_params, max_num_windows[train_params.batch_size], cw_size
#         )
#         if bytes_estimate > model_limit_bytes:
#             raise ValueError(
#                 "Estimate size {}B exceeds limit {}B".format(
#                     bytes_estimate, model_limit_bytes
#                 )
#             )
#     if optimize_kernel_sizes:
#         for layer_idx in range(num_conv):
#             kw, kh, co = model_params.kernel_sizes[layer_idx]
#             min_, max_ = OPTIM_BOUNDS["kernel_cout"]
#             if model_limit_bytes is not None:
#                 while True:
#                     model_params.kernel_sizes[layer_idx] = (kw, kh, max_)
#                     bytes_estimate = models.estimate_total_size_bytes(
#                         model_params, max_num_windows[train_params.batch_size], cw_size
#                     )
#                     if bytes_estimate <= model_limit_bytes:
#                         break
#                     max_ = (min_ + max_) // 2
#             co = trial.suggest_int("co_{}".format(layer_idx), min_, max_)
#             min_, max_ = OPTIM_BOUNDS["kernel_freq"]
#             if model_limit_bytes is not None:
#                 while True:
#                     model_params.kernel_sizes[layer_idx] = (kw, max_, co)
#                     bytes_estimate = models.estimate_total_size_bytes(
#                         model_params, max_num_windows[train_params.batch_size], cw_size
#                     )
#                     if bytes_estimate <= model_limit_bytes:
#                         break
#                     max_ = (min_ + max_) // 2
#             kh = trial.suggest_int("kh_{}".format(layer_idx), min_, max_)
#             min_, max_ = OPTIM_BOUNDS["kernel_time"]
#             if model_limit_bytes is not None:
#                 while True:
#                     model_params.kernel_sizes[layer_idx] = (max_, kh, co)
#                     bytes_estimate = models.estimate_total_size_bytes(
#                         model_params, max_num_windows[train_params.batch_size], cw_size
#                     )
#                     if bytes_estimate <= model_limit_bytes:
#                         break
#                     max_ = (min_ + max_) // 2
#             kw = trial.suggest_int("kw_{}".format(layer_idx), min_, max_)
#             model_params.kernel_sizes[layer_idx] = (kw, kh, co)
#     if optimize_hidden_sizes:
#         for layer_idx in range(num_fc):
#             min_, max_ = OPTIM_BOUNDS["hidden_size"]
#             if model_limit_bytes is not None:
#                 while True:
#                     model_params.hidden_sizes[layer_idx] = max_
#                     bytes_estimate = models.estimate_total_size_bytes(
#                         model_params, max_num_windows[train_params.batch_size], cw_size
#                     )
#                     if bytes_estimate <= model_limit_bytes:
#                         break
#                     max_ = (min_ + max_) // 2
#             model_params.hidden_sizes[layer_idx] = trial.suggest_int(
#                 "hidden_{}".format(layer_idx), min_, max_
#             )
