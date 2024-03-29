import os
from typing import Optional, Union
import warnings
import gc
import tracemalloc
import warnings

from copy import deepcopy
from shutil import rmtree

import torch
import optuna
import pydrobert.param.optuna as poptuna
import pydrobert.param.serialization as serialization
import pydrobert.torch.data as data
import running
import models
import sqlalchemy.engine.url

from common import get_num_avail_cores, construct_default_param_dict


warnings.filterwarnings("ignore", category=FutureWarning)


def init_study(
    train_dir: str,
    param_dict: dict,
    out_url: Union[str, sqlalchemy.engine.url.URL],
    only: set,
    study_name: Optional[str] = None,
    device: Union[torch.device, str] = "cpu",
    dev_prop: float = 0.1,
    mem_limit: int = 10 * (1024 ** 3),
    num_data_workers: int = get_num_avail_cores() - 1,
    num_trials: Optional[int] = None,
    pruner: str = "hyperband",
) -> optuna.Study:
    only = set(only)  # non-destructive

    out_url = sqlalchemy.engine.url.make_url(out_url)
    if out_url.database is None:
        raise ValueError(f"'{out_url}' database path is invalid or empty")
    if study_name is None:
        study_name = os.path.basename(out_url.database).split(".")[0]

    device = torch.device(device)
    if device.type == "cuda":
        try:
            avail_memory = torch.cuda.get_device_properties(device).total_memory
            if avail_memory < mem_limit:
                warnings.warn(
                    f"The device {device} has {avail_memory / (1024 ** 3)} GB of memory "
                    f"but the memory limit was set to {mem_limit / (1024 ** 3)} GB. "
                    f"Consider decreasing the memory limit with --mem-limit-bytes"
                )
        except RuntimeError:
            warnings.warn(
                f"The device {device} is unavailable. Unable to verify the amount of "
                "memory on device"
            )

    # we use relpath so that the experiment will work from another machine with the
    # same directory structure. To change this after the fact, you can use the Optuna
    # CLI:
    # $ optuna study set-user-attr -h
    train_dir = os.path.relpath(train_dir)
    num_filts, num_classes = running.get_filts_and_classes(train_dir)
    raw = num_filts == 1

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

    # get some other user restrictions
    if param_dict["training"].num_epochs is None or "training.num_epochs" in only:
        max_epochs = (
            param_dict["training"].param.params()["num_epochs"].get_soft_bounds()[1]
        )
        if "training.num_epochs" not in only:
            warnings.warn(
                f"The upper bound on the number of epochs has been set to {max_epochs}."
                "If this is undesired, set training.num_epochs in config file"
            )
    else:
        max_epochs = param_dict["training"].num_epochs
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
        storage=str(out_url), study_name=study_name, load_if_exists=True
    )

    def check_and_set(key, val, force=False):
        if key in study.user_attrs and study.user_attrs[key] != val:
            if force:
                warnings.warn(
                    f"User attribute {key} had value {study.user_attrs[key]}. "
                    f"Setting to new value {val}"
                )
            else:
                raise RuntimeError(
                    f"Expected user attribute {key} to have value {val}, "
                    f"got {study.user_attrs[key]}"
                )
        study.set_user_attr(key, val)

    if raw:
        check_and_set("raw", True)
    check_and_set("max_epochs", max_epochs)
    check_and_set("device", str(device), True)
    check_and_set("data_dir", train_dir, True)
    check_and_set("train_ids", sorted(train_ids))
    check_and_set("dev_ids", sorted(dev_ids))
    check_and_set("num_classes", num_classes)
    check_and_set("max_frames", max_frames)
    check_and_set("global_dict", serialized)
    check_and_set("num_filts", num_filts)
    check_and_set("mem_limit", mem_limit)
    check_and_set("only", sorted(only))
    check_and_set("num_data_workers", num_data_workers)
    check_and_set("num_trials", num_trials, True)
    check_and_set("pruner", pruner)
    return study


def _chuck_set_from_only(only: set, set_: set, reason: str):
    for tunable in only & set_:
        warnings.warn(
            f"Removing {tunable} from list of optimized hyperparameters since {reason}"
        )
        only.remove(tunable)


def get_forward_backward_memory(
    param_dict: dict,
    num_filts: int,
    num_classes: int,
    max_frames: int,
    device: Union[str, torch.device],
    runs: int = 10,
    autocast: bool = True,
) -> int:

    device = torch.device(device)
    autocast = autocast & (device.type == "cuda")

    # do our best to stop deallocs from outside this scope being counted
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        tracemalloc.start()

    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        else:
            tracemalloc.clear_traces()

        model = models.AcousticModel(
            num_filts, num_classes + 1, param_dict["model"]
        ).to(device)
        optimizer = param_dict["training"].optimizer(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.grad_scaler.GradScaler() if autocast else None
        x = torch.empty(
            param_dict["data"].batch_size, max_frames, num_filts, device=device
        )
        len_ = torch.full((param_dict["data"].batch_size,), max_frames, device=device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(autocast):
            logits, out_len = model(
                x,
                len_,
                param_dict["training"].dropout_prob,
                param_dict["training"].convolutional_dropout_2d,
            )
            z = logits.sum()
        if autocast:
            scaler.scale(z).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            z.backward()
            optimizer.step()
        del model, optimizer, out_len, logits, x, len_, z
        if scaler is not None:
            del scaler

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    else:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak


def objective(trial: optuna.Trial, checkpoint_dir: Optional[str] = None) -> float:
    user_attrs = trial.study.user_attrs

    # for some reason this is still necessary, despite the MaxTrialsCallback
    num_trials = user_attrs["num_trials"]
    if num_trials is not None:
        num_active = sum(
            1
            for _ in trial.study.get_trials(
                deepcopy=False,
                states=(
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.RUNNING,
                ),
            )
        )
        if num_active >= num_trials:
            trial.study.stop()
            if num_active > num_trials:
                raise RuntimeError(
                    f"Cancelling trial {trial.number}: already done {num_trials}"
                )

    if checkpoint_dir is not None:
        # determine the appropriate subdirectory
        trial_no = optuna.storages.RetryFailedTrialCallback.retried_trial_number(trial)
        if trial_no is None:
            trial_no = trial.number
        checkpoint_dir = os.path.join(checkpoint_dir, f"{trial_no:07d}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    # try to make a tensor on the device. Raises a runtime error if it can't
    device = torch.device(user_attrs["device"])
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

    def pruner_callback(epoch: int, train_loss: float, dev_err: float):
        # epochs are 1-indexed; steps are 0-indexed
        trial.report(dev_err, epoch - 1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned(f"Pruned epoch {epoch}@{dev_err:.2%}")

    raise_ = None
    try:

        # check that the forward-backward memory is less than the threshold.
        val = user_attrs.get("forward_backward_memory", None)
        if val is None:
            val = get_forward_backward_memory(
                param_dict,
                user_attrs["num_filts"],
                user_attrs["num_classes"],
                user_attrs["max_frames"],
                device,
                autocast=param_dict["training"].autocast,
            )
            trial.set_user_attr("forward_backward_memory", val)
        if val > user_attrs["mem_limit"]:
            raise optuna.exceptions.TrialPruned(
                f"forward-backward footprint ({val / (1024 ** 3)} GB) exceeds "
                f"memory limit ({user_attrs['mem_limit'] / (1024 ** 3)} GB)"
            )

        _, er = running.train_am(
            param_dict["model"],
            param_dict["training"],
            param_dict["data"],
            user_attrs["data_dir"],
            user_attrs["data_dir"],
            checkpoint_dir,
            None,
            device,
            user_attrs["num_data_workers"],
            [pruner_callback],
            dev_data_params=dev_params,
        )

    except (OSError, RuntimeError) as e:  # probably an OOM
        if any(
            isinstance(x, str)
            and (
                x.find("out of memory") > -1
                or x.find("Parameter configuration yields") > -1
                or x.find("can't allocate") > -1
            )
            for x in e.args
        ):
            raise_ = e.args
        else:
            raise
    finally:
        gc.collect()

    if checkpoint_dir is not None:
        # never used again
        rmtree(checkpoint_dir, ignore_errors=True)

    if raise_:
        raise optuna.exceptions.TrialPruned(*raise_)

    return er


def get_best(study: optuna.Study, independent: bool = False) -> dict:
    user_attrs = study.user_attrs
    global_dict = construct_default_param_dict()
    serialization.deserialize_from_dict(user_attrs["global_dict"], global_dict)
    only = set(user_attrs["only"])
    trial = study.best_trial  # even if doing independent, ensures one completed trial
    if independent:
        completed_trials = study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        if not len(completed_trials):
            raise RuntimeError()
        name2value2rates = dict()
        for trial in completed_trials:
            for param_name, param_value in trial.params.items():
                name2value2rates.setdefault(param_name, dict()).setdefault(
                    param_value, []
                ).append(trial.value)
        param_dict = dict()
        for param_name in only:
            value2rates = name2value2rates.get(param_name, None)
            if value2rates is None:
                warnings.warn(
                    f"No completed trials contain a parameter value for '{param_name}'"
                )
                continue
            best_value, best_median = None, float("inf")
            for value, rates in value2rates.items():
                rates.sort()
                i = len(rates) // 2
                if len(rates) % 2:
                    median = rates[i]
                else:
                    median = (rates[i - 1] + rates[i]) / 2
                if median < best_median:
                    best_value, best_median = value, median
            param_dict[param_name] = best_value

        # manual fixes that are necessary to ensure the parameter combinations make
        # sense
        if (
            "model.convolutional_layers" in param_dict
            and "model.convolutional_factor_schedule" in param_dict
        ):
            param_dict["model.convolutional_factor_schedule"] = min(
                param_dict["model.convolutional_layers"] + 1,
                param_dict["model.convolutional_factor_schedule"],
            )

        trial = optuna.trial.FixedTrial(param_dict)
    trial.study = study
    return poptuna.suggest_param_dict(trial, global_dict, only)
