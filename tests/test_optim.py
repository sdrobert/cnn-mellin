import os

import torch
import pytest
import optuna
import param
import pydrobert.param.optuna as poptuna
import cnn_mellin.command_line as command_line
import cnn_mellin.optim as optim
import numpy as np

from pydrobert.torch.command_line import get_torch_spect_data_dir_info
from pydrobert.param.serialization import deserialize_from_dict


@pytest.mark.cpu
@pytest.mark.parametrize("raw", [True, False])
def test_can_optimize_parameter_dict(raw):
    exp_dict = command_line.construct_default_param_dict()
    all_ = poptuna.get_param_dict_tunable(exp_dict)
    only = {
        "model.window_size",
        "training.optimizer",
        "data.batch_size",
        "model.convolutional_kernel_freq",
    }
    assert not (only - all_)
    del all_
    sampler = optuna.samplers.TPESampler(n_startup_trials=200, seed=0)
    study = optuna.create_study(sampler=sampler)
    if raw:
        study.set_user_attr("raw", True)
        exp_dict["model"].convolutional_kernel_freq = 1
        exp_dict["model"].window_size *= 100

    def get_loss(act_dict) -> float:
        loss = 0.0
        for tuned in only:
            dict_name, param_name = tuned.split(".")
            v1 = getattr(exp_dict[dict_name], param_name)
            v2 = getattr(act_dict[dict_name], param_name)
            entry = exp_dict[dict_name].param.params()[param_name]
            if isinstance(entry, param.Number):
                loss += (v1 - v2) ** 2 / max(v1, 1)
            elif v1 != v2:
                loss += 10
        return loss

    def objective(trial):
        param_dict = poptuna.suggest_param_dict(trial, exp_dict, only)
        return get_loss(param_dict)

    study.optimize(objective, n_trials=1000)
    best_trial = optuna.trial.FixedTrial(study.best_params)
    if raw:
        best_trial.set_user_attr("raw", True)
    best_params = poptuna.suggest_param_dict(best_trial, exp_dict, only)
    assert np.isclose(get_loss(best_params), 0.0)


@pytest.mark.cpu
@pytest.mark.parametrize("raw", [True, False])
def test_init_study(temp_dir, populate_torch_dir, raw):
    C, F, V = 300, (1 if raw else 100), 10
    train_dir = os.path.join(temp_dir, "train")
    _, refs, feat_sizes, utt_ids = populate_torch_dir(
        train_dir, C, num_filts=F, max_class=V - 1
    )
    exp_num_classes = max(x.max().item() for x in refs) + 1
    exp_max_frames = max(feat_sizes)
    ext_dir = os.path.join(temp_dir, "ext")
    os.makedirs(ext_dir)
    db_path = os.path.join(temp_dir, "optimize.db")
    db_url = f"sqlite:///{db_path}"
    assert not get_torch_spect_data_dir_info(
        [train_dir, os.path.join(ext_dir, "train.info.ark")]
    )
    global_dict = command_line.construct_default_param_dict()
    global_dict["data"].batch_size = 300
    global_dict["training"].num_epochs = 1
    only = {
        "model.window_size",
        "data.batch_size",
        "training.log10_learning_rate",
        "model.convolutional_kernel_freq",
    }
    study_1 = optim.init_study(train_dir, global_dict, db_url, only, dev_prop=0.5)
    attrs_1 = study_1.user_attrs
    if raw:
        assert "raw" in attrs_1
        assert "model.convolutional_kernel_freq" not in attrs_1["only"]
    else:
        assert "raw" not in attrs_1
        assert "model.convolutional_kernel_freq" in attrs_1["only"]
    assert exp_max_frames == attrs_1["max_frames"]
    assert len(attrs_1["train_ids"]) == C // 2
    assert len(attrs_1["dev_ids"]) == C // 2
    assert sorted(attrs_1["train_ids"] + attrs_1["dev_ids"]) == utt_ids
    assert attrs_1["num_classes"] == exp_num_classes
    assert attrs_1["num_filts"] == F
    # double-check that the serialization process worked
    deserialize_from_dict(attrs_1["global_dict"], global_dict)
    study_2 = optuna.load_study("optimize", db_url)
    assert attrs_1 == study_2.user_attrs


@pytest.mark.cpu
def test_get_forward_backward_memory():
    # this is an inherently difficult thing to check since deallocated memory is
    # subtracted from the total. We'll get a lower bound from what should definitely
    # be remaining by the backward call (excludes intermediate values)
    F, V, T, N = 30, 20, 100, 50
    param_dict = command_line.construct_default_param_dict()
    param_dict["model"].window_size = param_dict["model"].window_stride = 1
    # disable both convolutional and recurrent layers. The model should just be a
    # linear layer and change
    param_dict["model"].convolutional_layers = 0
    param_dict["model"].recurrent_layers = 0
    param_dict["training"].optimizer = torch.optim.SGD
    param_dict["data"].batch_size = N
    lower_bound = (
        (
            1  # lift tau
            + F * (V + 1)  # linear layer W
            + V  # linear layer b
            + N * T * F  # input x
            + N  # input len
            + N * T * F  # unfold x
            + N * T * F  # pack_padded
            + N * T * F  # lift x
            + T * N * (V + 1)  # output logits
            + N * T * F  # pad_packed
            + N  # output len
            + 1  # logit sum
        )
        * 2  # forward + backward
    ) * (
        4 if torch.float == torch.float32 else 8
    )  # float size
    act = optim.get_forward_backward_memory(param_dict, F, V, T)
    # FIXME(sdrobert): the lower bound is almost 3 times smaller than the actual amount
    assert lower_bound < act
    # reproducible
    assert act == optim.get_forward_backward_memory(param_dict, F, V, T)
