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
from cnn_mellin import construct_default_param_dict


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
    global_dict = construct_default_param_dict()
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


@pytest.mark.parametrize("autocast", [True, False])
def test_get_forward_backward_memory(device, autocast):
    # this is an inherently difficult thing to check since deallocated memory is
    # subtracted from the total. We'll get a lower bound from what should definitely
    # be remaining by the backward call (excludes intermediate values)
    F, V, T, N = 30, 20, 100, 50
    param_dict = construct_default_param_dict()
    param_dict["model"].window_size = param_dict["model"].window_stride = 1
    # disable both convolutional and recurrent layers. The model should just be a
    # linear layer and change
    param_dict["model"].convolutional_layers = 0
    param_dict["model"].recurrent_layers = 0
    param_dict["training"].optimizer = torch.optim.SGD
    param_dict["data"].batch_size = N
    # FIXME(sdrobert): some lower bound would be nice
    res = optim.get_forward_backward_memory(
        param_dict, F, V, T, device, autocast=autocast
    )
    assert res == optim.get_forward_backward_memory(
        param_dict, F, V, T, device, autocast=autocast
    )


def test_objective(device, temp_dir, populate_torch_dir):
    C, T, F, V = 100, 200, 10, 5
    train_dir = os.path.join(temp_dir, "train")
    _, refs, feat_sizes, utt_ids = populate_torch_dir(
        train_dir, C, num_filts=F, max_class=V - 1
    )
    ext_dir = os.path.join(temp_dir, "ext")
    os.makedirs(ext_dir)
    db_path = os.path.join(temp_dir, "optimize.db")
    db_url = f"sqlite:///{db_path}"
    assert not get_torch_spect_data_dir_info(
        [train_dir, os.path.join(ext_dir, "train.info.ark")]
    )
    global_dict = construct_default_param_dict()
    global_dict["data"].batch_size = C // 5
    global_dict["training"].num_epochs = 2
    global_dict["model"].convolutional_layers = 0
    global_dict["model"].recurrent_layers = 0
    only = {"model.window_size"}
    study = optim.init_study(train_dir, global_dict, db_url, only, None, device, 0.5)
    study.optimize(optim.objective, 5)
    assert len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])) == 5


def test_get_best(temp_dir, populate_torch_dir):
    C, T, F, V = 5, 10, 10, 5
    train_dir = os.path.join(temp_dir, "train")
    _, refs, feat_sizes, utt_ids = populate_torch_dir(
        train_dir, C, num_filts=F, max_class=V - 1
    )
    ext_dir = os.path.join(temp_dir, "ext")
    os.makedirs(ext_dir)
    db_path = os.path.join(temp_dir, "optimize.db")
    db_url = f"sqlite:///{db_path}"
    assert not get_torch_spect_data_dir_info(
        [train_dir, os.path.join(ext_dir, "train.info.ark")]
    )
    global_dict = construct_default_param_dict()
    global_dict["data"].batch_size = C // 5
    global_dict["training"].num_epochs = 2
    global_dict["model"].convolutional_layers = 0
    global_dict["model"].recurrent_layers = 0
    only = {"model.window_size"}
    study = optim.init_study(train_dir, global_dict, db_url, only, None, "cpu", 0.5)

    def objective(trial: optuna.Trial) -> float:
        param_dict = poptuna.suggest_param_dict(trial, global_dict, only)
        return (param_dict["model"].window_size - 5) ** 2

    study.optimize(objective, 1000)

    param_dict = optim.get_best(study)

    assert param_dict["model"].window_size == 5
