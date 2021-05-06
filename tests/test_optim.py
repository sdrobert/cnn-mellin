import pytest
import optuna
import param
import pydrobert.param.optuna as poptuna
import cnn_mellin.command_line as command_line
import numpy as np


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
