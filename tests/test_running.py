import os

import pytest
import torch
import cnn_mellin.layers as layers
import cnn_mellin.models as models
import cnn_mellin.running as running
import pydrobert.torch.training as training
import pydrobert.torch.data as data
import numpy as np

from pydrobert.torch.command_line import get_torch_spect_data_dir_info


class DummyAM(torch.nn.Module):
    def __init__(
        self, freq_dim: int, target_dim: int, params: models.AcousticModelParams
    ):
        super().__init__()
        self.lift = layers.DilationLift(0)
        self.params = params
        self.freq_dim = freq_dim
        self.target_dim = target_dim
        self.dropout = 0.0
        self.fc = torch.nn.Linear(self.freq_dim, self.target_dim)

    def reset_parameters(self):
        self.lift.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, lens):
        x = self.fc(self.lift(x.transpose(0, 1)))
        return x, lens


def test_train_am_for_epoch(temp_dir, device, populate_torch_dir):
    C, V, F = 50, 10, 5
    data_dir = os.path.join(temp_dir, "data")
    state_dir = os.path.join(temp_dir, "state")
    state_csv = os.path.join(temp_dir, "state.csv")
    populate_torch_dir(data_dir, C, num_filts=F, max_class=V - 1)
    loader = data.SpectTrainingDataLoader(
        data_dir, data.SpectDataSetParams(), batch_first=True
    )
    model = DummyAM(F, V + 1, models.AcousticModelParams())
    controller = training.TrainingStateController(
        running.MyTrainingStateParams(seed=0),
        state_csv_path=state_csv,
        state_dir=state_dir,
    )
    model.to(device)
    optimizer = controller.params.optimizer(model.parameters(), lr=1e-2)
    loss_a = running.train_am_for_epoch(model, loader, optimizer, controller)
    controller.update_for_epoch(model, optimizer, loss_a, 0.0)
    loss_b = running.train_am_for_epoch(model, loader, optimizer, controller)
    assert not np.isclose(loss_a, loss_b)
    controller.update_for_epoch(model, optimizer, loss_b, 0.0)
    loss_c = running.train_am_for_epoch(model, loader, optimizer, controller, epoch=2)
    assert np.isclose(loss_b, loss_c)
    loss_d = running.train_am_for_epoch(model, loader, optimizer, controller, epoch=1)
    assert np.isclose(loss_d, loss_a)


def test_greedy_decode_am(temp_dir, device, populate_torch_dir):
    C, V, F, N1, N2 = 30, 11, 7, 1, 5
    populate_torch_dir(temp_dir, C, num_filts=F, max_class=V - 1)
    data_params = data.SpectDataSetParams(batch_size=N1, drop_last=False)
    loader = data.SpectEvaluationDataLoader(temp_dir, data_params, batch_first=True)
    model = DummyAM(F, V + 1, models.AcousticModelParams())
    model.to(device)
    er_a = running.greedy_decode_am(model, loader)
    assert not np.isclose(er_a, 0.0)
    data_params.batch_size = N2
    loader = data.SpectEvaluationDataLoader(temp_dir, data_params, batch_first=True)
    er_b = running.greedy_decode_am(model, loader)
    assert np.isclose(er_a, er_b)


@pytest.mark.parametrize("include_hist", [True, False])
def test_train_am(temp_dir, device, populate_torch_dir, include_hist):
    C, V, F, X = 20, 5, 5, 3
    train_dir = os.path.join(temp_dir, "train")
    ext_dir = os.path.join(temp_dir, "ext")
    os.makedirs(ext_dir, exist_ok=True)
    if include_hist:
        model_dir = os.path.join(temp_dir, "models")
    else:
        model_dir = None
    populate_torch_dir(train_dir, C, num_filts=F, max_class=V - 1)
    assert not get_torch_spect_data_dir_info(
        [train_dir, os.path.join(ext_dir, "train.info.ark")]
    )
    _, er_a = running.train_am(
        models.AcousticModelParams(),
        running.MyTrainingStateParams(num_epochs=X, seed=0),
        data.SpectDataSetParams(),
        train_dir,
        train_dir,
        model_dir=model_dir,
        device=device,
        num_data_workers=0,
    )
    assert not np.isclose(er_a, 0.0)
    _, er_b = running.train_am(
        models.AcousticModelParams(),
        running.MyTrainingStateParams(num_epochs=X, seed=0),
        data.SpectDataSetParams(),
        train_dir,
        train_dir,
        model_dir=model_dir,
        device=device,
        num_data_workers=0,
    )
    assert np.isclose(er_a, er_b)
