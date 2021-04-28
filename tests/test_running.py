import os

import torch
import cnn_mellin.layers as layers
import cnn_mellin.models as models
import cnn_mellin.running as running
import pydrobert.torch.training as training
import pydrobert.torch.data as data
import numpy as np

# import pytest
# import pydrobert.torch.data as data
# import cnn_mellin.running as running

# __author__ = "Sean Robertson"
# __email__ = "sdrobert@cs.toronto.edu"
# __license__ = "Apache 2.0"
# __copyright__ = "Copyright 2018 Sean Robertson"


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
        x = self.fc(self.lift(x))
        return x, (lens - 1) // self.params.window_stride + 1


def test_train_am_for_epoch(temp_dir, device, populate_torch_dir):
    C, V, F = 50, 10, 5
    data_dir = os.path.join(temp_dir, "data")
    state_dir = os.path.join(temp_dir, "state")
    state_csv = os.path.join(temp_dir, "state.csv")
    populate_torch_dir(data_dir, C, num_filts=F, max_class=V - 1)
    loader = data.SpectTrainingDataLoader(
        data_dir, data.SpectDataSetParams(), batch_first=False
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
    loader = data.SpectEvaluationDataLoader(temp_dir, data_params, batch_first=False)
    model = DummyAM(F, V + 1, models.AcousticModelParams())
    model.to(device)
    er_a = running.greedy_decode_am(model, loader)
    assert not np.isclose(er_a, 0.0)
    data_params.batch_size = N2
    loader = data.SpectEvaluationDataLoader(temp_dir, data_params, batch_first=False)
    er_b = running.greedy_decode_am(model, loader)
    assert np.isclose(er_a, er_b)


# def test_train_am_for_epoch(temp_dir, device, populate_torch_dir):
#     populate_torch_dir(temp_dir, 50)
#     spect_p = data.ContextWindowDataSetParams(
#         context_left=1,
#         context_right=1,
#         batch_size=5,
#         seed=2,
#         drop_last=True,
#     )
#     data_loader = data.ContextWindowTrainingDataLoader(
#         temp_dir, spect_p, num_workers=4)
#     train_p = running.TrainingEpochParams(
#         num_epochs=10,
#         seed=3,
#         dropout_prob=.5,
#     )
#     model = DummyAM(5, 11)
#     # important! Use optimizer without history
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#     loss_a = running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device=device)
#     assert loss_a != 0
#     loss_b = running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device=device)
#     assert loss_a > loss_b  # we learned something, maybe?
#     optimizer.zero_grad()
#     # important! We have to initialize parameters on the same device to get the
#     # same results!
#     model.cpu().reset_parameters()
#     loss_c = running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, epoch=0, device=device)
#     assert abs(loss_a - loss_c) < 1e-5


# @pytest.mark.gpu
# def test_train_am_for_epoch_changing_devices(temp_dir, populate_torch_dir):
#     populate_torch_dir(temp_dir, 50)
#     spect_p = data.ContextWindowDataSetParams(
#         context_left=1,
#         context_right=1,
#         batch_size=5,
#         seed=2,
#         drop_last=True,
#     )
#     data_loader = data.ContextWindowTrainingDataLoader(
#         temp_dir, spect_p, num_workers=4)
#     model = DummyAM(5, 11)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#     train_p = running.TrainingEpochParams(seed=3)
#     running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device='cuda')
#     running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device='cpu')
#     running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device='cuda')
