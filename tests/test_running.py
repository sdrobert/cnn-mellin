from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import pytest
import cnn_mellin.running as running
import cnn_mellin.data as data
import cnn_mellin.params as params

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def test_get_am_alignment_cross_entropy(
        temp_dir, device, populate_torch_dir, DummyAM):
    populate_torch_dir(temp_dir, 50)
    p = params.SpectDataSetParams(
        context_left=1,
        context_right=1,
        batch_size=5,
        seed=2,
        drop_last=True,
    )
    data_loader = data.EvaluationDataLoader(temp_dir, p)
    model = DummyAM(5, 11)
    loss_a = running.get_am_alignment_cross_entropy(
        model, data_loader, device=device)
    assert loss_a != 0.  # highly unlikely that it would be zero
    loss_b = running.get_am_alignment_cross_entropy(
        model, data_loader, device=device)
    assert abs(loss_a - loss_b) < 1e-5


def test_train_am_for_epoch(temp_dir, device, populate_torch_dir, DummyAM):
    populate_torch_dir(temp_dir, 50)
    spect_p = params.SpectDataSetParams(
        context_left=1,
        context_right=1,
        batch_size=5,
        seed=2,
        drop_last=True,
    )
    data_loader = data.TrainingDataLoader(temp_dir, spect_p, num_workers=4)
    train_p = params.TrainingParams(
        num_epochs=10,
        seed=3,
        dropout_prob=.5,
    )
    model = DummyAM(5, 11)
    # important! Use optimizer without history
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_a = running.train_am_for_epoch(
        model, data_loader, optimizer, train_p, device=device)
    assert loss_a != 0
    loss_b = running.train_am_for_epoch(
        model, data_loader, optimizer, train_p, device=device)
    assert loss_a > loss_b  # we learned something, maybe?
    optimizer.zero_grad()
    # important! We have to initialize parameters on the same device to get the
    # same results!
    model.cpu().reset_parameters()
    loss_c = running.train_am_for_epoch(
        model, data_loader, optimizer, train_p, epoch=0, device=device)
    assert abs(loss_a - loss_c) < 1e-5


@pytest.mark.gpu
def test_train_am_for_epoch_changing_devices(
        temp_dir, populate_torch_dir, DummyAM):
    populate_torch_dir(temp_dir, 50)
    spect_p = params.SpectDataSetParams(
        context_left=1,
        context_right=1,
        batch_size=5,
        seed=2,
        drop_last=True,
    )
    data_loader = data.TrainingDataLoader(temp_dir, spect_p, num_workers=4)
    train_p = params.TrainingParams(
        num_epochs=10,
        seed=3,
        dropout_prob=.5,
    )
    model = DummyAM(5, 11)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    running.train_am_for_epoch(
        model, data_loader, optimizer, train_p, device='cuda')
    running.train_am_for_epoch(
        model, data_loader, optimizer, train_p, device='cpu')
    running.train_am_for_epoch(
        model, data_loader, optimizer, train_p, device='cuda')
