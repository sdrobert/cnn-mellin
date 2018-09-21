from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import cnn_mellin.running as running
import cnn_mellin.data as data
import cnn_mellin.params as params

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


class DummyAM(torch.nn.Module):
    def __init__(self, num_filts, num_classes):
        super(DummyAM, self).__init__()
        self.fc = torch.nn.Linear(num_filts, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x.sum(1)  # sum out the context window


def test_get_am_alignment_cross_entropy(temp_dir, device, populate_torch_dir):
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
        model, data_loader, cuda=device == 'cuda')
    assert loss_a != 0.  # highly unlikely that it would be zero
    loss_b = running.get_am_alignment_cross_entropy(
        model, data_loader, cuda=device == 'cuda')
    assert abs(loss_a - loss_b) < 1e-5
