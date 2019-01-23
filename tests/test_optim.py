__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

import os

import torch
import pytest
import cnn_mellin.optim as optim
import cnn_mellin.models as models
import cnn_mellin.running as running
import pydrobert.torch.data as data


@pytest.mark.parametrize('partition_style', ['round-robin', 'average', 'last'])
@pytest.mark.parametrize('val_partition', [True, False])
@pytest.mark.parametrize('partitions', [2, 3])
def test_optimize_am(
        temp_dir, populate_torch_dir, partition_style, val_partition,
        partitions, device):
    data_dir = os.path.join(temp_dir, 'data')
    history_csv = os.path.join(temp_dir, 'hist.csv')
    populate_torch_dir(data_dir, 100)
    partitions += 1 if val_partition else 0
    optim_params = optim.CNNMellinOptimParams(
        partition_style=partition_style,
        kernel_type='rbf',
        model_type='gp',
        noise_var=0.1,
        max_samples=10,
        batch_size=10,
        to_optimize=['weight_decay'],
    )
    base_model_params = models.AcousticModelParams(
        freq_dim=5,
        target_dim=11,
        num_conv=0,
        num_fc=1,
    )
    base_training_params = running.TrainingParams(
        num_epochs=1,
    )
    base_data_params = data.ContextWindowDataParams(
        context_left=0,
        context_right=0,
    )
    weight = torch.rand(11)
    model_params, training_params, data_set_params, weigh_training_samples = (
        optim.optimize_am(
            data_dir, partitions, optim_params, base_model_params,
            base_training_params, base_data_params,
            val_partition=val_partition,
            weight=weight,
            device=device,
            history_csv=history_csv,
        )
    )
    min_Y = float('inf')
    min_decay = -1.
    with open(history_csv) as f:
        for line_idx, line in enumerate(f):
            line = [x.strip() for x in line.split(',')]
            if not line_idx:
                Y_idx = line.index('Y')
                weight_decay_idx = line.index('weight_decay')
            else:
                Y, decay = float(line[Y_idx]), float(line[weight_decay_idx])
                if min_Y > Y:
                    min_Y, min_decay = Y, decay
    assert abs(min_decay - training_params.weight_decay) <= 1e-5
