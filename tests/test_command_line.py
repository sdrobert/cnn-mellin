from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import pytest
import torch

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import cnn_mellin.command_line as command_line

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def test_read_and_write_print_parameters_as_ini(capsys, temp_dir):
    assert not command_line.print_parameters_as_ini([])
    s = capsys.readouterr()
    assert s.out.find('[data]') != -1
    a_ini = s.out[s.out.find('[model]'):s.out.find('[training]')]
    # this doesn't replace the original number, just appends 1000
    a_ini = a_ini.replace('num_conv = ', 'num_conv = 1000')
    a_ini = a_ini.replace('time_factor = ', 'time_factor = 1000')
    with open(os.path.join(temp_dir, 'a.ini'), 'w') as f:
        f.write(a_ini + '\n')
    b_ini = '[model]\ntime_factor = 30\n'
    with open(os.path.join(temp_dir, 'b.ini'), 'w') as f:
        f.write(b_ini)
    command_line.print_parameters_as_ini([
        os.path.join(temp_dir, 'a.ini'),
        os.path.join(temp_dir, 'b.ini'),
    ])
    s = capsys.readouterr()
    assert s.out.find('[data]') != -1
    assert s.out.find('num_conv = 1000') != -1
    assert s.out.find('time_factor = 1000') == -1
    assert s.out.find('time_factor = 30') != -1


@pytest.mark.gpu
def test_train_acoustic_model(temp_dir, populate_torch_dir):
    torch.manual_seed(10)
    w = torch.rand(11)
    weight_pt_file = os.path.join(temp_dir, 'label_weights.pt')
    torch.save(w, weight_pt_file)
    ini_path = os.path.join(temp_dir, 'a.ini')
    with open(ini_path, 'w') as f:
        f.write(
            '[model]\n'
            'freq_dim = 3\n'
            'target_dim = 11\n'
            'seed = 1\n'
            '[data]\n'
            'context_left = 2\n'
            'context_right = 2\n'
            '[training]\n'
            'log10_learning_rate = -2\n'
            'num_epochs = 1\n'
            'seed = 2\n'
            'weight_tensor_file = {}\n'
            'saved_model_fmt = foo-{{epoch}}.pt\n'
            '[train_data]\n'
            'seed = 3\n'.format(weight_pt_file)
        )
    state_dir = os.path.join(temp_dir, 'states')
    train_dir = os.path.join(temp_dir, 'train')
    val_dir = os.path.join(temp_dir, 'dev')
    populate_torch_dir(train_dir, 50, num_filts=3)
    populate_torch_dir(val_dir, 10, num_filts=3)
    assert not command_line.train_acoustic_model([
        '--config', ini_path,
        '--device', 'cpu',
        state_dir,
        train_dir,
        val_dir,
    ])
    params_a = torch.load(
        os.path.join(state_dir, 'foo-1.pt'), map_location='cpu')
    os.remove(os.path.join(state_dir, 'foo-1.pt'))
    with open(ini_path, 'w') as f:
        f.write(
            '[model]\n'
            'freq_dim = 3\n'
            'target_dim = 11\n'
            'seed = 1\n'
            '[data]\n'
            'context_left = 2\n'
            'context_right = 2\n'
            '[training]\n'
            'log10_learning_rate = -2\n'
            'num_epochs = 2\n'
            'seed = 2\n'
            'weight_tensor_file = {}\n'
            'saved_model_fmt = foo-{{epoch}}.pt\n'
            '[train_data]\n'
            'seed = 3\n'.format(weight_pt_file)
        )
    assert not command_line.train_acoustic_model([
        '--config', ini_path,
        '--device', 'cuda',
        state_dir,
        train_dir,
        val_dir,
    ])
    params_b = torch.load(
        os.path.join(state_dir, 'foo-1.pt'), map_location='cpu')
    for key in params_a.keys():
        assert torch.allclose(params_a[key], params_b[key], atol=1e-4)
    assert os.path.isfile(os.path.join(state_dir, 'foo-2.pt'))
