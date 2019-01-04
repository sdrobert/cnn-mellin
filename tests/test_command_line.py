from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

from collections import namedtuple
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


import pytest
import torch

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


@pytest.fixture
def DummyAM():

    class _DummyAM(torch.nn.Module):
        def __init__(self, params, *args):
            super(_DummyAM, self).__init__()
            self.params = params
            self.fc = torch.nn.Linear(params.freq_dim, params.target_dim)
            self.reset_parameters()

        def reset_parameters(self):
            if self.params.seed is not None:
                torch.manual_seed(self.params.seed)
            self.fc.reset_parameters()

        def forward(self, x):
            x = x.sum(1)
            return self.fc(x)

    import cnn_mellin.models as models
    old = models.AcousticModel
    models.AcousticModel = _DummyAM
    yield _DummyAM
    models.AcousticModel = old


@pytest.fixture
def DummyTrainingController():

    class _DummyTrainingController(object):
        def __init__(self, params, state_csv_path=None, state_dir=None):
            self.params = params
            self.state_csv_path = state_csv_path
            self.state_dir = state_dir

        def save(self, epoch, state_dict):
            pth = os.path.join(
                self.state_dir, self.params.saved_model_fmt.format(epoch=epoch)
            )
            if not os.path.isdir(self.state_dir):
                os.makedirs(self.state_dir)
            torch.save(state_dict, pth)

        def get_best_epoch(self):
            return 1

        def get_last_epoch(self):
            return 2

        def get_info(self, epoch, *default):
            return {'epoch': epoch}

    import pydrobert.torch.training as training
    old = training.TrainingStateController
    training.TrainingStateController = _DummyTrainingController
    yield _DummyTrainingController
    training.TrainingStateController = old


@pytest.mark.gpu
def test_forward_acoustic_model(
        temp_dir, populate_torch_dir, DummyAM, DummyTrainingController):
    torch.manual_seed(10)
    prior = torch.rand(11)
    prior.clamp_(1e-4)
    prior /= prior.sum()
    log_prior = torch.log(prior)
    log_prior_file = os.path.join(temp_dir, 'log_prior.pt')
    torch.save(log_prior, log_prior_file)
    pdfs_a_dir = os.path.join(temp_dir, 'pdfs')
    pdfs_b_dir = os.path.join(temp_dir, 'pdfs_')
    populate_torch_dir(temp_dir, 10, num_filts=4)
    state_dir = os.path.join(temp_dir, 'states')
    state_csv = os.path.join(temp_dir, 'hist.csv')
    c_params = namedtuple('a', 'saved_model_fmt')(
        saved_model_fmt='foo-{epoch}.pt')
    controller = DummyTrainingController(
        c_params,
        state_csv_path=state_csv,
        state_dir=state_dir
    )
    MParams = namedtuple('MP', 'freq_dim target_dim seed')
    m_params_1 = MParams(freq_dim=4, target_dim=11, seed=1)
    m_1 = DummyAM(m_params_1)
    m_params_2 = MParams(freq_dim=4, target_dim=11, seed=2)
    m_2 = DummyAM(m_params_2)
    controller.save(1, m_1.state_dict())
    controller.save(2, m_2.state_dict())
    assert os.path.isfile(os.path.join(state_dir, 'foo-1.pt'))
    ini_path = os.path.join(temp_dir, 'a.ini')
    with open(ini_path, 'w') as f:
        f.write(
            '[model]\n'
            'freq_dim = 4\n'
            'target_dim = 11\n'
            '[training]\n'
            'saved_model_fmt = foo-{epoch}.pt\n'
        )
    assert not command_line.acoustic_model_forward_pdfs([
        '--config', ini_path,
        '--device', 'cpu',
        log_prior_file,
        temp_dir,
        'path', os.path.join(state_dir, 'foo-1.pt'),
    ])
    file_names = os.listdir(pdfs_a_dir)
    assert len(file_names) == 10
    assert not command_line.acoustic_model_forward_pdfs([
        '--config', ini_path,
        '--device', 'cuda',
        '--pdfs-dir', pdfs_b_dir,
        log_prior_file,
        temp_dir,
        'history', state_dir, state_csv,
    ])
    assert len(os.listdir(pdfs_b_dir)) == 10
    for file_name in file_names:
        pdf_a = torch.load(
            os.path.join(pdfs_a_dir, file_name), map_location='cpu')
        pdf_b = torch.load(
            os.path.join(pdfs_b_dir, file_name), map_location='cpu')
        assert torch.allclose(pdf_a, pdf_b, atol=1e-4)
    assert not command_line.acoustic_model_forward_pdfs([
        '--config', ini_path,
        '--device', 'cpu',
        '--pdfs-dir', pdfs_b_dir,
        log_prior_file,
        temp_dir,
        'history', state_dir, state_csv, '--last',
    ])
    for file_name in file_names:
        pdf_a = torch.load(
            os.path.join(pdfs_a_dir, file_name), map_location='cpu')
        pdf_b = torch.load(
            os.path.join(pdfs_b_dir, file_name), map_location='cpu')
        assert not torch.allclose(pdf_a, pdf_b, atol=1e-4)


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
            'num_fc = 1\n'
            'num_conv = 0\n'
            'seed = 1\n'
            '[data]\n'
            'context_left = 1\n'
            'context_right = 1\n'
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
    populate_torch_dir(train_dir, 10, num_filts=3)
    populate_torch_dir(val_dir, 5, num_filts=3)
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
            'num_fc = 1\n'
            'num_conv = 0\n'
            'seed = 1\n'
            '[data]\n'
            'context_left = 1\n'
            'context_right = 1\n'
            '[training]\n'
            'log10_learning_rate = -2\n'
            'num_epochs = 2\n'
            'seed = 2\n'
            'weight_tensor_file = {}\n'
            'saved_model_fmt = foo-{{epoch}}.pt\n'
            'keep_last_and_best_only = false\n'
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
