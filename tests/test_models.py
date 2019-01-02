__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"

import pytest
import torch
import cnn_mellin.models as models


def test_model_parameters_are_same_after_seeded_reset():
    params = models.AcousticModelParams(seed=1, freq_dim=10, target_dim=10)
    model = models.AcousticModel(params, 5)
    model_params_a = [x.data.clone() for x in model.parameters()]
    torch.manual_seed(1234)
    for x in model.parameters():
        x.data.random_()
    for x, y in zip(model_params_a, model.parameters()):
        assert not torch.allclose(x, y)
    model.reset_parameters()
    for x, y in zip(model_params_a, model.parameters()):
        assert torch.allclose(x, y)


@pytest.mark.parametrize('mellin,mconv_decimation_strategy', [
    (False, 'pad-then-dec'),
    (True, 'pad-then-dec'),
    (True, 'pad-to-dec-time-floor'),
    (True, 'pad-to-dec-time-ceil'),
])
@pytest.mark.parametrize('kernel', [1, 2, 3])
@pytest.mark.parametrize('window', [1, 2, 3])
@pytest.mark.parametrize('freq_dim', [1, 2, 10])
@pytest.mark.parametrize('target_dim', [1, 5])
@pytest.mark.parametrize('num_conv', [0, 1, 3])
@pytest.mark.parametrize('num_fc', [1, 2])
@pytest.mark.parametrize(
    'flatten_style', ['keep_filts', 'keep_chans', 'keep_both'])
def test_model_forward_backward_works(
        mellin, mconv_decimation_strategy, flatten_style,
        kernel, window, freq_dim, target_dim, num_conv, num_fc):
    params = models.AcousticModelParams(
        seed=1, mellin=mellin, kernel_time=kernel, kernel_freq=kernel,
        target_dim=target_dim, num_conv=num_conv, num_fc=num_fc,
        mconv_decimation_strategy=mconv_decimation_strategy, freq_dim=freq_dim,
        factor_sched=2, freq_factor=2, flatten_style=flatten_style,
    )
    model = models.AcousticModel(params, window)
    torch.manual_seed(10)
    x = torch.rand(3, window, freq_dim)
    y = model(x)
    assert y.size() == (3, target_dim)
    y = y.sum()
    y.backward()


@pytest.mark.gpu
@pytest.mark.parametrize('mellin', [True, False])
def test_model_to_gpu(mellin):
    params = models.AcousticModelParams(
        seed=2, mellin=mellin, target_dim=10, num_conv=1, num_fc=2,
        freq_dim=10,
    )
    model = models.AcousticModel(params, 5).cuda()
    torch.manual_seed(5)
    x = torch.rand(2, 5, 10).cuda()
    y = model(x)
