import pytest
import torch
import cnn_mellin.models as models
import numpy as np


@pytest.mark.cpu
def test_model_parameters_are_same_after_seeded_reset():
    params = models.AcousticModelParams(seed=1)
    model = models.AcousticModel(100, 100, params)
    model_params_a = [x.data.clone() for x in model.parameters()]
    torch.manual_seed(1234)
    for x in model.parameters():
        x.data.random_()
    for x, y in zip(model_params_a, model.parameters()):
        assert not torch.allclose(x, y)
    model.reset_parameters()
    for x, y in zip(model_params_a, model.parameters()):
        assert torch.allclose(x, y)


@pytest.mark.parametrize("mellin", [True, False])
@pytest.mark.parametrize("rnn", [torch.nn.LSTM, torch.nn.RNN])
@pytest.mark.parametrize("window_size", [40, 1])
@pytest.mark.parametrize("window_stride", [1, 2])
@pytest.mark.parametrize("time_factor", [1, 2])
@pytest.mark.parametrize("freq_factor", [1, 2])
@pytest.mark.parametrize("factor_sched", [1, 2])
@pytest.mark.parametrize("convolutional_layers", [0, 5])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_can_run(
    mellin,
    rnn,
    window_size,
    window_stride,
    time_factor,
    freq_factor,
    factor_sched,
    convolutional_layers,
    bidirectional,
    device,
):
    T, N, F, Y = 30, 50, 40, 10
    params = models.AcousticModelParams(
        seed=5,
        mellin=mellin,
        window_size=window_size,
        window_stride=window_stride,
        recurrent_type=rnn,
        time_factor=time_factor,
        freq_factor=freq_factor,
        factor_sched=factor_sched,
        initial_channels=3,
        channel_factor=2,
        convolutional_layers=convolutional_layers,
        bidirectional=bidirectional,
    )
    model = models.AcousticModel(F, Y, params).to(device)
    x = torch.rand((N, T, F), device=device)
    lens = torch.randint(1, T + 1, size=(N,), device=device)
    y, lens_ = model(x, lens)
    assert y.dim() == 3
    assert y.size(1) == N
    assert y.size(2) == Y
    assert lens_.dim() == 1
    assert lens_.size(0) == N
    y.sum().backward()


@pytest.mark.parametrize("window_size", [1, 10, 100], ids=("win1", "win10", "win100"))
@pytest.mark.parametrize("window_stride", [2, 13], ids=("stride2", "stride13"))
@pytest.mark.parametrize(
    "convolutional_layers", [0, 1, 3], ids=("layers0", "layers1", "layers3")
)
@pytest.mark.parametrize("mellin", [True, False], ids=("mellin", "linear"))
def test_padding_yields_same_gradients(
    window_size, window_stride, convolutional_layers, mellin, device
):
    T, N, F, Y = 95, 100, 3, 2
    params = models.AcousticModelParams(
        seed=5,
        window_size=window_size,
        window_stride=window_stride,
        convolutional_layers=convolutional_layers,
        mellin=True,
    )
    model = models.AcousticModel(F, Y, params).to(device)
    parameters = list(model.parameters())
    x = torch.rand((N, T, F), device=device, requires_grad=True)
    lens = torch.randint(1, T + 1, size=(N,), device=device)
    total_loss_1 = 0.0
    for n in range(N):
        lens_t = lens[n : n + 1]
        lens_t_ = lens_t.item()
        x_t = x[n : n + 1, :lens_t_]
        y_t, lens__t = model(x_t, lens_t)
        assert lens__t.item() == (lens_t_ - 1) // window_stride + 1
        assert y_t.size(0) == lens__t.item()
        assert y_t[lens_t:].eq(0.0).all()
        loss = y_t.sum() / (N * T)
        loss.backward()
        total_loss_1 += loss.item()
    grads = []
    for param_ in parameters:
        if param_.grad is not None:
            grads.append(param_.grad.data.clone())
            param_.grad.data.zero_()
    x.requires_grad_(True)
    y, lens_ = model(x, lens)
    assert y.size(0) == lens_.max().item()
    total_loss_2 = y.sum() / (N * T)
    assert np.isclose(total_loss_2.item(), total_loss_1)
    total_loss_2.backward()
    for param_ in parameters:
        if param_.grad is not None:
            grads_exp, grads_act = grads.pop(0), param_.grad.data
            assert grads_exp.shape == grads_act.shape
            assert torch.allclose(grads_exp, grads_act, atol=1e-5)
    assert len(grads) == 0
