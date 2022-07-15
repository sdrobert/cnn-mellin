import pytest
import torch
import numpy as np
import models


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


@pytest.mark.parametrize("mellin", [True, False], ids=("mcorr", "lcorr"))
@pytest.mark.parametrize("rnn", [torch.nn.LSTM, torch.nn.RNN])
@pytest.mark.parametrize("window_size", [40, 1], ids=("W=40", "W=1"))
@pytest.mark.parametrize("window_stride", [1, 2], ids=("w=1", "w=2"))
@pytest.mark.parametrize("time_factor", [1, 2], ids=("tf=1", "tf=2"))
@pytest.mark.parametrize("factor_sched", [1, 2], ids=("fs=1", "fs=2"))
@pytest.mark.parametrize("convolutional_layers", [0, 5], ids=("cl=0", "cl=5"))
@pytest.mark.parametrize("recurrent_layers", [0, 2], ids=("rl=0", "rl=2"))
@pytest.mark.parametrize("bidirectional", [True, False], ids=("bi", "uni"))
@pytest.mark.parametrize("freq_factor,raw", [(1, True), (1, False), (2, False)])
def test_can_run(
    mellin,
    rnn,
    window_size,
    window_stride,
    time_factor,
    freq_factor,
    factor_sched,
    convolutional_layers,
    recurrent_layers,
    bidirectional,
    raw,
    device,
):
    T, N, F, Y = 30, 50, 1 if raw else 40, 10
    params = models.AcousticModelParams(
        seed=5,
        convolutional_mellin=mellin,
        window_size=window_size,
        window_stride=window_stride,
        recurrent_type=rnn,
        convolutional_time_factor=time_factor,
        convolutional_freq_factor=freq_factor,
        convolutional_factor_schedule=factor_sched,
        convolutional_initial_channels=3,
        convolutional_channel_factor=2,
        convolutional_layers=convolutional_layers,
        recurrent_layers=recurrent_layers,
        recurrent_bidirectional=bidirectional,
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


@pytest.mark.parametrize("is_2d", [True, False], ids=("2d", "1d"))
@pytest.mark.parametrize("raw", [True, False], ids=("raw", "filt"))
def test_dropout(device, raw, is_2d):
    T, N, F, Y, p = 15, 40, 1 if raw else 20, 11, 0.5
    model = models.AcousticModel(F, Y, models.AcousticModelParams()).to(device)
    x = torch.rand((N, T, F), device=device)
    lens = torch.randint(1, T + 1, size=(N,), device=device)
    torch.manual_seed(1)
    logits_a, lens_a = model(x, lens, p, is_2d)
    logits_a = torch.nn.utils.rnn.pack_padded_sequence(
        logits_a, lens_a.cpu(), enforce_sorted=False
    )
    logits_b, lens_b = model(x, lens, p, is_2d)
    logits_b = torch.nn.utils.rnn.pack_padded_sequence(
        logits_b, lens_b.cpu(), enforce_sorted=False
    )
    assert (lens_a == lens_b).all()
    assert not torch.allclose(logits_a.data, logits_b.data, atol=1e-5)
    torch.manual_seed(1)
    logits_c, lens_c = model(x, lens, p, is_2d)
    logits_c = torch.nn.utils.rnn.pack_padded_sequence(
        logits_c, lens_c.cpu(), enforce_sorted=False
    )
    assert (lens_a == lens_c).all()
    assert torch.allclose(logits_a.data, logits_c.data)


# @pytest.mark.parametrize("window_size", [100], ids=("win100",))
@pytest.mark.parametrize("window_size", [1, 10, 100], ids=("win1", "win10", "win100"))
# @pytest.mark.parametrize("window_stride", [2], ids=("stride2",))
@pytest.mark.parametrize("window_stride", [2, 13], ids=("stride2", "stride13"))
# @pytest.mark.parametrize("convolutional_layers", [3], ids=("layers3",))
@pytest.mark.parametrize(
    "convolutional_layers", [0, 1, 3], ids=("layers0", "layers1", "layers3")
)
# @pytest.mark.parametrize("mellin", [True], ids=("mellin",))
@pytest.mark.parametrize("mellin", [True, False], ids=("mellin", "linear"))
# @pytest.mark.parametrize("raw", [False], ids=("filt",))
@pytest.mark.parametrize("raw", [True, False], ids=("raw", "filt"))
def test_padding_yields_same_gradients(
    window_size, window_stride, convolutional_layers, mellin, raw, device
):
    N, F, Y = 128, 1 if raw else 16, 2
    T = 1024 // F
    params = models.AcousticModelParams(
        seed=5,
        window_size=window_size,
        window_stride=window_stride,
        convolutional_layers=convolutional_layers,
        convolutional_mellin=mellin,
    )
    model = models.AcousticModel(F, Y, params).to(device).to(torch.double)
    parameters = list(model.parameters())
    x = torch.rand((N, T, F), device=device, requires_grad=True, dtype=torch.double)
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
