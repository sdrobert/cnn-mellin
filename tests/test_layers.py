import torch
import pytest
import cnn_mellin.layers as layers
import json
import os


with open(os.path.join(os.path.dirname(__file__), "buffers.json")) as file_:
    BUFFERS = json.load(file_)
del file_


@pytest.fixture(
    params=list(range(len(BUFFERS["mcorr1d"]))),
    ids=[f"mcorr1d-buff={x}" for x in range(len(BUFFERS["mcorr1d"]))],
)
def mcorr1d_buffers(request):
    entry = BUFFERS["mcorr1d"][request.param]
    # buffers stores in_ as (C_in, X), weight as (C_out, C_in, W), and
    # out_ as (C_out, Y)
    return (
        torch.tensor(entry["in_"]).t(),
        torch.tensor(entry["weight"]).transpose(0, 2),
        torch.tensor(entry["out"]).t(),
        entry["s"],
        entry["d"],
        entry["p"],
        entry["u"],
    )


@pytest.fixture(
    params=list(range(len(BUFFERS["mconv1d"]))),
    ids=[f"mconv1d-buff={x}" for x in range(len(BUFFERS["mconv1d"]))],
)
def mconv1d_buffers(request):
    entry = BUFFERS["mconv1d"][request.param]
    # buffers stores in_ as (C_in, X), weight as (C_out, C_in, W), and
    # out_ as (C_out, Y)
    return (
        torch.tensor(entry["in_"]).t(),
        torch.tensor(entry["weight"]).transpose(0, 2),
        torch.tensor(entry["out"]).t(),
        entry["s"],
        entry["d"],
        entry["p"],
        entry["u"],
    )


@pytest.fixture(
    params=list(range(len(BUFFERS["lcorr1d"]))),
    ids=[f"lcorr1d-buff={x}" for x in range(len(BUFFERS["lcorr1d"]))],
)
def lcorr1d_buffers(request):
    entry = BUFFERS["lcorr1d"][request.param]
    # buffers stores in_ as (C_in, X), weight as (C_out, C_in, W), and
    # out_ as (C_out, Y)
    return (
        torch.tensor(entry["in_"]).t(),
        torch.tensor(entry["weight"]).transpose(0, 2),
        torch.tensor(entry["out"]).t(),
        entry["s"],
        entry["d"],
        entry["p"],
        entry["u"],
    )


@pytest.fixture(
    params=list(range(len(BUFFERS["lconv1d"]))),
    ids=[f"lconv1d-buff={x}" for x in range(len(BUFFERS["lconv1d"]))],
)
def lconv1d_buffers(request):
    entry = BUFFERS["lconv1d"][request.param]
    # buffers stores in_ as (C_in, X), weight as (C_out, C_in, W), and
    # out_ as (C_out, Y)
    return (
        torch.tensor(entry["in_"]).t(),
        torch.tensor(entry["weight"]).transpose(0, 2),
        torch.tensor(entry["out"]).t(),
        entry["s"],
        entry["d"],
        entry["p"],
        entry["u"],
    )


@pytest.fixture(
    params=list(
        range(
            len(BUFFERS["mcorr1d"])
            + len(BUFFERS["lcorr1d"])
            + len(BUFFERS["mcorrlcorr"])
        )
    ),
    ids=[f"mcorr1d-buff={x}" for x in range(len(BUFFERS["mcorr1d"]))]
    + [f"lcorr1d-buff={x}" for x in range(len(BUFFERS["lcorr1d"]))]
    + [f"mcorrlcorr-buff={x}" for x in range(len(BUFFERS["mcorrlcorr"]))],
)
def mcorrlcorr_buffers(request):
    idx = request.param
    if idx < len(BUFFERS["mcorr1d"]):
        # buffers stores in_ as (C_in, X1), weight as (C_out, C_in, W1), and
        # out_ as (C_out, Y1)
        entry = BUFFERS["mcorr1d"][idx]
        return (
            torch.tensor(entry["in_"]).t().unsqueeze(1),
            torch.tensor(entry["weight"]).transpose(0, 2).unsqueeze(1),
            torch.tensor(entry["out"]).t().unsqueeze(1),
            (entry["s"], 1),
            (entry["d"], 1),
            (entry["p"], 0),
            (entry["u"], 1),
        )
    idx -= len(BUFFERS["mcorr1d"])
    if idx < len(BUFFERS["lcorr1d"]):
        # buffers stores in_ as (C_in, X2), weight as (C_out, C_in, W2), and
        # out_ as (C_out, Y2)
        entry = BUFFERS["lcorr1d"][idx]
        return (
            torch.tensor(entry["in_"]).t().unsqueeze(0),
            torch.tensor(entry["weight"]).transpose(0, 2).unsqueeze(0),
            torch.tensor(entry["out"]).t().unsqueeze(0),
            (1, entry["s"]),
            (1, entry["d"]),
            (0, entry["p"]),
            (1, entry["u"]),
        )
    idx -= len(BUFFERS["lcorr1d"])
    # buffers stores in_ as (C_in, X1, X2), weight as (C_out, C_in, W1, W2), and
    # out as (C_out, Y1, Y2)
    entry = BUFFERS["mcorrlcorr"][idx]
    return (
        torch.tensor(entry["in_"]).transpose(0, 1).transpose(1, 2),
        torch.tensor(entry["weight"]).transpose(0, 1).transpose(0, 2).transpose(1, 3),
        torch.tensor(entry["out"]).transpose(0, 1).transpose(1, 2),
        entry["s"],
        entry["d"],
        entry["p"],
        entry["u"],
    )


@pytest.fixture(
    params=list(
        range(
            len(BUFFERS["mconv1d"])
            + len(BUFFERS["lconv1d"])
            + len(BUFFERS["mconvlconv"])
        )
    ),
    ids=[f"mconv1d-buff={x}" for x in range(len(BUFFERS["mconv1d"]))]
    + [f"lconv1d-buff={x}" for x in range(len(BUFFERS["lconv1d"]))]
    + [f"mconvlconv-buff={x}" for x in range(len(BUFFERS["mconvlconv"]))],
)
def mconvlconv_buffers(request):
    idx = request.param
    if idx < len(BUFFERS["mconv1d"]):
        # buffers stores in_ as (C_in, X1), weight as (C_out, C_in, W1), and
        # out_ as (C_out, Y1)
        entry = BUFFERS["mconv1d"][idx]
        return (
            torch.tensor(entry["in_"]).t().unsqueeze(1),
            torch.tensor(entry["weight"]).transpose(0, 2).unsqueeze(1),
            torch.tensor(entry["out"]).t().unsqueeze(1),
            (entry["s"], 1),
            (entry["d"], 1),
            (entry["p"], 0),
            (entry["u"], 1),
        )
    idx -= len(BUFFERS["mconv1d"])
    if idx < len(BUFFERS["lconv1d"]):
        # buffers stores in_ as (C_in, X2), weight as (C_out, C_in, W2), and
        # out_ as (C_out, Y2)
        entry = BUFFERS["lconv1d"][idx]
        return (
            torch.tensor(entry["in_"]).t().unsqueeze(0),
            torch.tensor(entry["weight"]).transpose(0, 2).unsqueeze(0),
            torch.tensor(entry["out"]).t().unsqueeze(0),
            (1, entry["s"]),
            (1, entry["d"]),
            (0, entry["p"]),
            (1, entry["u"]),
        )
    idx -= len(BUFFERS["lconv1d"])
    # buffers stores in_ as (C_in, X1, X2), weight as (C_out, C_in, W1, W2), and
    # out as (C_out, Y1, Y2)
    entry = BUFFERS["mconvlconv"][idx]
    return (
        torch.tensor(entry["in_"]).transpose(0, 1).transpose(1, 2),
        torch.tensor(entry["weight"]).transpose(0, 1).transpose(0, 2).transpose(1, 3),
        torch.tensor(entry["out"]).transpose(0, 1).transpose(1, 2),
        entry["s"],
        entry["d"],
        entry["p"],
        entry["u"],
    )


@pytest.fixture(params=[1, 4], ids=("N=1", "N=4"))
def N(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=("s=1", "s=2"))
def s(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=("d=1", "d=2"))
def d(request):
    return request.param


@pytest.fixture(params=[0, 3], ids=("p=0", "p=3"))
def p(request):
    return request.param


@pytest.fixture(params=[0, 4], ids=("r=0", "r=4"))
def r(request):
    return request.param


@pytest.fixture(params=[1, 8], ids=("C_in=1", "C_in=8"))
def C_in(request):
    return request.param


@pytest.fixture(params=[1, 16], ids=("C_out=1", "C_out=16"))
def C_out(request):
    return request.param


@pytest.fixture(params=[1, 4], ids=("W=1", "W=4"))
def W(request):
    return request.param


@pytest.fixture(params=[1, 64], ids=("X=1", "X=64"))
def X(request):
    return request.param


def test_mcorr1d_direct(mcorr1d_buffers, N, device):
    in_, weight, exp_out, s, d, p, u = mcorr1d_buffers
    in_ = in_.to(device).unsqueeze(1).repeat(1, N, 1)
    weight = weight.to(device)
    exp_out = exp_out.to(device).unsqueeze(1).repeat(1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._mcorr1d(in_, weight, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_mconv1d_direct(mconv1d_buffers, N, device):
    in_, weight, exp_out, s, d, p, u = mconv1d_buffers
    in_ = in_.to(device).unsqueeze(1).repeat(1, N, 1)
    weight = weight.to(device)
    exp_out = exp_out.to(device).unsqueeze(1).repeat(1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._mconv1d(in_, weight, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_lcorr1d_direct(lcorr1d_buffers, N, device):
    in_, weight, exp_out, s, d, p, u = lcorr1d_buffers
    in_ = in_.to(device).unsqueeze(1).repeat(1, N, 1)
    weight = weight.to(device)
    exp_out = exp_out.to(device).unsqueeze(1).repeat(1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._lcorr1d(in_, weight, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_lconv1d_direct(lconv1d_buffers, N, device):
    in_, weight, exp_out, s, d, p, u = lconv1d_buffers
    in_ = in_.to(device).unsqueeze(1).repeat(1, N, 1)
    weight = weight.to(device)
    exp_out = exp_out.to(device).unsqueeze(1).repeat(1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._lconv1d(in_, weight, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_mcorrlcorr_direct(mcorrlcorr_buffers, N, device):
    in_, weight, exp_out, s, d, p, u = mcorrlcorr_buffers
    in_ = in_.to(device).unsqueeze(2).repeat(1, 1, N, 1)
    weight = weight.to(device)
    exp_out = exp_out.to(device).unsqueeze(2).repeat(1, 1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._mcorrlcorr(in_, weight, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_mconvlconv_direct(mconvlconv_buffers, N, device):
    in_, weight, exp_out, s, d, p, u = mconvlconv_buffers
    in_ = in_.to(device).unsqueeze(2).repeat(1, 1, N, 1)
    weight = weight.to(device)
    exp_out = exp_out.to(device).unsqueeze(2).repeat(1, 1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._mconvlconv(in_, weight, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_mcorr1d_gradients(N, C_out, C_in, X, W, s, d, p, r, device):
    in_ = torch.rand(
        (X, N, C_in), dtype=torch.double, device=device, requires_grad=True
    )
    weight = torch.rand(
        (W, C_in, C_out), dtype=torch.double, device=device, requires_grad=True
    )
    bias = torch.rand((C_out,), dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(
        lambda a, b, c: layers.mcorr1d(a, b, c, s, d, p, r), (in_, weight, bias)
    )


def test_mcorrlcorr_gradients(N, C_out, C_in, X, W, s, d, p, r, device):
    in_ = torch.rand(
        (X, X, N, C_in), dtype=torch.double, device=device, requires_grad=True
    )
    weight = torch.rand(
        (W, W, C_in, C_out), dtype=torch.double, device=device, requires_grad=True
    )
    bias = torch.rand((C_out,), dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(
        lambda a, b, c: layers.mcorrlcorr(a, b, c, s, d, p, r), (in_, weight, bias)
    )
