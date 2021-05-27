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
    # buffers stores in_ as (C_in, X), weights as (C_out, C_in, W), and
    # out_ as (C_out, Y)
    return (
        torch.tensor(entry["in_"]).t(),
        torch.tensor(entry["weights"]).transpose(0, 2),
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
    # buffers stores in_ as (C_in, X), weights as (C_out, C_in, W), and
    # out_ as (C_out, Y)
    return (
        torch.tensor(entry["in_"]).t(),
        torch.tensor(entry["weights"]).transpose(0, 2),
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
    # buffers stores in_ as (C_in, X), weights as (C_out, C_in, W), and
    # out_ as (C_out, Y)
    return (
        torch.tensor(entry["in_"]).t(),
        torch.tensor(entry["weights"]).transpose(0, 2),
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
    # buffers stores in_ as (C_in, X), weights as (C_out, C_in, W), and
    # out_ as (C_out, Y)
    return (
        torch.tensor(entry["in_"]).t(),
        torch.tensor(entry["weights"]).transpose(0, 2),
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
        # buffers stores in_ as (C_in, X1), weights as (C_out, C_in, W1), and
        # out_ as (C_out, Y1)
        entry = BUFFERS["mcorr1d"][idx]
        return (
            torch.tensor(entry["in_"]).t().unsqueeze(1),
            torch.tensor(entry["weights"]).transpose(0, 2).unsqueeze(1),
            torch.tensor(entry["out"]).t().unsqueeze(1),
            (entry["s"], 1),
            (entry["d"], 1),
            (entry["p"], 0),
            (entry["u"], 1),
        )
    idx -= len(BUFFERS["mcorr1d"])
    if idx < len(BUFFERS["lcorr1d"]):
        # buffers stores in_ as (C_in, X2), weights as (C_out, C_in, W2), and
        # out_ as (C_out, Y2)
        entry = BUFFERS["lcorr1d"][idx]
        return (
            torch.tensor(entry["in_"]).t().unsqueeze(0),
            torch.tensor(entry["weights"]).transpose(0, 2).unsqueeze(0),
            torch.tensor(entry["out"]).t().unsqueeze(0),
            (1, entry["s"]),
            (1, entry["d"]),
            (0, entry["p"]),
            (1, entry["u"]),
        )
    idx -= len(BUFFERS["lcorr1d"])
    # buffers stores in_ as (C_in, X1, X2), weights as (C_out, C_in, W1, W2), and
    # out as (C_out, Y1, Y2)
    entry = BUFFERS["mcorrlcorr"][idx]
    return (
        torch.tensor(entry["in_"]).transpose(0, 1).transpose(1, 2),
        torch.tensor(entry["weights"]).transpose(0, 1).transpose(0, 2).transpose(1, 3),
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
        # buffers stores in_ as (C_in, X1), weights as (C_out, C_in, W1), and
        # out_ as (C_out, Y1)
        entry = BUFFERS["mconv1d"][idx]
        return (
            torch.tensor(entry["in_"]).t().unsqueeze(1),
            torch.tensor(entry["weights"]).transpose(0, 2).unsqueeze(1),
            torch.tensor(entry["out"]).t().unsqueeze(1),
            (entry["s"], 1),
            (entry["d"], 1),
            (entry["p"], 0),
            (entry["u"], 1),
        )
    idx -= len(BUFFERS["mconv1d"])
    if idx < len(BUFFERS["lconv1d"]):
        # buffers stores in_ as (C_in, X2), weights as (C_out, C_in, W2), and
        # out_ as (C_out, Y2)
        entry = BUFFERS["lconv1d"][idx]
        return (
            torch.tensor(entry["in_"]).t().unsqueeze(0),
            torch.tensor(entry["weights"]).transpose(0, 2).unsqueeze(0),
            torch.tensor(entry["out"]).t().unsqueeze(0),
            (1, entry["s"]),
            (1, entry["d"]),
            (0, entry["p"]),
            (1, entry["u"]),
        )
    idx -= len(BUFFERS["lconv1d"])
    # buffers stores in_ as (C_in, X1, X2), weights as (C_out, C_in, W1, W2), and
    # out as (C_out, Y1, Y2)
    entry = BUFFERS["mconvlconv"][idx]
    return (
        torch.tensor(entry["in_"]).transpose(0, 1).transpose(1, 2),
        torch.tensor(entry["weights"]).transpose(0, 1).transpose(0, 2).transpose(1, 3),
        torch.tensor(entry["out"]).transpose(0, 1).transpose(1, 2),
        entry["s"],
        entry["d"],
        entry["p"],
        entry["u"],
    )


@pytest.fixture(params=[1, 8], ids=("N=1", "N=8"))
def N(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=("s1=1", "s1=2"))
def s1(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=("s2=1", "s2=2"))
def s2(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=("d1=1", "d1=2"))
def d1(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=("d2=1", "d2=2"))
def d2(request):
    return request.param


@pytest.fixture(params=[0, 3], ids=("p1=0", "p1=3"))
def p1(request):
    return request.param


@pytest.fixture(params=[0, 4], ids=("p2=0", "p2=4"))
def p2(request):
    return request.param


@pytest.fixture(params=[0, 4], ids=("r1=0", "r1=4"))
def r1(request):
    return request.param


@pytest.fixture(params=[0, 3], ids=("r2=0", "r2=3"))
def r1(request):
    return request.param


@pytest.fixture(params=[1, 16], ids=("C_in=1", "C_in=16"))
def C_in(request):
    return request.param


@pytest.fixture(params=[1, 32], ids=("C_out=1", "C_out=32"))
def C_out(request):
    return request.param


@pytest.fixture(params=[1, 4], ids=("W1=1", "W1=4"))
def W1(request):
    return request.param


@pytest.fixture(params=[1, 8], ids=("W2=1", "W2=8"))
def W2(request):
    return request.param


@pytest.fixture(params=[1, 128], ids=("X1=1", "X1=128"))
def X1(request):
    return request.param


@pytest.fixture(params=[1, 16], ids=("X2=1", "X2=16"))
def X2(request):
    return request.param


def test_mcorr1d_direct(mcorr1d_buffers, N, device):
    in_, weights, exp_out, s, d, p, u = mcorr1d_buffers
    in_ = in_.to(device).unsqueeze(1).repeat(1, N, 1)
    weights = weights.to(device)
    exp_out = exp_out.to(device).unsqueeze(1).repeat(1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._mcorr1d(in_, weights, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_mconv1d_direct(mconv1d_buffers, N, device):
    in_, weights, exp_out, s, d, p, u = mconv1d_buffers
    in_ = in_.to(device).unsqueeze(1).repeat(1, N, 1)
    weights = weights.to(device)
    exp_out = exp_out.to(device).unsqueeze(1).repeat(1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._mconv1d(in_, weights, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_lcorr1d_direct(lcorr1d_buffers, N, device):
    in_, weights, exp_out, s, d, p, u = lcorr1d_buffers
    in_ = in_.to(device).unsqueeze(1).repeat(1, N, 1)
    weights = weights.to(device)
    exp_out = exp_out.to(device).unsqueeze(1).repeat(1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._lcorr1d(in_, weights, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_lconv1d_direct(lconv1d_buffers, N, device):
    in_, weights, exp_out, s, d, p, u = lconv1d_buffers
    in_ = in_.to(device).unsqueeze(1).repeat(1, N, 1)
    weights = weights.to(device)
    exp_out = exp_out.to(device).unsqueeze(1).repeat(1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._lconv1d(in_, weights, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_mcorrlcorr_direct(mcorrlcorr_buffers, N, device):
    in_, weights, exp_out, s, d, p, u = mcorrlcorr_buffers
    in_ = in_.to(device).unsqueeze(2).repeat(1, 1, N, 1)
    weights = weights.to(device)
    exp_out = exp_out.to(device).unsqueeze(2).repeat(1, 1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._mcorrlcorr(in_, weights, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_mconvlconv_direct(mconvlconv_buffers, N, device):
    in_, weights, exp_out, s, d, p, u = mconvlconv_buffers
    in_ = in_.to(device).unsqueeze(2).repeat(1, 1, N, 1)
    weights = weights.to(device)
    exp_out = exp_out.to(device).unsqueeze(2).repeat(1, 1, N, 1)
    act_out = torch.zeros_like(exp_out)
    layers._mconvlconv(in_, weights, act_out, s, d, p, u)
    assert torch.allclose(exp_out, act_out)


def test_mcorr1d_gradients(N, C_out, C_in, X1, W1, s1, d1, p1, r1, device):
    in_ = torch.rand(
        (X1, N, C_in), dtype=torch.double, device=device, requires_grad=True
    )
    weights = torch.rand(
        (W1, C_in, C_out), dtype=torch.double, device=device, requires_grad=True
    )
    bias = torch.rand((C_out,), dtype=torch.double, device=device, requires_grad=True)
    torch.autograd.gradcheck(
        lambda a, b, c: layers.mcorr1d(a, b, c, s1, d1, p1, r1), (in_, weights, bias)
    )

