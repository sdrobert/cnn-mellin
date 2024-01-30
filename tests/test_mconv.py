"""Test the pytorch implementation of mellin kernels"""

import os
import re

import pytest
import torch
import mconv


def _find_next_match(fp, matches):
    # consume the buffer until the next match and return a tuple of a string
    # of the values before, and the match itself
    buff = ""
    while True:
        nxt = fp.read(1)
        if not nxt:
            return buff, ""
        buff += nxt
        for match in matches:
            if buff.endswith(match):
                return buff[: -len(match)], match


def _get_entries_from_test_file(path, entry_names):
    # this is a really simple and easily broken parser of the
    # test_x_buffers.hpp files from the parent C++ library.
    pattern = re.compile(r"(/\*.*?\*/|//.*?(\n|$))")
    fp = open(path)
    entries = dict()
    while True:
        _, entry = _find_next_match(fp, entry_names)
        if not entry:
            return entries
        if entry in entries:
            raise IOError("Duplicate entries for {}".format(entry))
        _, paren = _find_next_match(fp, "{")
        if not paren:
            raise IOError("Found no open paren after {}".format(entry))
        val = []
        stack = [val]
        while stack:
            buff, paren = _find_next_match(fp, "}{")
            if not paren:
                raise IOError("Found no matching close paren for {}".format(entry))
            else:
                last = stack[-1]
                # get rid of /* */ comments
                buff = pattern.sub("", buff)
                buff = (x.strip() for x in buff.split(","))
                buff = (x.split("/") for x in buff if x)
                buff = [
                    float(x[0])
                    if len(x) == 1
                    else float(x[0].strip()) / float(x[1].strip())
                    for x in buff
                ]
                last += buff
                if paren == "{":
                    next = []
                    last.append(next)
                    stack.append(next)
                else:
                    stack.pop()
        entries[entry] = val


_1D_ENTRIES = {
    "kF",
    "kG",
    "kH",
    "kNF",
    "kNG",
    "kCIn",
    "kCOut",
    "kS",
    "kD",
    "kP",
    "kU",
}

_2D_ENTRIES = {
    "kF",
    "kG",
    "kH",
    "kNFX",
    "kNFY",
    "kNGX",
    "kNGY",
    "kCIn",
    "kCOut",
    "kSX",
    "kSY",
    "kDX",
    "kDY",
    "kPX",
    "kPY",
    "kUX",
    "kUY",
}


_MCONV1D_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_mconv1d_buffers.h")
    ),
    _1D_ENTRIES,
)

_MCORR1D_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_mcorr1d_buffers.h")
    ),
    _1D_ENTRIES,
)


_LCONV1D_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_lconv1d_buffers.h")
    ),
    _1D_ENTRIES,
)

_LCORR1D_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_lcorr1d_buffers.h")
    ),
    _1D_ENTRIES,
)


_MCONVLCONV_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_mconvlconv_buffers.h")
    ),
    _2D_ENTRIES,
)

_MCORRLCORR_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_mcorrlcorr_buffers.h")
    ),
    _2D_ENTRIES,
)


_SND2COL_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_snd2col_buffers.h")
    ),
    _1D_ENTRIES,
)

_COL2SND_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_col2snd_buffers.h")
    ),
    _1D_ENTRIES,
)


_LIN2COL_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_lin2col_buffers.h")
    ),
    _1D_ENTRIES,
)
_COL2LIN_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_col2lin_buffers.h")
    ),
    _1D_ENTRIES,
)


_SPEC2COL_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_spec2col_buffers.h")
    ),
    _2D_ENTRIES,
)
_COL2SPEC_BUFFER_ENTRIES = _get_entries_from_test_file(
    os.path.abspath(
        os.path.join(__file__, "..", "..", "c", "tests", "test_col2spec_buffers.h")
    ),
    _2D_ENTRIES,
)


def _mconv_nh(nf, ng, s, d, p, u):
    return ((nf + d - 1) * (ng + u - 1) - 1) // (p + 1) - s + 2


def _mcorr_nh(nf, ng, s, d, p, u):
    return (p + 1) * (ng + u - 1) // d - (s - 1)


def _lconv_nh(nf, ng, s, d, p, u):
    return (d * (nf - 1) + u * (ng - 1) - p) // s + 1


def _lcorr_nh(nf, ng, s, d, p, u):
    return (u * (ng - 1) + p) // s + 1


def _1d_buffers(idx, entries, nh_func):
    f = torch.tensor(entries["kF"][idx], dtype=torch.float) if "kF" in entries else None
    g = torch.tensor(entries["kG"][idx], dtype=torch.float)
    h = torch.tensor(entries["kH"][idx], dtype=torch.float)
    nf = int(entries["kNF"][idx])
    ng = int(entries["kNG"][idx])
    c_in = int(entries["kCIn"][idx])
    c_out = int(entries["kCOut"][idx]) if "kCOut" in entries else None
    s = int(entries["kS"][idx])
    d = int(entries["kD"][idx])
    p = int(entries["kP"][idx])
    u = int(entries["kU"][idx])
    nh = nh_func(nf, ng, s, d, p, u)
    f = torch.empty((1, c_in, nf)) if f is None else f.expand(c_out, c_in, nf)
    g = g.reshape(1, c_in, ng)
    h = h.reshape((1, c_in, nf, nh) if c_out is None else (1, c_out, nh))
    return f, g, h, s, d, p, u


def _2d_buffers(idx, entries, nhx_func, nhy_func):
    f = torch.tensor(entries["kF"][idx], dtype=torch.float) if "kF" in entries else None
    g = torch.tensor(entries["kG"][idx], dtype=torch.float)
    h = torch.tensor(entries["kH"][idx], dtype=torch.float)
    nfx = int(entries["kNFX"][idx])
    nfy = int(entries["kNFY"][idx])
    ngx = int(entries["kNGX"][idx])
    ngy = int(entries["kNGY"][idx])
    c_in = int(entries["kCIn"][idx])
    c_out = int(entries["kCOut"][idx]) if "kCOut" in entries else None
    sx = int(entries["kSX"][idx])
    sy = int(entries["kSY"][idx])
    dx = int(entries["kDX"][idx])
    dy = int(entries["kDY"][idx])
    px = int(entries["kPX"][idx])
    py = int(entries["kPY"][idx])
    ux = int(entries["kUX"][idx])
    uy = int(entries["kUY"][idx])
    nhx = nhx_func(nfx, ngx, sx, dx, px, ux)
    nhy = nhy_func(nfy, ngy, sy, dy, py, uy)
    f = (
        torch.empty((1, c_in, nfx, nfy))
        if f is None
        else f.reshape(c_out, c_in, nfx, nfy)
    )
    g = g.reshape(1, c_in, ngx, ngy)
    if c_out is None:
        h = h.reshape(1, c_in, nfx, nfy, nhx, nhy)
    else:
        h = h.reshape(1, c_out, nhx, nhy)
    return f, g, h, (sx, sy), (dx, dy), (px, py), (ux, uy)


# def _make_json():
#     root_dict = dict()
#     for name, entries, nhx_func, nhy_func in (
#         ("mconv1d", _MCONV1D_BUFFER_ENTRIES, _mconv_nh, None),
#         ("mcorr1d", _MCORR1D_BUFFER_ENTRIES, _mcorr_nh, None),
#         ("lconv1d", _LCONV1D_BUFFER_ENTRIES, _lconv_nh, None),
#         ("lcorr1d", _LCORR1D_BUFFER_ENTRIES, _lcorr_nh, None),
#         ("mconvlconv", _MCONVLCONV_BUFFER_ENTRIES, _mconv_nh, _lconv_nh),
#         ("mcorrlcorr", _MCORRLCORR_BUFFER_ENTRIES, _mcorr_nh, _lcorr_nh),
#     ):
#         op_dict = root_dict.setdefault(name, [])
#         for idx in range(len(entries["kF"])):
#             if nhy_func is None:
#                 x = _1d_buffers(idx, entries, nhx_func)
#             else:
#                 x = _2d_buffers(idx, entries, nhx_func, nhy_func)
#             op_dict.append(
#                 {
#                     "f": x[0].tolist(),
#                     "g": x[1].tolist(),
#                     "h": x[2].tolist(),
#                     "s": x[3],
#                     "d": x[4],
#                     "p": x[5],
#                     "u": x[6],
#                 }
#             )
#     with open("test.json", "w") as fp:
#         json.dump(root_dict, fp, indent=1)


# _make_json()


@pytest.fixture(
    scope="session", params=list(range(len(_MCONV1D_BUFFER_ENTRIES["kG"]))),
)
def mconv1d_buffers(request):
    return _1d_buffers(request.param, _MCONV1D_BUFFER_ENTRIES, _mconv_nh)


@pytest.fixture(
    scope="session", params=list(range(len(_MCORR1D_BUFFER_ENTRIES["kG"]))),
)
def mcorr1d_buffers(request):
    return _1d_buffers(request.param, _MCORR1D_BUFFER_ENTRIES, _mcorr_nh)


@pytest.fixture(
    scope="session", params=list(range(len(_LCONV1D_BUFFER_ENTRIES["kG"]))),
)
def lconv1d_buffers(request):
    return _1d_buffers(request.param, _LCONV1D_BUFFER_ENTRIES, _lconv_nh)


@pytest.fixture(
    scope="session", params=list(range(len(_LCORR1D_BUFFER_ENTRIES["kG"]))),
)
def lcorr1d_buffers(request):
    return _1d_buffers(request.param, _LCORR1D_BUFFER_ENTRIES, _lcorr_nh)


@pytest.fixture(
    scope="session", params=list(range(len(_SND2COL_BUFFER_ENTRIES["kG"]))),
)
def snd2col_buffers(request):
    return _1d_buffers(request.param, _SND2COL_BUFFER_ENTRIES, _mcorr_nh)


@pytest.fixture(
    scope="session", params=list(range(len(_COL2SND_BUFFER_ENTRIES["kG"]))),
)
def col2snd_buffers(request):
    return _1d_buffers(request.param, _COL2SND_BUFFER_ENTRIES, _mcorr_nh)


@pytest.fixture(
    scope="session", params=list(range(len(_LIN2COL_BUFFER_ENTRIES["kG"]))),
)
def lin2col_buffers(request):
    return _1d_buffers(request.param, _LIN2COL_BUFFER_ENTRIES, _lcorr_nh)


@pytest.fixture(
    scope="session", params=list(range(len(_COL2LIN_BUFFER_ENTRIES["kG"]))),
)
def col2lin_buffers(request):
    return _1d_buffers(request.param, _COL2LIN_BUFFER_ENTRIES, _lcorr_nh)


@pytest.fixture(
    scope="session",
    params=list(
        range(
            len(_MCONV1D_BUFFER_ENTRIES["kF"])
            + len(_LCONV1D_BUFFER_ENTRIES["kF"])
            + len(_MCONVLCONV_BUFFER_ENTRIES["kF"])
        )
    ),
    ids=["{} (mconv1d)".format(x) for x in range(len(_MCONV1D_BUFFER_ENTRIES["kF"]))]
    + ["{} (lconv1d)".format(x) for x in range(len(_LCONV1D_BUFFER_ENTRIES["kF"]))]
    + [
        "{} (mconvlconv)".format(x)
        for x in range(len(_MCONVLCONV_BUFFER_ENTRIES["kF"]))
    ],
)
def mconvlconv_buffers(request):
    idx = request.param
    if idx < len(_MCONV1D_BUFFER_ENTRIES["kF"]):
        f, g, h, sx, dx, px, ux = _1d_buffers(idx, _MCONV1D_BUFFER_ENTRIES, _mconv_nh)
        f = f[..., None]
        g = g[..., None]
        h = h[..., None]
        return f, g, h, (sx, 1), (dx, 1), (px, 0), (ux, 1)
    idx -= len(_MCONV1D_BUFFER_ENTRIES["kF"])
    if idx < len(_LCONV1D_BUFFER_ENTRIES["kF"]):
        f, g, h, sy, dy, py, uy = _1d_buffers(idx, _LCONV1D_BUFFER_ENTRIES, _lconv_nh)
        f = f[..., None, :]
        g = g[..., None, :]
        h = h[..., None, :]
        return f, g, h, (1, sy), (1, dy), (0, py), (1, uy)
    idx -= len(_LCONV1D_BUFFER_ENTRIES["kF"])
    return _2d_buffers(idx, _MCONVLCONV_BUFFER_ENTRIES, _mconv_nh, _lconv_nh)


@pytest.fixture(
    scope="session",
    params=list(
        range(
            len(_MCORR1D_BUFFER_ENTRIES["kF"])
            + len(_LCORR1D_BUFFER_ENTRIES["kF"])
            + len(_MCORRLCORR_BUFFER_ENTRIES["kF"])
        )
    ),
    ids=["{} (mcorr1d)".format(x) for x in range(len(_MCORR1D_BUFFER_ENTRIES["kF"]))]
    + ["{} (lcorr1d)".format(x) for x in range(len(_LCORR1D_BUFFER_ENTRIES["kF"]))]
    + [
        "{} (mcorrlcorr)".format(x)
        for x in range(len(_MCORRLCORR_BUFFER_ENTRIES["kF"]))
    ],
)
def mcorrlcorr_buffers(request):
    idx = request.param
    if idx < len(_MCORR1D_BUFFER_ENTRIES["kF"]):
        f, g, h, sx, dx, px, ux = _1d_buffers(idx, _MCORR1D_BUFFER_ENTRIES, _mcorr_nh)
        f = f[..., None]
        g = g[..., None]
        h = h[..., None]
        return f, g, h, (sx, 1), (dx, 1), (px, 0), (ux, 1)
    idx -= len(_MCORR1D_BUFFER_ENTRIES["kF"])
    if idx < len(_LCORR1D_BUFFER_ENTRIES["kF"]):
        f, g, h, sy, dy, py, uy = _1d_buffers(idx, _LCORR1D_BUFFER_ENTRIES, _lcorr_nh)
        f = f[..., None, :]
        g = g[..., None, :]
        h = h[..., None, :]
        return f, g, h, (1, sy), (1, dy), (0, py), (1, uy)
    idx -= len(_LCORR1D_BUFFER_ENTRIES["kF"])
    return _2d_buffers(idx, _MCORRLCORR_BUFFER_ENTRIES, _mcorr_nh, _lcorr_nh)


@pytest.fixture(
    scope="session",
    params=list(
        range(
            len(_SND2COL_BUFFER_ENTRIES["kG"])
            + len(_LIN2COL_BUFFER_ENTRIES["kG"])
            + len(_SPEC2COL_BUFFER_ENTRIES["kG"])
        )
    ),
    ids=["{} (snd2col)".format(x) for x in range(len(_SND2COL_BUFFER_ENTRIES["kG"]))]
    + ["{} (lin2col)".format(x) for x in range(len(_LIN2COL_BUFFER_ENTRIES["kG"]))]
    + ["{} (spec2col)".format(x) for x in range(len(_SPEC2COL_BUFFER_ENTRIES["kG"]))],
)
def spec2col_buffers(request):
    idx = request.param
    if idx < len(_SND2COL_BUFFER_ENTRIES["kG"]):
        f, g, h, sx, dx, px, ux = _1d_buffers(idx, _SND2COL_BUFFER_ENTRIES, _mcorr_nh)
        f = f[..., None]
        g = g[..., None]
        h = h[..., None, :, None]
        return f, g, h, (sx, 1), (dx, 1), (px, 0), (ux, 1)
    idx -= len(_SND2COL_BUFFER_ENTRIES["kG"])
    if idx < len(_LIN2COL_BUFFER_ENTRIES["kG"]):
        f, g, h, sy, dy, py, uy = _1d_buffers(idx, _LIN2COL_BUFFER_ENTRIES, _lcorr_nh)
        f = f[..., None, :]
        g = g[..., None, :]
        h = h[..., None, :, None, :]
        return f, g, h, (1, sy), (1, dy), (0, py), (1, uy)
    idx -= len(_LIN2COL_BUFFER_ENTRIES["kG"])
    return _2d_buffers(idx, _SPEC2COL_BUFFER_ENTRIES, _mcorr_nh, _lcorr_nh)


@pytest.fixture(
    scope="session",
    params=list(
        range(
            len(_COL2SND_BUFFER_ENTRIES["kG"])
            + len(_COL2LIN_BUFFER_ENTRIES["kG"])
            + len(_COL2SPEC_BUFFER_ENTRIES["kG"])
        )
    ),
    ids=["{} (col2snd)".format(x) for x in range(len(_COL2SND_BUFFER_ENTRIES["kG"]))]
    + ["{} (col2lin)".format(x) for x in range(len(_COL2LIN_BUFFER_ENTRIES["kG"]))]
    + ["{} (col2spec)".format(x) for x in range(len(_COL2SPEC_BUFFER_ENTRIES["kG"]))],
)
def col2spec_buffers(request):
    idx = request.param
    if idx < len(_COL2SND_BUFFER_ENTRIES["kG"]):
        f, g, h, sx, dx, px, ux = _1d_buffers(idx, _COL2SND_BUFFER_ENTRIES, _mcorr_nh)
        f = f[..., None]
        g = g[..., None]
        h = h[..., None, :, None]
        return f, g, h, (sx, 1), (dx, 1), (px, 0), (ux, 1)
    idx -= len(_COL2SND_BUFFER_ENTRIES["kG"])
    if idx < len(_COL2LIN_BUFFER_ENTRIES["kG"]):
        f, g, h, sy, dy, py, uy = _1d_buffers(idx, _COL2LIN_BUFFER_ENTRIES, _lcorr_nh)
        f = f[..., None, :]
        g = g[..., None, :]
        h = h[..., None, :, None, :]
        return f, g, h, (1, sy), (1, dy), (0, py), (1, uy)
    idx -= len(_COL2LIN_BUFFER_ENTRIES["kG"])
    return _2d_buffers(idx, _COL2SPEC_BUFFER_ENTRIES, _mcorr_nh, _lcorr_nh)


@pytest.fixture(params=[1, 2, 5], ids=["N=1", "N=2", "N=5"], scope="session")
def batch(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("ext_ops", marks=pytest.mark.ext_ops),
        pytest.param("python_ops", marks=pytest.mark.python_ops),
    ],
    scope="session",
)
def mcorr1d(request):
    if request.param == "python_ops":
        op = mconv._native_mcorr1d
    elif request.param == "ext_ops":
        op = mconv._ext_mcorr1d
        if op is None:
            pytest.skip("Extension not available")
    else:
        raise NotImplementedError
    return op


@pytest.fixture(
    params=[
        pytest.param("ext_ops", marks=pytest.mark.ext_ops),
        pytest.param("python_ops", marks=pytest.mark.python_ops),
    ],
    scope="session",
)
def mcorrlcorr(request):
    if request.param == "python_ops":
        op = mconv._native_mcorrlcorr
    elif request.param == "ext_ops":
        op = mconv._ext_mcorrlcorr
        if op is None:
            pytest.skip("Extension not available")
    else:
        raise NotImplementedError
    return op


@pytest.fixture(
    params=[
        pytest.param("ext_ops", marks=pytest.mark.ext_ops),
        pytest.param("python_ops", marks=pytest.mark.python_ops),
    ],
    scope="session",
)
def snd2col(request):
    if request.param == "python_ops":
        return mconv._native_snd2col
    elif request.param == "ext_ops":
        op = mconv._ext_snd2col
        if op is None:
            pytest.skip("Extension not available")
    else:
        raise NotImplementedError
    return op


@pytest.fixture(
    params=[
        pytest.param("ext_ops", marks=pytest.mark.ext_ops),
        pytest.param("python_ops", marks=pytest.mark.python_ops),
    ],
    scope="session",
)
def spec2col(request):
    if request.param == "python_ops":
        return mconv._native_spec2col
    elif request.param == "ext_ops":
        op = mconv._ext_spec2col
        if op is None:
            pytest.skip("Extension not available")
    else:
        raise NotImplementedError
    return op


def _add_bias(h):
    b = h.new(h.size()[1])
    b.random_()
    b_exp = b.unsqueeze(0)
    for _ in range(len(h.size()) - 2):
        b_exp = b_exp.unsqueeze(-1)
    b_exp = b_exp.expand_as(h)
    h = h + b_exp
    return h, b


def _test_values(f, g, h_exp, op, *op_args, is_col=False):
    ndim = f.dim() - 2
    if is_col:
        f = f.shape[2:]
        if ndim == 1:
            f = f[0]
    is_col = int(is_col)
    nh = h_exp.size()[-ndim:]
    # NB: op args are (input, weights, ...)
    h_1 = op(g, f, *op_args)
    r = tuple(nh[i] - h_1.size(-ndim + i) for i in range(ndim))
    if ndim == 1:
        r = r[0]
    if len(op_args) == 0 - is_col:
        op_args += (None,)  # bias
    if len(op_args) == 1 - is_col:
        op_args += (1,) * ndim  # s
    if len(op_args) == 2 - is_col:
        op_args += (1,) * ndim  # d
    if len(op_args) == 3 - is_col:
        op_args += (0,) * ndim  # p
    if len(op_args) == 4 - is_col:
        op_args += (1,) * ndim  # u
    op_args += (r,)
    h_act = op(g, f, *op_args)
    assert h_act.size() == h_exp.size()
    assert torch.allclose(h_exp, h_act)


@pytest.mark.check_values
@pytest.mark.parametrize("bias", [True, False])
def test_mcorr1d_values(mcorr1d, mcorr1d_buffers, bias):
    f, g, h_exp, s, d, p, u = mcorr1d_buffers
    if bias:
        h_exp, b = _add_bias(h_exp)
    else:
        b = None
    _test_values(f, g, h_exp, mcorr1d, b, s, d, p, u)


@pytest.mark.check_values
@pytest.mark.parametrize("bias", [True, False])
def test_mcorrlcorr_values(mcorrlcorr, mcorrlcorr_buffers, bias):
    f, g, h_exp, s, d, p, u = mcorrlcorr_buffers
    if bias:
        h_exp, b = _add_bias(h_exp)
    else:
        b = None
    _test_values(f, g, h_exp, mcorrlcorr, b, s, d, p, u)


@pytest.mark.check_values
def test_snd2col_values(snd2col, snd2col_buffers):
    f, g, h_exp, s, d, p, u = snd2col_buffers
    _test_values(f, g, h_exp, snd2col, s, d, p, u, is_col=True)


def test_mcorr1d_impl(device):
    torch.manual_seed(10)
    batch, c_out, c_in, nf, ng, s, d, p, r = 128, 8, 16, 4, 12, 1, 2, 3, 4
    f = torch.rand((c_out, c_in, nf), device=device)
    g = torch.rand((batch, c_in, ng), device=device)
    b = torch.rand((c_out,), device=device)
    h_exp = mconv.mcorr1d(g, f, b, s, d, p, r, _impl="direct")
    h_act = mconv.mcorr1d(g, f, b, s, d, p, r, _impl="mm")
    assert h_exp.size() == h_act.size()
    assert torch.allclose(h_exp, h_act)


@pytest.mark.check_values
def test_spec2col_values(spec2col, spec2col_buffers):
    f, g, h_exp, s, d, p, u = spec2col_buffers
    _test_values(f, g, h_exp, spec2col, s, d, p, u, is_col=True)


def test_mcorrlcorr_impl(device):
    torch.manual_seed(20)
    batch, c_out, c_in, nf, ng = 64, 16, 8, (32, 32), (8, 8)
    s, d, p, r = (1, 2), (2, 1), (3, 3), (1, 1)
    f = torch.rand((c_out, c_in) + nf, device=device)
    g = torch.rand((batch, c_in) + ng, device=device)
    b = torch.rand((c_out,), device=device)
    h_exp = mconv.mcorrlcorr(g, f, b, s, d, p, r, _impl="direct")
    h_act = mconv.mcorrlcorr(g, f, b, s, d, p, r, _impl="mm")
    assert h_exp.size() == h_act.size()
    assert torch.allclose(h_exp, h_act)


@pytest.mark.check_gradients
@pytest.mark.parametrize(
    "f_shape,g_shape",
    [
        ((1, 1, 1), (1, 1, 2)),
        pytest.param((2, 2, 2), (1, 2, 1), marks=pytest.mark.check_small_gradients),
        ((5, 3, 10), (50, 3, 15)),
    ],
)
@pytest.mark.parametrize("s", [1, 2], ids=("s=1", "s=2"))
@pytest.mark.parametrize("d", [1, 2], ids=("d=1", "d=2"))
@pytest.mark.parametrize("p", [3, 4], ids=("p=3", "p=4"))
@pytest.mark.parametrize("u", [1, 2], ids=("u=1", "u=2"))
def test_mcorr1d_gradients(device, mcorr1d, f_shape, g_shape, s, d, p, u):
    torch.manual_seed(1)
    f = torch.randn(*f_shape, device=device, dtype=torch.double, requires_grad=True)
    g = torch.randn(*g_shape, device=device, dtype=torch.double, requires_grad=True)
    b = torch.randn(f_shape[0], device=device, dtype=torch.double, requires_grad=True)
    torch.autograd.gradcheck(
        lambda g, f, b: mcorr1d(g, f, b, s, d, p, u), (g, f, b),
    )


@pytest.mark.check_gradients
@pytest.mark.parametrize(
    "f_shape,g_shape",
    [
        ((1, 1, 1, 1), (1, 1, 2, 1)),
        pytest.param(
            (2, 2, 2, 2), (1, 2, 1, 1), marks=pytest.mark.check_small_gradients
        ),
        ((6, 3, 3, 4), (10, 3, 5, 5)),
        ((4, 1, 5, 4), (6, 1, 2, 2)),
    ],
)
@pytest.mark.parametrize("s", [1, 2])
@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("p", [3, 4])
@pytest.mark.parametrize("u", [1, 2])
@pytest.mark.parametrize("extra_layers", [True, False])
def test_mcorrlcorr_gradients(
    device, mcorrlcorr, f_shape, g_shape, s, d, p, u, extra_layers
):
    torch.manual_seed(1)
    f = torch.rand(f_shape, dtype=torch.double, device=device, requires_grad=True)
    g = torch.rand(g_shape, dtype=torch.double, device=device, requires_grad=True)
    b = torch.rand(f_shape[:1], dtype=torch.double, device=device, requires_grad=True)
    with torch.no_grad():
        h_shape = mcorrlcorr(g, f, b, (s, s), (d, d), (p, p), (u, u), (p, p)).size()
    if extra_layers:
        W_first = torch.rand((g_shape[-1],) * 2, dtype=torch.double, device=device)
        W_last = torch.rand((h_shape[-1],) * 2, dtype=torch.double, device=device)
    else:
        W_first = torch.eye(g_shape[-1], dtype=torch.double, device=device)
        W_last = torch.eye(h_shape[-1], dtype=torch.double, device=device)
    atol = 1e-6
    torch.autograd.gradcheck(
        # last p doubles as r
        lambda g, f, b: torch.matmul(
            mcorrlcorr(
                torch.matmul(g, W_first), f, b, (s, s), (d, d), (p, p), (u, u), (p, p)
            ),
            W_last,
        ),
        (g, f, b),
        atol=atol,
    )


@pytest.mark.check_gradients
@pytest.mark.parametrize(
    "g_shape,nf",
    [
        ((1, 1, 1), 1),
        pytest.param((3, 4, 10), 5, marks=pytest.mark.check_small_gradients,),
        ((2, 4, 6), 7),
    ],
)
@pytest.mark.parametrize("s", [1, 2])
@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("p", [3, 4])
@pytest.mark.parametrize("u", [1, 2])
def test_snd2col_gradients(device, snd2col, g_shape, nf, s, d, p, u):
    torch.manual_seed(5)
    g = torch.rand(g_shape, dtype=torch.double, device=device, requires_grad=True)
    atol, dtol = 1e-6, 1e-6
    torch.autograd.gradcheck(
        lambda g: snd2col(g, nf, s, d, p, u), (g,), atol=atol, nondet_tol=dtol,
    )


@pytest.mark.check_gradients
@pytest.mark.parametrize(
    "g_shape,nf",
    [
        ((1, 1, 1, 1), (1, 1)),
        pytest.param((3, 4, 10, 10), (5, 5), marks=pytest.mark.check_small_gradients,),
        ((2, 4, 6, 8), (7, 9)),
    ],
)
@pytest.mark.parametrize("s", [(1, 2), (2, 1)])
@pytest.mark.parametrize("d", [(1, 2), (2, 1)])
@pytest.mark.parametrize("p", [(3, 4), (4, 3)])
@pytest.mark.parametrize("u", [(1, 2), (2, 1)])
def test_spec2col_gradients(device, spec2col, g_shape, nf, s, d, p, u):
    torch.manual_seed(6)
    g = torch.rand(g_shape, dtype=torch.double, device=device, requires_grad=True)
    atol, dtol = 1e-6, 1e-6
    torch.autograd.gradcheck(
        lambda g: spec2col(g, nf, s, d, p, u), (g,), atol=atol, nondet_tol=dtol,
    )


@pytest.mark.ext_ops
def test_can_train_mellin_correlation_layer(device):
    N, Cin, Cout, L, ell = 50, 40, 30, 5, 20
    torch.manual_seed(30)
    mcorr = mconv.MCorr1d(Cin, Cout, L).to(device)
    inp = torch.rand(N, Cin, ell, requires_grad=True, device=device)
    loss = mcorr(inp).sum()
    loss.backward()
    post_back_grad_bias = mcorr.bias.grad
    post_back_grad_weight = mcorr.weight.grad
    assert not torch.allclose(
        torch.zeros_like(post_back_grad_bias), post_back_grad_bias
    )
    assert not torch.allclose(
        torch.zeros_like(post_back_grad_weight), post_back_grad_weight
    )


@pytest.mark.ext_ops
def test_can_train_mcorrlcorr_layer(device):
    N, Cin, Cout, W, H, w, h = 13, 12, 11, 10, 9, 15, 14
    torch.manual_seed(5)
    lcorrmcorr = mconv.MCorrLCorr(Cin, Cout, (W, H)).to(device)
    ins = torch.randn(N, Cin, w, h, requires_grad=True, device=device)
    targets = torch.randn(N, requires_grad=True, device=device)
    outs = lcorrmcorr(ins).sum(1).sum(1).sum(1)
    loss = torch.nn.MSELoss()(outs, targets)
    loss.backward()
    post_back_grad_bias = lcorrmcorr.bias.grad
    post_back_grad_weight = lcorrmcorr.weight.grad
    assert not torch.allclose(
        torch.zeros_like(post_back_grad_bias), post_back_grad_bias
    )
    assert not torch.allclose(
        torch.zeros_like(post_back_grad_weight), post_back_grad_weight
    )


@pytest.mark.gpu
def test_autocast_mcorrlcorr_func(mcorrlcorr):

    N, C_in, C_out, T_in, F_in, T_kern, F_kern = 128, 64, 256, 128, 64, 16, 8

    input = torch.rand(N, C_in, T_in, F_in, device="cuda", dtype=torch.half).float()
    weight = torch.rand(
        C_out, C_in, T_kern, F_kern, dtype=torch.half, device="cuda"
    ).float()
    bias = torch.rand(C_out, device="cuda", dtype=torch.half).float()

    out_a = mcorrlcorr(input, weight, bias).half()

    with torch.cuda.amp.autocast():
        out_b = mcorrlcorr(input, weight, bias)
    assert out_b.dtype == torch.half

    # the precision of the extension op is around 1e-3, but the python op is 1e-2.
    assert torch.allclose(out_a, out_b, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
def test_autocast_mcorr1d_layer():
    N, C_in, C_out, T_in, T_kern = 128, 64, 32, 16, 8

    mcorr1d = mconv.MCorr1d(C_in, C_out, T_kern).cuda()
    optimizer = torch.optim.Adam(mcorr1d.parameters())
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    input = torch.randn(N, C_in, T_in, device="cuda")
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        output = mcorr1d(input)
        loss = output.sum()

    scaler.scale(loss).backward()
    scaler.step(optimizer)

    scaler.update()


@pytest.mark.gpu
def test_autocast_mcorrlcorr_layer():
    N, C_in, C_out, T_in, F_in, T_kern, F_kern = 128, 64, 32, 16, 8, 4, 2

    mcorrlcorr = mconv.MCorrLCorr(C_in, C_out, (T_kern, F_kern)).cuda()
    optimizer = torch.optim.Adam(mcorrlcorr.parameters())
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    input = torch.randn(N, C_in, T_in, F_in, device="cuda")
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        output = mcorrlcorr(input)
        loss = output.sum()

    scaler.scale(loss).backward()
    scaler.step(optimizer)

    scaler.update()


@pytest.mark.gpu
def test_autocast_mcorr1d_func(mcorr1d):

    N, C_in, C_out, T_in, T_kern = 128, 64, 256, 128, 16

    input = torch.rand(N, C_in, T_in, device="cuda", dtype=torch.half).float()
    weight = torch.rand(C_out, C_in, T_kern, dtype=torch.half, device="cuda").float()
    bias = torch.rand(C_out, device="cuda", dtype=torch.half).float()

    out_a = mcorr1d(input, weight, bias).half()

    with torch.cuda.amp.autocast():
        out_b = mcorr1d(input, weight, bias)
    assert out_b.dtype == torch.half

    # the precision of the extension op is around 1e-3, but the python op is 1e-2.
    assert torch.allclose(out_a, out_b, rtol=1e-2, atol=1e-2)


@pytest.mark.ext_ops
@pytest.mark.gpu
@pytest.mark.parametrize("times", [1, 2, 10])
def test_mellin_layer_stays_sequential(times):
    torch.manual_seed(6)
    x = torch.randn(5, 1, 10, 10, requires_grad=True)
    lin1 = torch.nn.Linear(10, 4)
    lcorrmcorr = mconv.MCorrLCorr(1, 1, (3, 3), p=(2, 1), r=(0, 1))
    lin2 = torch.nn.Linear(4, 10)
    y = lin2(lcorrmcorr(lin1(x))).sum()
    y_exp = y.detach().cuda()
    dx_exp = torch.autograd.grad(y, x)[0].cuda()
    x = x.cuda()
    lin1 = lin1.cuda()
    lcorrmcorr = lcorrmcorr.cuda()
    lin2 = lin2.cuda()
    y_acts = []
    dx_acts = []
    for _ in range(times):
        y = lin2(lcorrmcorr(lin1(x))).sum()
        y_acts.append(y.detach())
        dx_acts.append(torch.autograd.grad(y, x)[0])
    for y_act in y_acts:
        assert torch.allclose(y_act, y_exp)
    for dx_act in dx_acts:
        assert torch.allclose(dx_act, dx_exp)
