import pytest
import os
import math

from tempfile import mkdtemp
from shutil import rmtree

import torch

@pytest.fixture
def temp_dir():
    dir_name = mkdtemp()
    yield dir_name
    rmtree(dir_name)


CUDA_AVAIL = torch.cuda.is_available()


@pytest.fixture(
    params=[
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=pytest.mark.gpu),
    ],
    scope="session",
)
def device(request):
    if request.param == "cuda":
        return torch.device(torch.cuda.current_device())
    else:
        return torch.device(request.param)


def pytest_runtest_setup(item):
    if any(mark.name == "gpu" for mark in item.iter_markers()):
        if not CUDA_AVAIL:
            pytest.skip("cuda is not available")
    # implicitly seeds all tests for the sake of reproducibility
    torch.manual_seed(abs(hash(item.name)))


@pytest.fixture(scope="session")
def populate_torch_dir():
    def _populate_torch_dir(
        dr,
        num_utts,
        min_width=1,
        max_width=10,
        num_filts=5,
        max_class=10,
        include_refs=True,
        file_prefix="",
        file_suffix=".pt",
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        feats_dir = os.path.join(dr, "feat")
        ref_dir = os.path.join(dr, "ref")
        os.makedirs(feats_dir, exist_ok=True)
        if include_refs:
            os.makedirs(ref_dir, exist_ok=True)
        feats, feat_sizes, utt_ids = [], [], []
        refs = [] if include_refs else None
        utt_id_fmt_str = "{{:0{}d}}".format(int(math.log10(num_utts)) + 1)
        for utt_idx in range(num_utts):
            utt_id = utt_id_fmt_str.format(utt_idx)
            feat_size = torch.randint(min_width, max_width + 1, (1,)).item()
            feat = torch.rand(feat_size, num_filts)
            torch.save(
                feat, os.path.join(feats_dir, file_prefix + utt_id + file_suffix)
            )
            feats.append(feat)
            feat_sizes.append(feat_size)
            utt_ids.append(utt_id)
            if include_refs:
                ref_size = torch.randint(1, feat_size + 1, (1,)).item()
                ref_tokens = torch.randint(max_class + 1, (ref_size, 1))
                ref_bounds = torch.full((ref_size, 2), -1, dtype=torch.long)
                ref = torch.cat([ref_tokens, ref_bounds], 1)
                torch.save(
                    ref, os.path.join(ref_dir, file_prefix + utt_id + file_suffix)
                )
                refs.append(ref)
        return feats, refs, feat_sizes, utt_ids

    return _populate_torch_dir
