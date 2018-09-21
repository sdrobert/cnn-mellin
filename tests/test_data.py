from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from itertools import repeat

import pytest
import torch
import torch.utils.data

import cnn_mellin.data as data
import cnn_mellin.params as params

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


@pytest.mark.parametrize("left", [0, 1, 100])
@pytest.mark.parametrize("right", [0, 1, 100])
@pytest.mark.parametrize("T", [1, 5, 10])
def test_extract_window(left, right, T):
    # FIXME(sdrobert): the float conversion is due to a bug in torch.allclose.
    # Fix when fixed
    signal = torch.arange(T).float().view(-1, 1).expand(-1, 10)
    for frame_idx in range(T):
        window = data.extract_window(signal, frame_idx, left, right)
        left_pad = max(left - frame_idx, 0)
        right_pad = max(frame_idx + right + 1 - T, 0)
        assert tuple(window.size()) == (1 + left + right, 10)
        if left_pad:
            assert torch.allclose(window[:left_pad], torch.tensor([0]).float())
        if right_pad:
            assert torch.allclose(
                window[-right_pad:], torch.tensor([T - 1]).float())
        assert torch.allclose(
            window[left_pad:1 + left + right - right_pad],
            torch.arange(
                frame_idx - left + left_pad,
                frame_idx + right - right_pad + 1
            ).float().view(-1, 1).expand(-1, 10)
        )


@pytest.mark.parametrize('num_utts', [1, 2, 10])
def test_valid_spect_data_set(temp_dir, num_utts):
    torch.manual_seed(1)
    feats_dir = os.path.join(temp_dir, 'feats')
    ali_dir = os.path.join(temp_dir, 'ali')
    os.makedirs(feats_dir)
    os.makedirs(ali_dir)
    utt_ids = [
        '{:010d}'.format(x)
        for x in range(num_utts)
    ]
    feats = [
        torch.rand(x + 1, num_utts + 1)
        for x in range(num_utts)
    ]
    alis = [
        torch.randint(x + 1, (num_utts + 1,))
        for x in range(num_utts)
    ]
    with open(os.path.join(feats_dir, 'ignore_me'), 'w') as f:
        pass
    for utt_id, feat in zip(utt_ids, feats):
        torch.save(feat, os.path.join(feats_dir, utt_id + '.pt'))
    data_set = data.SpectDataSet(temp_dir)
    assert not data_set.has_ali
    assert len(utt_ids) == len(data_set.utt_ids)
    assert all(
        utt_a == utt_b for (utt_a, utt_b) in zip(utt_ids, data_set.utt_ids))
    assert all(
        ali_b is None and torch.allclose(feat_a, feat_b)
        for (feat_a, (feat_b, ali_b)) in zip(feats, data_set)
    )
    for utt_id, ali in zip(utt_ids, alis):
        torch.save(ali, os.path.join(ali_dir, utt_id + '.pt'))
    data_set = data.SpectDataSet(temp_dir)
    assert data_set.has_ali
    assert len(utt_ids) == len(data_set.utt_ids)
    assert all(
        utt_a == utt_b for (utt_a, utt_b) in zip(utt_ids, data_set.utt_ids))
    assert all(
        torch.allclose(ali_a, ali_b) and torch.allclose(feat_a, feat_b)
        for ((feat_a, ali_a), (feat_b, ali_b))
        in zip(zip(feats, alis), data_set)
    )


def test_spect_data_set_warnings(temp_dir):
    torch.manual_seed(1)
    feats_dir = os.path.join(temp_dir, 'feats')
    ali_dir = os.path.join(temp_dir, 'ali')
    os.makedirs(feats_dir)
    os.makedirs(ali_dir)
    torch.save(torch.rand(3, 3), os.path.join(feats_dir, 'a.pt'))
    torch.save(torch.rand(4, 3), os.path.join(feats_dir, 'b.pt'))
    torch.save(torch.randint(10, (4,)).long(), os.path.join(ali_dir, 'b.pt'))
    torch.save(torch.randint(10, (5,)).long(), os.path.join(ali_dir, 'c.pt'))
    data_set = data.SpectDataSet(temp_dir, warn_on_missing=False)
    assert data_set.has_ali
    assert data_set.utt_ids == ('b',)
    with pytest.warns(UserWarning) as warnings:
        data_set = data.SpectDataSet(temp_dir)
    assert len(warnings) == 2
    assert any(
        str(x.message) == "Missing ali for uttid: 'a'" for x in warnings)
    assert any(
        str(x.message) == "Missing feats for uttid: 'c'" for x in warnings)


@pytest.mark.parametrize('cuda', [
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason='no cuda available'
            )
        ),
    False,
])
def test_spect_data_write_pdf(temp_dir, cuda):
    torch.manual_seed(1)
    feats_dir = os.path.join(temp_dir, 'feats')
    os.makedirs(feats_dir)
    torch.save(torch.rand(3, 3), os.path.join(feats_dir, 'a.pt'))
    data_set = data.SpectDataSet(temp_dir)
    z = torch.randint(10, (4, 5)).long()
    if cuda:
        data_set.write_pdf('b', z.cuda())
    else:
        data_set.write_pdf('b', z)
    zp = torch.load(os.path.join(temp_dir, 'pdfs', 'b.pt'))
    assert isinstance(zp, torch.FloatTensor)
    assert torch.allclose(zp, z.float())
    data_set.write_pdf(0, torch.rand(10, 4))
    assert os.path.exists(os.path.join(temp_dir, 'pdfs', 'a.pt'))


def test_spect_data_set_validity(temp_dir):
    torch.manual_seed(1)
    feats_dir = os.path.join(temp_dir, 'feats')
    ali_dir = os.path.join(temp_dir, 'ali')
    feats_a_pt = os.path.join(feats_dir, 'a.pt')
    feats_b_pt = os.path.join(feats_dir, 'b.pt')
    ali_a_pt = os.path.join(ali_dir, 'a.pt')
    ali_b_pt = os.path.join(ali_dir, 'b.pt')
    os.makedirs(feats_dir)
    os.makedirs(ali_dir)
    torch.save(torch.rand(10, 4), feats_a_pt)
    torch.save(torch.rand(4, 4), feats_b_pt)
    torch.save(torch.randint(10, (10,)).long(), ali_a_pt)
    torch.save(torch.randint(10, (4,)).long(), ali_b_pt)
    data_set = data.SpectDataSet(temp_dir)
    data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 4).long(), feats_b_pt)
    with pytest.raises(ValueError, match='is not a FloatTensor'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4,), feats_b_pt)
    with pytest.raises(ValueError, match='does not have two axes'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 3), feats_b_pt)
    with pytest.raises(ValueError, match='has second axis size 3.*'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.rand(4, 4), feats_b_pt)
    data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4,)), ali_b_pt)
    with pytest.raises(ValueError, match='is not a LongTensor'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4, 1)).long(), ali_b_pt)
    with pytest.raises(ValueError, match='does not have one axis'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (3,)).long(), ali_b_pt)
    with pytest.raises(ValueError, match='does not have the same first'):
        data.validate_spect_data_set(data_set)
    torch.save(torch.randint(10, (4,)).long(), ali_b_pt)
    data.validate_spect_data_set(data_set)


def test_utterance_context_window_data_set(temp_dir):
    torch.manual_seed(1)
    feats_dir = os.path.join(temp_dir, 'feats')
    os.makedirs(feats_dir)
    a = torch.rand(2, 10)
    torch.save(a, os.path.join(feats_dir, 'a.pt'))
    data_set = data.UtteranceContextWindowDataSet(temp_dir, 1, 1)
    windowed, _ = data_set[0]
    assert tuple(windowed.size()) == (2, 3, 10)
    assert torch.allclose(a[0], windowed[0, :2])
    assert torch.allclose(a[1], windowed[0, 2])
    assert torch.allclose(a[0], windowed[1, 0])
    assert torch.allclose(a[1], windowed[1, 1:])


def test_single_context_window_data_set(temp_dir):
    torch.manual_seed(1)
    feats_dir = os.path.join(temp_dir, 'feats')
    ali_dir = os.path.join(temp_dir, 'ali')
    os.makedirs(feats_dir)
    os.makedirs(ali_dir)
    a = torch.rand(2, 5)
    b = torch.rand(4, 5)
    torch.save(a, os.path.join(feats_dir, 'a.pt'))
    torch.save(b, os.path.join(feats_dir, 'b.pt'))
    data_set = data.SingleContextWindowDataSet(temp_dir, 1, 1)
    assert len(data_set) == 6
    assert all(feats.size() == (3, 5) for (feats, ali) in data_set)
    assert torch.allclose(a[0], data_set[0][0][:2])
    assert torch.allclose(a[1], data_set[0][0][2])
    assert torch.allclose(a[0], data_set[1][0][0])
    assert torch.allclose(a[1], data_set[1][0][1:])
    assert torch.allclose(b[0], data_set[2][0][:2])
    assert torch.allclose(b[1], data_set[2][0][2])
    assert torch.allclose(b[:3], data_set[3][0])
    assert torch.allclose(b[1:], data_set[4][0])
    assert torch.allclose(b[2], data_set[5][0][0])
    assert torch.allclose(b[3], data_set[5][0][1:])
    assert torch.allclose(data_set[1][0], data_set[-5][0])
    torch.save(torch.arange(2).long(), os.path.join(ali_dir, 'a.pt'))
    torch.save(torch.arange(2, 6).long(), os.path.join(ali_dir, 'b.pt'))
    data_set = data.SingleContextWindowDataSet(temp_dir, 1, 1)
    assert tuple(ali for (feats, ali) in data_set) == tuple(range(6))


def test_epoch_random_sampler(temp_dir):
    data_source = torch.utils.data.TensorDataset(torch.arange(100))
    sampler = data.EpochRandomSampler(data_source, base_seed=1)
    samples_ep0 = tuple(sampler)
    samples_ep1 = tuple(sampler)
    assert samples_ep0 != samples_ep1
    assert sorted(samples_ep0) == list(range(100))
    assert sorted(samples_ep1) == list(range(100))
    assert samples_ep0 == tuple(sampler.get_samples_for_epoch(0))
    assert samples_ep1 == tuple(sampler.get_samples_for_epoch(1))
    sampler = data.EpochRandomSampler(data_source, init_epoch=10, base_seed=1)
    assert samples_ep0 == tuple(sampler.get_samples_for_epoch(0))
    assert samples_ep1 == tuple(sampler.get_samples_for_epoch(1))


@pytest.mark.parametrize('feat_sizes', [
    ((3, 5, 4), (4, 5, 4), (1, 5, 4)),
    ((2, 10),) * 10,
])
@pytest.mark.parametrize('include_ali', [True, False])
def test_context_window_seq_to_batch(feat_sizes, include_ali):
    torch.manual_seed(1)
    includes_frames = len(feat_sizes[0]) == 3
    feats = tuple(torch.rand(*x) for x in feat_sizes)
    if includes_frames:
        num_frames = sum(x[0] for x in feat_sizes)
        alis = tuple(torch.randint(10, (x[0],)).long() for x in feat_sizes)
    else:
        num_frames = len(feat_sizes)
        alis = tuple(
            x.item() for x in torch.randint(10, (num_frames,)).long())
    if not include_ali:
        alis = repeat(None)
    seq = zip(feats, alis)
    batch_feats, batch_ali = data.context_window_seq_to_batch(seq)
    assert tuple(batch_feats.size()) == (
        num_frames, *feat_sizes[0][int(includes_frames):])
    if include_ali:
        assert tuple(batch_ali.size()) == (num_frames,)
        # FIXME(sdrobert): casting to floats because of a bug in torch.allclose
        # fix when fixed
        batch_ali = batch_ali.float()
    else:
        assert not batch_ali
    if includes_frames:
        assert torch.allclose(torch.cat(feats), batch_feats)
        if include_ali:
            assert torch.allclose(torch.cat(alis).float(), batch_ali)
    else:
        assert torch.allclose(torch.stack(feats), batch_feats)
        if include_ali:
            assert torch.allclose(torch.tensor(alis).float(), batch_ali)


def test_training_data_loader(temp_dir):
    torch.manual_seed(1)
    feats_dir = os.path.join(temp_dir, 'feats')
    ali_dir = os.path.join(temp_dir, 'ali')
    os.makedirs(feats_dir)
    os.makedirs(ali_dir)
    p = params.SpectDataSetParams(
        context_left=1,
        context_right=1,
        batch_size=5,
        seed=2,
        drop_last=True,
    )
    for utt_idx in range(5):
        utt_fname = '{:02d}.pt'.format(utt_idx)
        num_frames = torch.randint(1, 5, (1,)).long().item()
        torch.save(
            torch.rand(num_frames, 2), os.path.join(feats_dir, utt_fname))
        torch.save(
            torch.randint(10, (num_frames,)).long(),
            os.path.join(ali_dir, utt_fname))
    data_loader = data.TrainingDataLoader(temp_dir, p)
    total_windows_ep0 = 0
    for feat, ali in data_loader:
        assert tuple(feat.size()) == (5, 3, 2)
        assert tuple(ali.size()) == (5,)
        total_windows_ep0 += 5
    assert total_windows_ep0 >= 5
    feats_ep1_a, alis_ep1_a = [], []
    total_windows_ep1 = 0
    for feat, ali in data_loader:
        assert tuple(feat.size()) == (5, 3, 2)
        assert tuple(ali.size()) == (5,)
        feats_ep1_a.append(feat)
        alis_ep1_a.append(ali)
        total_windows_ep1 += 5
    assert total_windows_ep0 == total_windows_ep1
    data_loader = data.TrainingDataLoader(
        temp_dir, p,
        init_epoch=1,
        num_workers=4,
    )
    feats_ep1_b, alis_ep1_b = [], []
    for feat, ali in data_loader:
        assert tuple(feat.size()) == (5, 3, 2)
        assert tuple(ali.size()) == (5,)
        feats_ep1_b.append(feat)
        alis_ep1_b.append(ali)
    assert all(
        torch.allclose(feat_a, feat_b)
        for (feat_a, feat_b) in zip(feats_ep1_a, feats_ep1_b)
    )
    assert all(
        torch.allclose(ali_a.float(), ali_b.float())
        for (ali_a, ali_b) in zip(alis_ep1_a, alis_ep1_b)
    )
