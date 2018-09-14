'''I/O-related objects and functions'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import torch
import torch.utils.data

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def extract_window(signal, frame_idx, left, right):
    '''Slice the signal to extract a context window

    Parameters
    ----------
    signal : torch.Tensor
        Of shape ``(T, F)``, where ``T`` is the time/frame axis, and ``F``
        is the frequency axis
    frame_idx : int
        The "center frame" ``0 <= frame_idx < T``
    left : int
        The number of frames in the window to the left (before) the center
        frame. Any frames below zero are edge-padded
    right : int
        The number of frames in the window to the right (after) the center
        frame. Any frames above ``T`` are edge-padded

    Returns
    -------
    window : torch.Tensor
        Of shape ``(1 + left + right, F)``
    '''
    T, F = signal.size()
    if frame_idx - left < 0 or frame_idx + right + 1 > T:
        win_size = 1 + left + right
        window = signal.new(win_size, F)
        left_pad = max(left - frame_idx, 0)
        right_pad = max(frame_idx + right + 1 - T, 0)
        window[left_pad:win_size - right_pad] = signal[
            max(0, frame_idx - left):frame_idx + right + 1]
        if left_pad:
            window[:left_pad] = signal[0]
        if right_pad:
            window[-right_pad:] = signal[-1]
    else:
        window = signal[frame_idx - left:frame_idx + right + 1]
    return window


class SpectDataSet(torch.utils.data.Dataset):
    '''Accesses spectrographic filter data stored in a data directory

    ``SpectDataSet`` assumes that `data_dir` is structured as

    ::
        data_dir/
            feats/
                <uttid1><file_suffix>
                <uttid2><file_suffix>
                ...
            [
            ali/
                <uttid1><file_suffix>
                <uttid2><file_suffix>
                ...
            ]

    The ``feats`` dir stores filter bank data in the form of
    ``torch.FloatTensor``s of size ``(T, F)``, where ``T`` is the time
    dimension and ``F`` is the filter/log-frequency dimension. ``ali`` stores
    ``torch.LongTensor`s of size ``(T,)`` indicating the pdf-id of the most
    likely target. ``ali/`` is optional.

    As a ``Sequence``, the ``i``th index of a ``SpectDataSet`` instance
    returns a pair of ``feat[uttids[i]], ali[uttids[i]]``. If ``ali/`` did not
    exist on initialization, the second element of the pair will be ``None``

    Parameters
    ----------
    data_dir : str
        A path to feature directory
    file_suffix : str, optional
        The suffix that indicates that the file counts toward the data set
    warn_on_missing : bool, optional
        If some files with ``file_suffix`` exist in the ``ali/`` dir,
        there's a mismatch between the utterances in ``feats/`` and ``ali/``,
        and `warn_on_missing` is ``True``, a warning will be issued
        (via ``warnings``) regarding each such mismatch

    Attributes
    ----------
    data_dir : str
    file_suffix : str
    has_ali : bool
        Whether alignment data exist
    utt_ids : tuple
        A tuple of all utterance ids extracted from the data directory. They
        are stored in the same order as features and alignments via
        ``__getitem__``. If the ``ali/`` directory exists, `utt_ids`
        contains only the utterances in the intersection of each directory.
    '''

    def __init__(self, data_dir, file_suffix='.pt', warn_on_missing=True):
        super(SpectDataSet, self).__init__()
        self.data_dir = data_dir
        self.file_suffix = file_suffix
        self.has_ali = os.path.isdir(os.path.join(data_dir, 'ali'))
        if self.has_ali:
            self.has_ali = bool(sum(
                1 for x in os.listdir(os.path.join(data_dir, 'ali'))
                if x.endswith(file_suffix)
            ))
        self.utt_ids = tuple(sorted(
            self.find_utt_ids(warn_on_missing)))

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        return self.get_utterance_pair(idx)

    def find_utt_ids(self, warn_on_missing):
        '''Iterator : all utterance ids from data_dir'''
        utt_ids = (
            os.path.basename(x)[:-len(self.file_suffix)]
            for x in os.listdir(os.path.join(self.data_dir, 'feats'))
            if x.endswith(self.file_suffix)
        )
        try:
            ali_utt_ids = set(
                os.path.basename(x)[:-len(self.file_suffix)]
                for x in os.listdir(os.path.join(self.data_dir, 'ali'))
                if x.endswith(self.file_suffix)
            )
        except FileNotFoundError:
            assert not self.has_ali
            ali_utt_ids = set()
        if ali_utt_ids:
            utt_ids = set(utt_ids)
            if warn_on_missing:
                for utt_id in utt_ids.difference(ali_utt_ids):
                    warnings.warn("Missing ali for uttid: '{}'".format(utt_id))
                for utt_id in ali_utt_ids.difference(utt_ids):
                    warnings.warn(
                        "Missing feats for uttid: '{}'".format(utt_id))
            utt_ids &= ali_utt_ids
        return utt_ids

    def get_utterance_pair(self, idx):
        '''Get a pair of features, alignments'''
        utt_id = self.utt_ids[idx]
        feats = torch.load(
            os.path.join(self.data_dir, 'feats', utt_id + self.file_suffix))
        if self.has_ali:
            ali = torch.load(
                os.path.join(self.data_dir, 'ali', utt_id + self.file_suffix))
        else:
            ali = None
        return feats, ali

    def write_pdf(self, utt, pdf):
        '''Write a pdf FloatTensor to the data directory

        This method writes a pdf matrix to the directory ``data_dir/pdfs``
        with the name ``<utt><file_suffix>``

        Parameters
        ----------
        utt : str or int
            The name of the utterance to write. If an integer is specified,
            `utt` is assumed to index an utterance id specified in
            ``self.utt_ids``
        pdf : torch.Tensor
            The tensor to write. It will be converted to a ``FloatTensor``
            using the command ``pdf.cpu().float()``
        '''
        if isinstance(utt, int):
            utt = self.utt_ids[utt]
        pdfs_dir = os.path.join(self.data_dir, 'pdfs')
        os.makedirs(pdfs_dir, exist_ok=True)
        torch.save(
            pdf.cpu().float(),
            os.path.join(pdfs_dir, utt + self.file_suffix)
        )


def validate_spect_data_set(data_set):
    '''Validate SpectDataSet data directory

    The data directory is valid if the following conditions are observed

     1. All features are ``FloatTensor`` instances
     2. All features have two axes
     3. All features have the same size second axis
     4. All alignments (if present) are ``LongTensor`` instances
     5. All alignments (if present) have one axis
     6. Features and alignments (if present) have the same size first
        axes for a given utterance id

    Raises a ``ValueError`` if a condition is violated
    '''
    num_filts = None
    for idx, (feats, ali) in enumerate(data_set):
        if not isinstance(feats, torch.FloatTensor):
            raise ValueError(
                "'{}' (index {}) in '{}' is not a FloatTensor".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, 'feats')))
        if len(feats.size()) != 2:
            raise ValueError(
                "'{}' (index {}) in '{}' does not have two axes".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, 'feats')
                ))
        if num_filts is None:
            num_filts = feats.size()[1]
        elif feats.size()[1] != num_filts:
            raise ValueError(
                "'{}' (index {}) in '{}' has second axis size {}, which "
                "does not match prior utterance ('{}') size of {}".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, 'feats'),
                    feats.size()[1],
                    data_set.utt_ids[idx - 1] + data_set.file_suffix,
                    num_filts))
        if ali is None:
            continue
        if not isinstance(ali, torch.LongTensor):
            raise ValueError(
                "'{}' (index {}) in '{}' is not a LongTensor".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, 'ali')))
        if len(ali.size()) != 1:
            raise ValueError(
                "'{}' (index {}) in '{}' does not have one axis".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, 'ali')))
        if ali.size()[0] != feats.size()[0]:
            raise ValueError(
                "'{}' (index {}) in '{}' does not have the same first axis "
                "size ({}) as it's companion in '{}' ({})".format(
                    data_set.utt_ids[idx] + data_set.file_suffix, idx,
                    os.path.join(data_set.data_dir, 'feats'),
                    feats.size()[0],
                    os.path.join(data_set.data_dir, 'ali'),
                    ali.size()[0]))
