'''I/O-related objects and functions'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import numpy as np
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
        neg_fsl = -len(self.file_suffix)
        if not neg_fsl:
            neg_fsl = None
        utt_ids = (
            os.path.basename(x)[:neg_fsl]
            for x in os.listdir(os.path.join(self.data_dir, 'feats'))
            if x.endswith(self.file_suffix)
        )
        try:
            ali_utt_ids = set(
                os.path.basename(x)[:neg_fsl]
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
    for idx in range(len(data_set.utt_ids)):
        feats, ali = data_set.get_utterance_pair(idx)
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


class UtteranceContextWindowDataSet(SpectDataSet):
    '''SpectDataSet, extracting fixed-width windows over the utterance

    Like a ``SpectDataSet``, ``UtteranceContextWindowDataSet`` indexes pairs of
    features and alignments. Instead of returning features of shape ``(T, F)``,
    instances return features of shape ``(T, 1 + left + right, F)``, where the
    ``T`` axis indexes the so-called center frame and the ``1 + left + right``
    axis contains frame vectors (size ``F``) including the center frame,
    ``left`` frames in time before the center frame, and ``right`` frames after

    Parameters
    ----------
    data_dir : str
    left : int
    right : int
    file_suffix : str, optional
    warn_on_missing : bool, optional

    Attributes
    ----------
    data_dir : str
    left : int
    right : int
    has_ali : bool
    utt_ids : tuple
    '''

    def __init__(self, data_dir, left, right, **kwargs):
        super(UtteranceContextWindowDataSet, self).__init__(data_dir, **kwargs)
        self.left = left
        self.right = right

    def get_windowed_utterance(self, idx):
        '''Get pair of features (w/ context window) and alignments'''
        feats, ali = self.get_utterance_pair(idx)
        num_frames, num_filts = feats.size()
        windowed = torch.empty(
            num_frames, 1 + self.left + self.right, num_filts)
        for center_frame in range(num_frames):
            windowed[center_frame] = extract_window(
                feats, center_frame, self.left, self.right)
        return windowed, ali

    def __getitem__(self, idx):
        return self.get_windowed_utterance(idx)


class SingleContextWindowDataSet(SpectDataSet):
    '''SpectDataSet, returning a single context window per index

    Like ``SpectDataSet``, ``SingleContextWindowDataSet`` indexes pairs of
    features and alignments. Instead of indexing features of shape ``(T, F)``
    and alignments of shape ``T``, instances break down each utterance into
    ``T`` separate context windows and ``T`` integers and indexes those. For
    the ``t``-th context window of features, we have a tensor of size ``(1 +
    left + right, F)``, where ``window[left]`` is the "center frame",
    ``window[:left]`` are the frames before it, and ``window[left + 1:]`` are
    those after. Context windows are ordered first by utterance, second by
    center frame idx. For example, if there are only two utterances, "a" and
    "b" of sizes ``(2, 5)`` and ``(4, 5)``, respectively, then the first index
    (0) of the data set would be the context window of "a"'s first frame,
    whereas the 4th index (3) of the data set points to the context window of
    "b"'s second frame (1).

    Parameters
    ----------
    data_dir : str
    left : int
    right : int
    has_ali : bool
    utt_ids : tuple
    utt_lens : tuple
        A tuple containing the number of frames of each utterance (i.e. ``T``)
    '''

    def __init__(self, data_dir, left, right, **kwargs):
        super(SingleContextWindowDataSet, self).__init__(data_dir, **kwargs)
        self.left = left
        self.right = right
        self.utt_lens = tuple(
            torch.load(os.path.join(
                self.data_dir, 'feats', x + self.file_suffix)).size()[0]
            for x in self.utt_ids
        )

    def __len__(self):
        return sum(self.utt_lens)

    def get_context_window(self, idx):
        idx_err_msg = 'list index out of range'
        if idx < 0:
            idx += len(self)
            if idx < 0:
                raise IndexError(idx_err_msg)
        utt_idx = 0
        for utt_len in self.utt_lens:
            if idx < utt_len:
                break
            idx -= utt_len
            utt_idx += 1
        if utt_idx == len(self.utt_ids):
            raise IndexError(idx_err_msg)
        feats, ali = self.get_utterance_pair(utt_idx)
        window = extract_window(feats, idx, self.left, self.right)
        return window, ali[idx].item() if ali is not None else ali

    def __getitem__(self, idx):
        return self.get_context_window(idx)


class EpochRandomSampler(torch.utils.data.Sampler):
    '''Return random samples that are the same for a fixed epoch

    Parameters
    ----------
    data_source : torch.data.utils.Dataset
        The total number of samples
    init_epoch : int, optional
        The initial epoch
    base_seed : int, optional
        Determines the starting seed of the sampler. Sampling is seeded with
        ``base_seed + epoch``. If unset, a seed is randomly generated from
        the default generator

    Attributes
    ----------
    epoch : int
        The current epoch. Responsible for seeding the upcoming samples
    data_source : torch.data.utils.Dataset
    base_seed : int

    Examples
    --------
    >>> sampler = EpochRandomSampler(
    ...     torch.data.utils.TensorDataset(torch.arange(100)))
    >>> samples_ep0 = tuple(sampler)  # random
    >>> samples_ep1 = tuple(sampler)  # random, probably not same as first
    >>> assert tuple(sampler.get_samples_for_epoch(0)) == samples_ep0
    >>> assert tuple(sampler.get_samples_for_epoch(1)) == samples_ep1
    '''

    def __init__(self, data_source, init_epoch=0, base_seed=None):
        super(EpochRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.epoch = init_epoch
        if base_seed is None:
            base_seed = np.random.randint(np.iinfo(np.uint32).max)
        self.base_seed = base_seed

    def __len__(self):
        return len(self.data_source)

    def get_samples_for_epoch(self, epoch):
        '''tuple : samples for a specific epoch'''
        rs = np.random.RandomState(self.base_seed + epoch)
        return rs.permutation(range(len(self.data_source)))

    def __iter__(self):
        ret = iter(self.get_samples_for_epoch(self.epoch))
        self.epoch += 1
        return ret


def context_window_seq_to_batch(seq):
    '''Convert a sequence of context window elements to a batch

    Given a sequence of ``feats, ali`` pairs, where ``feats`` is either of size
    ``(T, C, F)`` or ``(C, F)``, where ``T`` is some number of windows (which
    can vary across elements in the sequence), ``C`` is the window size, and
    ``F`` is some number filters, and ``ali`` is either of size ``(T,)`` or is
    an integer, this method batches all the elements of the sequence into a
    pair of ``batch_feats, batch_ali``, where ``batch_feats`` is of size
    ``(N, C, F)``, ``N`` being the total number of windows, and ``batch_ali``
    is of size ``(N,)``

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    batch_feats, batch_ali : torch.FloatTensor, torch.LongTensor or None
    '''
    batch_feats = []
    batch_ali = []
    for feats, ali in seq:
        if len(feats.size()) == 2:
            feats = feats.unsqueeze(0)
        batch_feats.append(feats)
        if ali is None:
            # assume every remaining ali will be none
            batch_ali = None
        else:
            if isinstance(ali, int):
                ali = torch.tensor([ali])
            batch_ali.append(ali)
    batch_feats = torch.cat(batch_feats)
    if batch_ali is not None:
        batch_ali = torch.cat(batch_ali)
    return batch_feats, batch_ali


class TrainingDataLoader(torch.utils.data.DataLoader):
    '''Serve batches of context windows randomly for training

    Parameters
    ----------
    data_dir : str
        Path to the torch data directory. Should have the format
        ::
            data_dir/
                feats/
                    <utt1>.pt
                    <utt2>.pt
                    ...
                ali/
                    <utt1>.pt
                    <utt2>.pt
                    ...
    params : cnn_model.params.SpectDataSetParams
        Parameters for things like context window size, batch size, and
        seed
    init_epoch : int, optional
        Where training should resume from
    kwargs : keyword arguments, optional
        Additional ``DataLoader`` arguments

    Attributes
    ----------
    data_dir : str
    params : cnn_model.params.SpectDataParams
    '''

    def __init__(self, data_dir, params, init_epoch=0, **kwargs):
        for bad_kwarg in (
                'batch_size', 'sampler', 'batch_sampler', 'shuffle',
                'collate_fn'):
            if bad_kwarg in kwargs:
                raise TypeError(
                    'keyword argument "{}" invalid for {} types'.format(
                        bad_kwarg, type(self)))
        self.data_dir = data_dir
        self.params = params
        data_source = SingleContextWindowDataSet(
            data_dir, params.context_left, params.context_right)
        if not data_source.has_ali:
            raise ValueError(
                "'{}' must have alignment info for training".format(
                    data_dir))
        sampler = EpochRandomSampler(
            data_source, init_epoch=init_epoch, base_seed=params.seed)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, params.batch_size, drop_last=params.drop_last)
        super(TrainingDataLoader, self).__init__(
            data_source,
            batch_sampler=batch_sampler,
            collate_fn=context_window_seq_to_batch,
            **kwargs
        )
