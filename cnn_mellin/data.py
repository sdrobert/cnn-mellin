'''I/O-related objects and functions'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

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
