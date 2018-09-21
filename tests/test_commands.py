from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pytest
import torch
import cnn_mellin.command_line as command_line

from pydrobert.kaldi.io import open as kaldi_open

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


@pytest.mark.cpu
def test_write_table_to_torch_dir(temp_dir):
    out_dir = os.path.join(temp_dir, 'test_write_table_to_torch_dir')
    os.makedirs(out_dir)
    rwspecifier = 'ark:' + os.path.join(out_dir, 'table.ark')
    a = torch.rand(10, 4)
    b = torch.rand(5, 2)
    c = torch.rand(5, 100)
    with kaldi_open(rwspecifier, 'bm', mode='w') as table:
        table.write('a', a.numpy())
        table.write('b', b.numpy())
        table.write('c', c.numpy())
    assert not command_line.write_table_to_torch_dir([rwspecifier, out_dir])
    assert torch.allclose(c, torch.load(os.path.join(out_dir, 'c.pt')))
    assert torch.allclose(b, torch.load(os.path.join(out_dir, 'b.pt')))
    assert torch.allclose(a, torch.load(os.path.join(out_dir, 'a.pt')))


@pytest.mark.cpu
def test_write_torch_dir_to_table(temp_dir):
    import torch
    in_dir = os.path.join(temp_dir, 'test_write_torch_dir_to_table')
    rwspecifier = 'ark:' + os.path.join(in_dir, 'table.ark')
    os.makedirs(in_dir)
    a = torch.rand(5, 4)
    b = torch.rand(4, 3)
    c = torch.rand(3, 2)
    torch.save(a, os.path.join(in_dir, 'a.pt'))
    torch.save(b, os.path.join(in_dir, 'b.pt'))
    torch.save(c, os.path.join(in_dir, 'c.pt'))
    assert not command_line.write_torch_dir_to_table([in_dir, rwspecifier])
    with kaldi_open(rwspecifier, 'bm') as table:
        keys, vals = zip(*table.items())
        keys = tuple(keys)
        vals = tuple(vals)
    assert keys == ('a', 'b', 'c')
    assert len(vals) == 3
    for dval, tval in zip((a, b, c), vals):
        assert torch.allclose(dval, torch.from_numpy(tval))
