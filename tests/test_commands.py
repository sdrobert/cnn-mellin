from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def test_write_table_to_torch_dir(temp_dir):
    import torch
    out_dir = os.path.join(temp_dir, 'test_write_table_to_torch_dir')
    os.makedirs(out_dir)
    rwspecifier = 'ark:' + os.path.join(out_dir, 'table.ark')
    a = torch.rand(10, 4)
    b = torch.rand(5, 2)
    c = torch.rand(5, 100)
    from pydrobert.kaldi.io import open as kaldi_open
    with kaldi_open(rwspecifier, 'bm', mode='w') as table:
        table.write('a', a.numpy())
        table.write('b', b.numpy())
        table.write('c', c.numpy())
    from cnn_mellin.command_line import write_table_to_torch_dir
    write_table_to_torch_dir([rwspecifier, out_dir])
    assert torch.allclose(c, torch.load(os.path.join(out_dir, 'c.pt')))
    assert torch.allclose(b, torch.load(os.path.join(out_dir, 'b.pt')))
    assert torch.allclose(a, torch.load(os.path.join(out_dir, 'a.pt')))
