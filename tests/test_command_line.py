from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import sys
import os

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import cnn_mellin.command_line as command_line

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def test_read_and_write_print_parameters_as_ini(capsys, temp_dir):
    command_line.print_parameters_as_ini([])
    s = capsys.readouterr()
    assert s.out.find('[data]') != -1
    a_ini = s.out[s.out.find('[model]'):s.out.find('[training]')]
    # this doesn't replace the original number, just appends 1000
    a_ini = a_ini.replace('num_conv = ', 'num_conv = 1000')
    a_ini = a_ini.replace('time_factor = ', 'time_factor = 1000')
    with open(os.path.join(temp_dir, 'a.ini'), 'w') as f:
        f.write(a_ini + '\n')
    b_ini = '[model]\ntime_factor = 30\n'
    with open(os.path.join(temp_dir, 'b.ini'), 'w') as f:
        f.write(b_ini)
    command_line.print_parameters_as_ini([
        os.path.join(temp_dir, 'a.ini'),
        os.path.join(temp_dir, 'b.ini'),
    ])
    s = capsys.readouterr()
    assert s.out.find('[data]') != -1
    assert s.out.find('num_conv = 1000') != -1
    assert s.out.find('time_factor = 1000') == -1
    assert s.out.find('time_factor = 30') != -1
