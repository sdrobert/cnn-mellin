from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from tempfile import mkdtemp
from shutil import rmtree

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


@pytest.fixture
def temp_dir():
    dir_name = mkdtemp()
    yield dir_name
    rmtree(dir_name)
