from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import platform

from codecs import open
from os import path
from setuptools import setup

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

PWD = path.abspath(path.dirname(__file__))
with open(path.join(PWD, 'README.md'), encoding='utf-8') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

SETUP_REQUIRES = ['setuptools_scm']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    SETUP_REQUIRES += ['pytest-runner', 'pytest']

INSTALL_REQUIRES = [
    'numpy', 'six', 'future', 'pydrobert-kaldi', 'param'
]
try:
    import torch
except ImportError:
    # workaround for conda install
    INSTALL_REQUIRES.append('pytorch')

if __name__ == '__main__':
    # this block is necessary to ensure multiprocessing works with pytest on
    # windows. I'm not sure *how* it fixes it, but see
    # https://stackoverflow.com/questions/49831205/how-to-get-setup-py-test-working-with-multiprocessing-on-windows
    setup(
        name='cnn-mellin',
        description='Speech processing with Python',
        long_description=LONG_DESCRIPTION,
        use_scm_version=True,
        zip_safe=False,
        url='https://github.com/sdrobert/cnn-mellin',
        author=__author__,
        author_email=__email__,
        license=__license__,
        packages=['cnn_mellin'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Researchers',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        install_requires=INSTALL_REQUIRES,
        setup_requires=SETUP_REQUIRES,
        entry_points={
            'console_scripts': [
                'write-table-to-torch-dir = cnn_mellin.command_line:write_'
                'table_to_torch_dir',
                'write-torch-dir-to-table = cnn_mellin.command_line:write_'
                'torch_dir_to_table',
                'get-torch-data-dir-info = cnn_mellin.command_line:get_torch_'
                'data_dir_info',
            ]
        }
    )
