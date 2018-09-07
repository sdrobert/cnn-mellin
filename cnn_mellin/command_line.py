'''Command line hooks for cnn_mellin'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

import pydrobert.kaldi.io.enums as enums

from pydrobert.kaldi.io import open as kaldi_open
from pydrobert.kaldi.io.argparse import KaldiParser
from pydrobert.kaldi.logging import kaldi_logger_decorator
from pydrobert.kaldi.logging import kaldi_vlog_level_cmd_decorator
from pydrobert.kaldi.logging import register_logger_for_kaldi

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def _write_table_to_torch_dir_parse_args(args, logger):
    parser = KaldiParser(
        description=write_table_to_torch_dir.__doc__,
        add_verbose=True, logger=logger,
    )
    parser.add_argument(
        'rspecifier', type='kaldi_rspecifier', help='The table to read')
    parser.add_argument(
        'dir', type=str, help='The folder to write files to'
    )
    parser.add_argument(
        '-i', '--in-type', type='kaldi_dtype',
        default=enums.KaldiDataType.BaseMatrix,
        help='The type of table to read'
    )
    parser.add_argument(
        '-o', '--out-type', default=None,
        choices=[
            'float', 'double', 'half', 'byte', 'char', 'short', 'int', 'long',
        ],
        help='The type of torch tensor to write. If unset, it is inferrred '
        'from the input type'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='What to append to the key when making a file name'
    )
    options = parser.parse_args(args)
    return options


# sdrobert: probably should port to pydrobert-kaldi
@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def write_table_to_torch_dir(args=None):
    '''Write a kaldi table to a series of files in a directory

    Writes to a folder in the format:

       folder/
          <key_1><file_suffix>
          <key_2><file_suffix>
          ...
    '''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(logger)
    try:
        options = _write_table_to_torch_dir_parse_args(args, logger)
    except SystemExit as ex:
        return ex.code
    out_type = options.out_type
    if out_type is None:
        if options.in_type in {
                enums.KaldiDataType.BaseMatrix,
                enums.KaldiDataType.BaseVector,
                enums.KaldiDataType.WaveMatrix,
                enums.KaldiDataType.Base,
                enums.KaldiDataType.BasePairVector}:
            if options.in_type.is_double:
                out_type = 'double'
            else:
                out_type = 'float'
        elif options.in_type in {
                enums.KaldiDataType.FloatMatrix,
                enums.KaldiDataType.FloatVector}:
            out_type = 'float'
        elif options.in_type in {
                enums.KaldiDataType.DoubleMatrix,
                enums.KaldiDataType.Double}:
            out_type = 'double'
        elif options.in_type in {
                enums.KaldiDataType.Int32,
                enums.KaldiDataType.Int32Vector,
                enums.KaldiDataType.Int32VectorVector}:
            out_type = 'int'
        elif options.in_type == enums.KaldiDataType.Boolean:
            out_type = 'byte'
        else:
            print(
                'Do not know how to convert {} to torch type'.format(
                    options.in_type),
                file=sys.stderr)
            return 1
    import torch
    if out_type == 'float':
        out_type = torch.float
    elif out_type == 'double':
        out_type = torch.double
    elif out_type == 'half':
        out_type = torch.half
    elif out_type == 'byte':
        out_type = torch.uint8
    elif out_type == 'char':
        out_type = torch.int8
    elif out_type == 'short':
        out_type = torch.short
    elif out_type == 'int':
        out_type = torch.int
    elif out_type == 'long':
        out_type = torch.long
    try:
        os.makedirs(options.dir)
    except FileExistsError:
        pass
    with kaldi_open(options.rspecifier, options.in_type) as table:
        for key, value in table.items():
            value = torch.tensor(value).type(out_type)
            torch.save(
                value, os.path.join(options.dir, key + options.file_suffix))
    return 0
