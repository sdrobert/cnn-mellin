'''Command line hooks for cnn_mellin'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import math

import pydrobert.kaldi.io.enums as enums
import cnn_mellin.data as data

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
    parser.add_argument('dir', type=str, help='The folder to write files to')
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
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    options = parser.parse_args(args)
    return options


# TODO(sdrobert): probably should port to pydrobert-kaldi
@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def write_table_to_torch_dir(args=None):
    '''Write a kaldi table to a series of files in a directory

    Writes to a folder in the format:

    ::
        folder/
          <file_prefix><key_1><file_suffix>
          <file_prefix><key_2><file_suffix>
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
                value, os.path.join(
                    options.dir,
                    options.file_prefix + key + options.file_suffix))
    return 0


def _write_torch_dir_to_table_parse_args(args, logger):
    parser = KaldiParser(
        description=write_torch_dir_to_table.__doc__,
        add_verbose=True, logger=logger,
    )
    parser.add_argument('dir', type=str, help='The folder to read files from')
    parser.add_argument(
        'wspecifier', type='kaldi_wspecifier', help='The table to write to')
    parser.add_argument(
        '-o', '--out-type', type='kaldi_dtype',
        default=enums.KaldiDataType.BaseMatrix,
        help='The type of table to write to'
    )
    parser.add_argument(
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    options = parser.parse_args(args)
    return options


# TODO(sdrobert): probably should port to pydrobert-kaldi
@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def write_torch_dir_to_table(args=None):
    '''Write a torch data directory to a kaldi table

    Reads from a folder in the format:

    ::
        folder/
          <file_prefix><key_1><file_suffix>
          <file_prefix><key_2><file_suffix>
          ...
    '''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(logger)
    try:
        options = _write_torch_dir_to_table_parse_args(args, logger)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.dir):
        print("'{}' is not a directory".format(options.dir), file=sys.stderr)
        return 1
    import torch
    is_bool = False
    if options.out_type in {
            enums.KaldiDataType.BaseMatrix,
            enums.KaldiDataType.BaseVector,
            enums.KaldiDataType.WaveMatrix,
            enums.KaldiDataType.Base,
            enums.KaldiDataType.BasePairVector}:
        if options.out_type.is_double:
            torch_type = torch.double
        else:
            torch_type = torch.float
    elif options.out_type in {
            enums.KaldiDataType.FloatMatrix,
            enums.KaldiDataType.FloatVector}:
        torch_type = torch.float
    elif options.out_type in {
            enums.KaldiDataType.DoubleMatrix,
            enums.KaldiDataType.Double}:
        torch_type = torch.double
    elif options.out_type in {
            enums.KaldiDataType.Int32,
            enums.KaldiDataType.Int32Vector,
            enums.KaldiDataType.Int32VectorVector}:
        torch_type = torch.int
    elif options.out_type == enums.KaldiDataType.Boolean:
        torch_type = torch.uint8
        is_bool = True
    else:
        print(
            'Do not know how to convert {} from torch type'.format(
                options.out_type),
            file=sys.stderr)
        return 1
    neg_fsl = -len(options.file_suffix)
    if not neg_fsl:
        neg_fsl = None
    fpl = len(options.file_prefix)
    utt_ids = sorted(
        os.path.basename(x)[fpl:neg_fsl]
        for x in os.listdir(options.dir)
        if x.startswith(options.file_prefix) and
        x.endswith(options.file_suffix)
    )
    with kaldi_open(options.wspecifier, options.out_type, mode='w') as table:
        for utt_id in utt_ids:
            val = torch.load(os.path.join(
                options.dir,
                options.file_prefix + utt_id + options.file_suffix))
            val = val.cpu().type(torch_type).numpy()
            if is_bool:
                val = bool(val)  # make sure val is a scalar!
            table.write(utt_id, val)
    return 0


def _get_torch_data_dir_info_parse_args(args, logger):
    parser = KaldiParser(
        description=get_torch_data_dir_info.__doc__,
        add_verbose=True, logger=logger,
    )
    parser.add_argument('dir', type=str, help='The torch data directory')
    parser.add_argument(
        'wspecifier', type='kaldi_wspecifier', help='The table to write to')
    parser.add_argument(
        '--strict', type='kaldi_bool', default=True,
        help='If set, validate the data directory before collecting info. The '
        'process is described in cnn_mellin.data.validate_spect_data_set'
    )
    parser.add_argument(
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    return parser.parse_args(args)


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def get_torch_data_dir_info(args=None):
    '''Write a kaldi table with info about the specified torch data dir

    A torch data dir is of the form::

        dir/
            feats/
                <file_prefix><utt1><file_suffix>
                <file_prefix><utt2><file_suffix>
                ...
            [ali/
                <file_prefix><utt1><file_suffix>
                <file_prefix><utt1><file_suffix>
                ...
            ]

    Where ``feats`` contains ``FloatTensor``s of shape ``(N, F)``, where
    ``N`` is the number of frames (variable) and ``F`` is the number of
    filters (fixed) and ``ali``, if there, contains ``LongTensor``s of shape
    ``(N,)`` indicating the appropriate class labels.

    This command writes a sorted Kaldi table of integers with the following
    key-value pairs:

    1. "num_utterances", the total number of listed utterances
    2. "num_filts", ``F``
    3. "total_frames", ``sum(N)`` over the data dir
    4. "count_<i>", the number of instances of the class "<i>" that appear
        in ``ali`` (if available). If "count_<i>" is a valid key, then so
        are "count_<0 to i>". "count_<i>" is left-padded with zeros to ensure
        that the keys remain in the same order in the table as the class
        indices
    '''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(logger)
    try:
        options = _get_torch_data_dir_info_parse_args(args, logger)
    except SystemExit as ex:
        return ex.code
    if not os.path.isdir(options.dir):
        print("'{}' is not a directory".format(options.dir), file=sys.stderr)
        return 1
    logging.captureWarnings(True)
    data_set = data.SpectDataSet(
        options.dir,
        file_prefix=options.file_prefix,
        file_suffix=options.file_suffix,
    )
    if options.strict:
        data.validate_spect_data_set(data_set)
    info_dict = {
        'num_utterances': len(data_set),
        'total_frames': 0
    }
    counts = dict()
    max_class_idx = -1
    for feat, ali in data_set:
        info_dict['num_filts'] = feat.size()[1]
        info_dict['total_frames'] += feat.size()[0]
        if ali is not None:
            for class_idx in ali:
                class_idx = class_idx.item()
                max_class_idx = max(class_idx, max_class_idx)
                if max_class_idx < 0:
                    raise ValueError('Got a negative class idx')
                counts[class_idx] = counts.get(class_idx, 0) + 1
    if max_class_idx == 0:
        info_dict['count_0'] = counts[0]
    elif max_class_idx > 0:
        count_fmt_str = 'count_{{:0{}d}}'.format(
            int(math.log10(max_class_idx)) + 1)
        for class_idx in range(max_class_idx + 1):
            info_dict[count_fmt_str.format(class_idx)] = counts.get(
                class_idx, 0)
    info_list = sorted(info_dict.items())
    with kaldi_open(options.wspecifier, 'i', mode='w') as table:
        for key, value in info_list:
            table.write(key, value)
    return 0
