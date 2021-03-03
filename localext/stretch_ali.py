#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import torch


def main(args=None):
    '''Stretch alignments to a new length by a factor'''
    parser = argparse.ArgumentParser(
        description=main.__doc__
    )
    parser.add_argument('--factor', default=1, type=int)
    parser.add_argument('in_pt')
    parser.add_argument('out_pt', nargs='?', default=None)
    options = parser.parse_args(args)
    t = torch.load(options.in_pt)
    t = torch.cat([t] * options.factor).view(options.factor, -1)
    t = t.transpose(0, 1).flatten()
    if options.out_pt is None:
        out_pt = options.in_pt
    else:
        out_pt = options.out_pt
    torch.save(t, out_pt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
