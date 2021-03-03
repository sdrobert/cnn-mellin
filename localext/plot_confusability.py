#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd


def plot_type_isolate(df, type_, path):
    df = df.set_index(['trial', 'type', 'pair'])
    trials = df.index.levels[0]
    types = df.index.levels[1]
    type_axes = len(types) - 1
    if not type_axes:
        raise ValueError('More than one type needed')
    trial_axes = len(trials)
    if type_axes == 1:
        x_subplots, y_subplots, transpose_axes = trial_axes, type_axes, True
    else:
        x_subplots, y_subplots, transpose_axes = type_axes, trial_axes, False
    fig, axes = plt.subplots(
        y_subplots, x_subplots, figsize=(3 * x_subplots, 3 * y_subplots),
        sharex=True, sharey=True, constrained_layout=True)
    if y_subplots == 1:
        axes = axes[None]
    if x_subplots == 1:
        axes = axes[:, None]
    if transpose_axes:
        axes = axes.T
    N_accum = 0.
    r_accums = dict()
    for (trial_no, df_trial), axes_x in zip(df.groupby('trial'), axes):
        df_trial = df_trial.loc[trial_no]
        df_type = df_trial.loc[type_]
        for (other_type, df_other_type), ax in zip(
                df_trial.drop(index=[type_], level=0).groupby('type'), axes_x):
            df_other_type = df_other_type.loc[other_type]
            suffixes = (' ' + type_, ' ' + other_type)
            names = tuple('count' + suffix for suffix in suffixes)
            merge = pd.merge(
                df_type, df_other_type, on='pair', suffixes=suffixes)
            N = len(merge)
            N_accum += N
            r = merge.corr()[names[0]][names[1]]
            r_accums[other_type] = r_accums.get(other_type, 0.) + r
            merge.plot(
                kind='scatter', ax=ax, alpha=.15,
                x=names[0], y=names[1],
                title='Trial {}\n(N={}, r={:.3f})'.format(trial_no, N, r))
    if len(types) == 2:
        other_name = types[1] if types[0] == type_ else types[0]
        averages = 'av. r={:.3f}'.format(r_accums[other_name] / trial_axes)
    else:
        other_name = 'others'
        averages = ', '.join([
            'av. {} r={:.3f}'.format(k, v / trial_axes)
            for (k, v) in r_accums.items()])
    fig.suptitle(
        '{} vs {} (av. N={}, {})'.format(
            type_, other_name, N_accum / (type_axes * trial_axes),
            averages))
    fig.savefig(path)


def plot_type_combine(df, path):
    types = df.index.levels[0]
    if len(types) != 2:
        raise ValueError('Cannot plot with more than two types')
    suffixes = tuple('_{}'.format(type_) for type_ in types)
    names = tuple('count' + suffix for suffix in suffixes)
    df_a = df.loc[types[0]]
    df_b = df.loc[types[1]]
    merge = pd.merge(df_a, df_b, on=['trial', 'pair'], suffixes=suffixes)
    assert len(merge) == len(df) // 2
    fig, ax = plt.subplots(1, figsize=(5, 5))
    merge.plot(
        kind='scatter', x=names[0], y=names[1], ax=ax, alpha=.15,
        title='Combined {} vs {} (N={}, r={:.3f})'.format(
            types[0], types[1],
            len(merge), merge.corr()[names[0]][names[1]]))
    fig.tight_layout()
    fig.savefig(path)


def plot_trial_isolate(df, trial, path):
    df = df.set_index(['type', 'trial', 'pair'])
    types = df.index.levels[0]
    trials = df.index.levels[1]
    trial_axes = len(trials) - 1
    type_axes = len(types)
    if not trial_axes:
        raise ValueError('More than one type needed')
    if trial_axes == 1:
        x_subplots, y_subplots, transpose_axes = type_axes, trial_axes, True
    else:
        x_subplots, y_subplots, transpose_axes = trial_axes, type_axes, False
    fig, axes = plt.subplots(
        y_subplots, x_subplots, figsize=(3 * x_subplots, 3 * y_subplots),
        sharex=True, sharey=True, constrained_layout=True)
    if y_subplots == 1:
        axes = axes[None]
    if x_subplots == 1:
        axes = axes[:, None]
    if transpose_axes:
        axes = axes.T
    N_accum = 0.
    r_accums = dict()
    for (type_, df_type), axes_x in zip(df.groupby('type'), axes):
        df_type = df_type.loc[type_]
        df_trial = df_type.loc[trial]
        for (other_trial, df_other_trial), ax in zip(
                df_type.drop(index=[trial], level=0).groupby('trial'), axes_x):
            df_other_trial = df_other_trial.loc[other_trial]
            suffixes = (' trial ' + str(trial), ' trial ' + str(other_trial))
            names = tuple('count' + suffix for suffix in suffixes)
            merge = pd.merge(
                df_trial, df_other_trial, on='pair', suffixes=suffixes)
            N = len(merge)
            N_accum += N
            r = merge.corr()[names[0]][names[1]]
            r_accums[other_trial] = r_accums.get(other_trial, 0.) + r
            merge.plot(
                kind='scatter', ax=ax, alpha=.15,
                x=names[0], y=names[1],
                title='{}\n(N={}, r={:.3f})'.format(type_, N, r))
    if len(trials) == 2:
        other_name = trials[1] if trials[0] == trial else trials[0]
        averages = 'av. r={:.3f}'.format(r_accums[other_name] / type_axes)
    else:
        other_name = 'others'
        averages = ', '.join([
            'av. trial {} r={:.3f}'.format(k, v / type_axes)
            for (k, v) in r_accums.items()])
    fig.suptitle(
        'Trial {} vs {} (av. N={}, {})'.format(
            trial, other_name, N_accum / (trial_axes * type_axes),
            averages))
    fig.savefig(path)


def _parse_args(args):
    parser = argparse.ArgumentParser(
        description=main.__doc__,
    )
    parser.add_argument(
        'data_csv', type=argparse.FileType('r'),
        help='Path to data csv. Should have columns type, pair, trial, and '
        'count')
    subparsers = parser.add_subparsers(
        help='What sort of comparison to plot', dest='comparison')
    type_isolate_parser = subparsers.add_parser(
        'type_isolate',
        help='Plot type vs type, fixing one type. Enumerate trials'
    )
    type_isolate_parser.add_argument(
        'type', help='The type to fix'
    )
    type_isolate_parser.add_argument(
        'out_file', nargs='?',
        default='type_isolate_{type}.pdf',
        help='File to plot to'
    )
    type_combine_parser = subparsers.add_parser(
        'type_combine',
        help='Plot type vs type, combining all trials into one graph'
    )
    type_combine_parser.add_argument(
        'out_file', nargs='?',
        default='type_combine.pdf',
        help='File to plot to'
    )
    trial_isolate_parser = subparsers.add_parser(
        'trial_isolate',
        help='Plot trial vs trial, fixing one trial. Enumerates types'
    )
    trial_isolate_parser.add_argument(
        'trial_no', type=int,
        help='Trial to fix in comparison'
    )
    trial_isolate_parser.add_argument(
        'out_file', nargs='?',
        default='trial_isolate_{trial_no}.pdf',
        help='File to plot to'
    )
    return parser.parse_args(args)


def main(args=None):
    '''Plot confusability data'''
    options = _parse_args(args)
    df = pd.read_csv(options.data_csv).assign(
        type=lambda x: pd.Categorical(x['type']),
        pair=lambda x: pd.Categorical(x['pair'], ordered=True),
        trial=lambda x: pd.Categorical(x['trial']))
    types = df['type'].dtype.categories
    trials = df['trial'].dtype.categories
    pairs = df['pair'].dtype.categories
    mi = pd.MultiIndex.from_product(
        [types, trials, pairs], names=['type', 'trial', 'pair'])
    df = (
        df.set_index(['type', 'trial', 'pair'])
        .reindex(mi, fill_value=0.)
        .reset_index())
    if options.comparison == 'type_isolate':
        plot_type_isolate(
            df, options.type, options.out_file.format(type=options.type))
    elif options.comparison == 'type_combine':
        plot_type_combine(df, options.out_file)
    elif options.comparison == 'trial_isolate':
        plot_trial_isolate(
            df, options.trial_no,
            options.out_file.format(trial_no=options.trial_no))


if __name__ == '__main__':
    sys.exit(main())
