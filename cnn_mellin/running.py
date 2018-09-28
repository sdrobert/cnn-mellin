'''Functions involved in running the models'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def get_am_alignment_cross_entropy(
        model, data_loader, cuda=False, weight=None):
    '''Get the mean cross entropy of alignments over a data set

    Arguments
    ---------
    model : AcousticModel
    data_loader : EvaluationDataLoader
    cuda : bool, optional
        Whether to move the model/data to the GPU
    weight : FloatTensor, optional
        Relative weights to assign to each class. If unset, weights are
        uniform

    Returns
    -------
    average_loss : float
        The cross-entropy loss averaged over all context windows
    '''
    if not len(data_loader):
        raise ValueError('There must be at least one batch of data!')
    total_windows = 0
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, reduction='sum')
    if cuda:
        model.cuda()
    else:
        model.cpu()
    model.eval()
    with torch.no_grad():
        for feats, ali, feat_sizes, _ in data_loader:
            total_windows += sum(feat_sizes)
            if ali is None:
                raise ValueError('Alignments must be specified!')
            if cuda:
                feats = feats.cuda()
                ali = ali.cuda()
            pdfs = model(feats)
            total_loss += loss_fn(pdfs, ali).item()
    return total_loss / total_windows


def train_am_for_epoch(
        model, data_loader, optimizer, params,
        epoch=None, cuda=False, weight=None):
    '''Train an acoustic model for one epoch using cross-entropy loss

    Parameters
    ----------
    model : AcousticModel
    data_loader : TrainingDataLoader
    optimizer : torch.optim.Optimizer
    params : TrainingParams
    epoch : int, optional
        The epoch we are running. If unset, does not touch `data_loader`'s
        epoch
    cuda : bool, optional
        Whether to move the model/data to the GPU
    weight : FloatTensor, optional
        Relative weights to assign to each class. If unset, weights are
        uniform

    Returns
    -------
    running_loss : float
        The batch-averaged cross-entropy loss for the epoch
    '''
    epoch_loss = 0.
    total_batches = 0
    if epoch is None:
        epoch = data_loader.epoch
    else:
        data_loader.epoch = epoch
    if cuda:
        model.cuda()
    else:
        model.cpu()
    # the call to train here actually shepherds the optimizer parameters over
    # to
    model.train()
    if params.seed is not None:
        torch.manual_seed(params.seed + epoch)
    model.dropout = params.dropout_prob
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    for feats, ali in data_loader:
        if cuda:
            feats = feats.cuda()
            ali = ali.cuda()
        optimizer.zero_grad()
        pdfs = model(feats)
        loss = loss_fn(pdfs, ali)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total_batches += 1
    return epoch_loss / total_batches
