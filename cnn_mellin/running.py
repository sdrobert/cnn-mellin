'''Functions involved in running the models'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import param
import pydrobert.torch.training as training

from pydrobert.torch.util import optimizer_to

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def get_am_alignment_cross_entropy(
        model, data_loader, device='cpu', weight=None):
    '''Get the mean cross entropy of alignments over a data set

    Arguments
    ---------
    model : AcousticModel
    data_loader : EvaluationDataLoader
    device : torch.device or str, optional
        What device should the model/data be on
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
    if weight is not None:
        weight = weight.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, reduction='sum')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for feats, ali, feat_sizes, _ in data_loader:
            total_windows += sum(feat_sizes)
            if ali is None:
                raise ValueError('Alignments must be specified!')
            feats = feats.to(device)
            ali = ali.to(device)
            pdfs = model(feats)
            total_loss += loss_fn(pdfs, ali).item()
    return total_loss / total_windows


class TrainingEpochParams(param.Parameterized):
    seed = param.Integer(
        None,
        doc='The seed used to seed PyTorch at every epoch of training. Will '
        'control things like dropout masks. Will be incremented at every '
        'epoch. If unset, will not touch the torch seed.')
    dropout_prob = param.Magnitude(
        0.,
        doc='The model dropout probability'
    )


def train_am_for_epoch(
        model, data_loader, optimizer, params,
        epoch=None, device='cpu', weight=None):
    '''Train an acoustic model for one epoch using cross-entropy loss

    Parameters
    ----------
    model : AcousticModel
    data_loader : pydrobert.torch.data.TrainingDataLoader
    params : TrainingEpochParams
    optimizer : torch.optim.Optimizer
    init_seed : int, optional
        The initial training seed. After every epoch, the torch seed will
        be set to ``init_seed + epoch``. If unset, does not touch torch
        seed
    epoch : int, optional
        The epoch we are running. If unset, does not touch `data_loader`'s
        epoch
    device : torch.device or str, optional
        On what device should the model/data be on
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
    model = model.to(device)
    optimizer_to(optimizer, device)
    if weight is not None:
        weight = weight.to(device)
    model.train()
    if params.seed is not None:
        torch.manual_seed(params.seed + epoch)
    model.dropout = params.dropout_prob
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    for feats, ali in data_loader:
        feats = feats.to(device)
        ali = ali.to(device)
        optimizer.zero_grad()
        pdfs = model(feats)
        loss = loss_fn(pdfs, ali)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total_batches += 1
    return epoch_loss / total_batches
