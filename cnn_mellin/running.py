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
    '''
    if not len(data_loader):
        raise ValueError('There must be at least one batch of data!')
    total_frames = 0
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, reduction='sum')
    if cuda:
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        for feats, ali, feat_sizes, _ in data_loader:
            total_frames += sum(feat_sizes)
            if ali is None:
                raise ValueError('Alignments must be specified!')
            if cuda:
                feats = feats.cuda()
                ali = ali.cuda()
            pdfs = model(feats)
            total_loss += loss_fn(pdfs, ali).item()
    return total_loss / total_frames
