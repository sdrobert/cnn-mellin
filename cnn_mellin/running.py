'''Functions involved in running the models'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import warnings

import torch
import param
import pydrobert.torch.training as training
import pydrobert.torch.data as data
import cnn_mellin.models as models

from pydrobert.torch.util import optimizer_to

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


def get_am_alignment_cross_entropy(
        model, data_loader, device='cpu', weight=None):
    '''Get the mean cross entropy of alignments over a data set

    Parameters
    ----------
    model : AcousticModel
    data_loader : pydrobert.torch.data.EvaluationDataLoader
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
            joint = model(feats)
            total_loss += loss_fn(joint, ali).item()
    return total_loss / total_windows


def write_am_pdfs(model, data_loader, log_prior, device='cpu', pdfs_dir=None):
    '''Write emission probabilities for a data set

    Parameters
    ----------
    model : AcousticModel
    data_loader : pydrobert.torch.data.EvaluationDataLoader
    log_prior : FloatTensor
        A prior distribution over the targets (senones), in natural logarithm.
        `log_prior` is necessary for converting the joint distribution of
        input and targets produced by the acoustic model into the probabilities
        of inputs conditioned on the targets
    device : torch.device or str, optional
        What device to perform computations on. The pdfs will always be saved
        as (cpu) ``torch.FloatTensor``s
    pdfs_dir : str or None, optional
        If set, pdfs will be written to this directory. Otherwise, pdfs will
        be written to the `data_loader`'s ``data_dir + '/pdfs'``
    '''
    model = model.to(device)
    log_prior = log_prior.to(device)
    model.eval()
    with torch.no_grad():
        for feats, _, feat_sizes, utt_ids in data_loader:
            feats = feats.to(device)
            y = model(feats)
            joint = torch.nn.functional.log_softmax(y, dim=1)
            pdf = joint - log_prior
            for feat_size, utt_id in zip(feat_sizes, utt_ids):
                pdf_utt = pdf[:feat_size]
                data_loader.data_source.write_pdf(
                    utt_id, pdf_utt, pdfs_dir=pdfs_dir)
                pdf = pdf[feat_size:]


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


class TrainingParams(TrainingEpochParams, training.TrainingStateParams):
    optimizer = param.ObjectSelector(
        torch.optim.Adam, objects={
            'adam': torch.optim.Adam,
            'adadelta': torch.optim.Adadelta,
            'adagrad': torch.optim.Adagrad,
            'sgd': torch.optim.SGD,
        },
        doc='The optimizer to train with'
    )
    weight_decay = param.Number(
        0, bounds=(0, None),
        doc='The L2 penalty to apply to weights'
    )


def train_am(
        model_params, training_params, train_dir, train_params, val_dir,
        val_params, state_dir=None, state_csv=None, weight=None, device='cpu',
        train_num_data_workers=os.cpu_count() - 1, print_epochs=True):
    '''Train an acoustic model for multiple epochs

    Parameters
    ----------
    model_params : AcousticModelParams
        Parameters used to configure the model
    training_params : TrainingParams
        Parameters used to configure the training process
    train_dir : str
        The path to the training data directory
    train_params : pydrobert.data.ContextWindowDataSetParams
        Parameters describing the training data
    val_dir : str
        The path to the validation data directory
    val_params : pydrobert.data.ContextWindowDataSetParams
        Parameters describing the validation data
    state_dir : str, optional
        If set, model and optimizer states will be stored in this directory
    state_csv : str, optional
        If set, training history will be read and written from this file.
        Training will resume from the last epoch, if applicable.
    weight : FloatTensor, optional
        If set, training and validation loss will be weighed according to this
        vector, which shares a length with the number of targets
    train_num_data_workers: int, optional
        The number of worker threads to spawn to serve training data. 0 means
        data are served on the main thread. The default is one fewer than the
        number of CPUs available
    print_epochs : bool, optional
        Print the results of each epoch, and their timings, to stdout

    Returns
    -------
    model : AcousticModel
        The trained model
    '''
    if train_params.context_left != val_params.context_right:
        raise ValueError(
            'context_left does not match for train_params and val_params')
    if train_params.context_right != val_params.context_right:
        raise ValueError(
            'context_right does not match for train_params and val_params')
    if weight is not None and len(weight) != model_params.target_dim:
        raise ValueError(
            'weight tensor does not match model_params.target_dim')
    model = models.AcousticModel(
        model_params,
        1 + train_params.context_left + train_params.context_right,
    )
    optimizer = training_params.optimizer(
        model.parameters(),
        weight_decay=training_params.weight_decay,
    )
    device = torch.device(device)
    train_data = data.ContextWindowTrainingDataLoader(
        train_dir, train_params,
        pin_memory=(device.type == 'cuda'),
        num_workers=train_num_data_workers,
    )
    val_data = data.ContextWindowEvaluationDataLoader(
        val_dir, val_params,
        pin_memory=(device.type == 'cuda'),
    )
    controller = training.TrainingStateController(
        training_params,
        state_csv_path=state_csv,
        state_dir=state_dir,
    )
    controller.load_model_and_optimizer_for_epoch(
        model, optimizer, controller.get_last_epoch())
    min_epoch = controller.get_last_epoch() + 1
    num_epochs = training_params.num_epochs
    if num_epochs is None:
        epoch_it = icount(min_epoch)
        if not training_params.early_stopping_threshold:
            warnings.warn(
                'Neither a maximum number of epochs nor an early stopping '
                'threshold have been set. Training will continue indefinitely')
    else:
        epoch_it = range(min_epoch, num_epochs + 1)
    for epoch in epoch_it:
        epoch_start = time.time()
        train_loss = train_am_for_epoch(
            model,
            train_data,
            optimizer,
            training_params,
            epoch=epoch,
            device=device,
            weight=weight
        )
        val_loss = get_am_alignment_cross_entropy(
            model,
            val_data,
            device=device,
            weight=weight
        )
        if print_epochs:
            print('epoch {:03d} ({:.03f}s): train={:e} val={:e}'.format(
                epoch, time.time() - epoch_start, train_loss, val_loss))
        if not controller.update_for_epoch(
                model, optimizer, train_loss, val_loss):
            break
    return model
