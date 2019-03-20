'''Functions involved in running the models'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import warnings

from itertools import count as icount

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
    device = torch.device(device)
    if not len(data_loader):
        raise ValueError('There must be at least one batch of data!')
    total_windows = 0
    total_loss = 0
    if weight is not None:
        weight = weight.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, reduction='sum')
    model = model.to(device)
    model.eval()
    non_blocking = device.type == 'cpu' or data_loader.pin_memory
    with torch.no_grad():
        for feats, ali, feat_sizes, _ in data_loader:
            total_windows += sum(feat_sizes)
            if ali is None:
                raise ValueError('Alignments must be specified!')
            feats = feats.to(device, non_blocking=non_blocking)
            ali = ali.to(device, non_blocking=non_blocking)
            joint = model(feats)
            loss = loss_fn(joint, ali)
            total_loss += loss.item()
            del feats, ali, joint, loss
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
    device = torch.device(device)
    model = model.to(device)
    log_prior = log_prior.to(device)
    model.eval()
    non_blocking = device.type == 'cpu' or data_loader.pin_memory
    with torch.no_grad():
        for feats, _, feat_sizes, utt_ids in data_loader:
            feats = feats.to(device, non_blocking=non_blocking)
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
    weigh_training_samples = param.Boolean(
        True,
        doc='If a weight tensor is provided during training and this is '
        '``True``, per-frame loss will be weighed with the index matching '
        'the target'
    )


def train_am_for_epoch(
        model, data_loader, optimizer, params,
        epoch=None, device='cpu', weight=None, batch_callbacks=tuple()):
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
        Relative weights to assign to each class.
        `params.weigh_training_samples` must also be ``True`` to use during
        training
    batch_callbacks : sequence, optional
        A sequence of callbacks to perform after every batch. Callbacks should
        accept two positional arguments: one for the batch number, the other
        for the batch training loss

    Returns
    -------
    running_loss : float
        The batch-averaged cross-entropy loss for the epoch
    '''
    device = torch.device(device)
    epoch_loss = 0.
    total_batches = 0
    if epoch is None:
        epoch = data_loader.epoch
    else:
        data_loader.epoch = epoch
    model = model.to(device)
    optimizer_to(optimizer, device)
    if params.weigh_training_samples:
        if weight is None:
            warnings.warn(
                '{}.weigh_training_samples is True, but no weight vector was '
                'passed to train_am_for_epoch'.format(params.name))
        else:
            weight = weight.to(device)
    else:
        weight = None
    model.train()
    if params.seed is not None:
        torch.manual_seed(params.seed + epoch)
    model.dropout = params.dropout_prob
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    non_blocking = device.type == 'cpu' or data_loader.pin_memory
    for feats, ali in data_loader:
        optimizer.zero_grad()
        feats = feats.to(device, non_blocking=non_blocking)
        ali = ali.to(device, non_blocking=non_blocking)
        loss = loss_fn(model(feats), ali)
        loss_val = loss.item()
        epoch_loss += loss_val
        loss.backward()
        optimizer.step()
        del feats, ali, loss
        for callback in batch_callbacks:
            callback(total_batches, loss_val)
        total_batches += 1
    return epoch_loss / total_batches


class TrainingParams(TrainingEpochParams, training.TrainingStateParams):
    optimizer = param.ObjectSelector(
        'adam', objects=['adam', 'adadelta', 'adagrad', 'sgd'],
        doc='The optimizer to train with'
    )
    weight_decay = param.Number(
        0, bounds=(0, None),
        doc='The L2 penalty to apply to weights'
    )


def train_am(
        model_params, training_params, train_dir, train_params, val_dir,
        val_params, state_dir=None, state_csv=None, weight=None, device='cpu',
        train_num_data_workers=os.cpu_count() - 1, print_epochs=True,
        batch_callbacks=tuple(), epoch_callbacks=tuple()):
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
        Relative weights to assign to each class.
        `train_params.weigh_training_samples` must also be ``True`` to use
        during training
    train_num_data_workers : int, optional
        The number of worker threads to spawn to serve training data. 0 means
        data are served on the main thread. The default is one fewer than the
        number of CPUs available
    print_epochs : bool, optional
        Print the results of each epoch, and their timings, to stdout
    batch_callbacks : sequence, optional
        A sequence of functions that accepts two positional arguments: the
        first is a batch index; the second is the batch loss. The functions
        are called after a batch's loss has been propagated
    epoch_callbacks : sequence, optional
        A list of functions that accepts a dictionary as a positional argument,
        containing:
        - 'epoch': the current epoch
        - 'train_loss': the training loss for the epoch
        - 'val_loss': the validation loss for the epoch
        - 'model': the model trained to this point
        - 'optimizer': the optimizer at this point
        - 'controller': the underlying training state controller
        - 'will_stop': whether the controller thinks training should stop
        The callback occurs after the controller has been updated for the epoch

    Returns
    -------
    model : AcousticModel
        The trained model
    '''
    if train_params.context_left != val_params.context_left:
        raise ValueError(
            'context_left does not match for train_params and val_params')
    if train_params.context_right != val_params.context_right:
        raise ValueError(
            'context_right does not match for train_params and val_params')
    if weight is not None and len(weight) != model_params.target_dim:
        raise ValueError(
            'weight tensor does not match model_params.target_dim')
    device = torch.device(device)
    train_data = data.ContextWindowTrainingDataLoader(
        train_dir, train_params,
        pin_memory=(device.type == 'cuda'),
        num_workers=train_num_data_workers,
    )
    assert len(train_data)
    val_data = data.ContextWindowEvaluationDataLoader(
        val_dir, val_params,
        pin_memory=(device.type == 'cuda'),
    )
    assert len(val_data)
    model = models.AcousticModel(
        model_params,
        1 + train_params.context_left + train_params.context_right,
    )
    if training_params.optimizer == 'adam':
        optimizer = torch.optim.Adam
    elif training_params.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta
    elif training_params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad
    else:
        optimizer = torch.optim.SGD
    optimizer_kwargs = {
        'weight_decay': training_params.weight_decay,
    }
    if training_params.log10_learning_rate is not None:
        optimizer_kwargs['lr'] = 10 ** training_params.log10_learning_rate
    optimizer = optimizer(
        model.parameters(),
        **optimizer_kwargs
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
            weight=weight,
            batch_callbacks=batch_callbacks,
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
        will_stop = not controller.update_for_epoch(
            model, optimizer, train_loss, val_loss) or epoch == num_epochs
        if epoch_callbacks:
            callback_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model': model,
                'optimizer': optimizer,
                'controller': controller,
                'will_stop': will_stop,
            }
            for callback in epoch_callbacks:
                callback(callback_dict)
        if will_stop:
            break
    return model
