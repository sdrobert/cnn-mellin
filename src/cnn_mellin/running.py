"""Functions involved in running the models"""

# import os

# import time
# import warnings

# from itertools import count as icount

from typing import Any, Callable, Optional, Sequence
import torch
import param
import pydrobert.torch.training as training

import pydrobert.torch.data as data
import cnn_mellin.models as models


class MyTrainingStateParams(training.TrainingStateParams):
    optimizer = param.ObjectSelector(
        torch.optim.Adam,
        objects={
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rms": torch.optim.RMSprop,
        },
        doc="Which method of gradient descent to perform",
    )
    dropout_prob = param.Magnitude(0.0, doc="The model dropout probability")

    @classmethod
    def get_tunable(cls):
        return super().get_tunable() | {"dropout_prob", "optimizer"}

    @classmethod
    def suggest_params(cls, trial, base, only, prefix):
        params = super().suggest_params(trial, base=base, only=only, prefix=prefix)

        if only is None:
            only = cls.get_tunable()
        if "dropout_prob" in only:
            params.dropout_prob = trial.suggest_uniform(
                prefix + "dropout_prob", 0.0, 1.0
            )
        if "optimizer" in only:
            dict_ = params.param.params().get_range()
            chosen = trial.suggest_categorical(prefix + "optimizer", sorted(dict_))
            params.optimizer = dict_[chosen]

        return params


def train_am_for_epoch(
    model: models.AcousticModel,
    loader: data.SpectTrainingDataLoader,
    optimizer: torch.optim.Optimizer,
    controller: training.TrainingStateController,
    params: Optional[MyTrainingStateParams] = None,
    epoch: Optional[int] = None,
    batch_callbacks: Sequence[Callable[[int, float], Any]] = tuple(),
) -> float:
    """Train the acoustic model with a CTC objective"""

    if epoch is None:
        epoch = controller.get_last_epoch() + 1
    loader.epoch = epoch

    if params is None:
        params = controller.params
        if not isinstance(params, MyTrainingStateParams):
            raise ValueError(
                "if params = None, controller.params must be MyTrainingStateParams"
            )

    if loader.batch_first:
        raise ValueError("data loader batch_first must be false")

    device = model.lift.log_tau.device
    non_blocking = device.type == "cpu" or loader.pin_memory

    controller.load_model_and_optimizer_for_epoch(model, optimizer, epoch - 1, True)

    model.dropout = params.dropout_prob
    model.train()

    loss_fn = torch.nn.CTCLoss(blank=model.target_dim - 1, zero_infinity=True)

    total_loss = 0.0
    total_batches = 0
    for feats, _, refs, feat_lens, ref_lens in loader:
        feats = feats.to(device, non_blocking=non_blocking)
        refs = refs.to(device, non_blocking=non_blocking)
        feat_lens = feat_lens.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        if refs.dim() == 3:
            refs = refs[..., 0]
        refs = refs.t()  # (N, S)
        logits, lens = model(feats, feat_lens)
        logits = torch.nn.functional.log_softmax(logits, 2)
        loss = loss_fn(logits, refs, lens, ref_lens)
        loss.backward()
        loss = loss.item()
        del feats, refs, feat_lens, ref_lens, lens, logits
        optimizer.step()
        for callback in batch_callbacks:
            callback(total_batches, loss)
        total_loss += loss
        total_batches += 1

    return total_loss


# def train_am_for_epoch(
#     model,
#     data_loader,
#     optimizer,
#     params,
#     epoch=None,
#     device="cpu",
#     weight=None,
#     batch_callbacks=tuple(),
# ):
#     """Train an acoustic model for one epoch using cross-entropy loss

#     Parameters
#     ----------
#     model : AcousticModel
#     data_loader : pydrobert.torch.data.TrainingDataLoader
#     params : TrainingEpochParams
#     optimizer : torch.optim.Optimizer
#     init_seed : int, optional
#         The initial training seed. After every epoch, the torch seed will
#         be set to ``init_seed + epoch``. If unset, does not touch torch
#         seed
#     epoch : int, optional
#         The epoch we are running. If unset, does not touch `data_loader`'s
#         epoch
#     device : torch.device or str, optional
#         On what device should the model/data be on
#     weight : FloatTensor, optional
#         Relative weights to assign to each class.
#         `params.weigh_training_samples` must also be ``True`` to use during
#         training
#     batch_callbacks : sequence, optional
#         A sequence of callbacks to perform after every batch. Callbacks should
#         accept two positional arguments: one for the batch number, the other
#         for the batch training loss

#     Returns
#     -------
#     running_loss : float
#         The batch-averaged cross-entropy loss for the epoch
#     """
#     device = torch.device(device)
#     epoch_loss = 0.0
#     total_batches = 0
#     if epoch is None:
#         epoch = data_loader.epoch
#     else:
#         data_loader.epoch = epoch
#     model = model.to(device)
#     optimizer_to(optimizer, device)
#     if params.weigh_training_samples:
#         if weight is None:
#             warnings.warn(
#                 "{}.weigh_training_samples is True, but no weight vector was "
#                 "passed to train_am_for_epoch".format(params.name)
#             )
#         else:
#             weight = weight.to(device)
#     else:
#         weight = None
#     model.train()
#     if params.seed is not None:
#         torch.manual_seed(params.seed + epoch)
#     model.dropout = params.dropout_prob
#     loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
#     non_blocking = device.type == "cpu" or data_loader.pin_memory
#     for feats, ali in data_loader:
#         optimizer.zero_grad()
#         feats = feats.to(device, non_blocking=non_blocking)
#         ali = ali.to(device, non_blocking=non_blocking)
#         loss = loss_fn(model(feats), ali)
#         loss_val = loss.item()
#         epoch_loss += loss_val
#         loss.backward()
#         optimizer.step()
#         del feats, ali, loss
#         for callback in batch_callbacks:
#             callback(total_batches, loss_val)
#         total_batches += 1
#     return epoch_loss / total_batches


# class TrainingParams(TrainingEpochParams, training.TrainingStateParams):
#     optimizer = param.ObjectSelector(
#         "adam",
#         objects=["adam", "adadelta", "adagrad", "sgd"],
#         doc="The optimizer to train with",
#     )
#     weight_decay = param.Number(
#         0, bounds=(0, None), doc="The L2 penalty to apply to weights"
#     )


# def train_am(
#     model_params,
#     training_params,
#     train_dir,
#     train_params,
#     val_dir,
#     val_params,
#     state_dir=None,
#     state_csv=None,
#     weight=None,
#     device="cpu",
#     train_num_data_workers=os.cpu_count() - 1,
#     print_epochs=True,
#     batch_callbacks=tuple(),
#     epoch_callbacks=tuple(),
# ):
#     """Train an acoustic model for multiple epochs

#     Parameters
#     ----------
#     model_params : AcousticModelParams
#         Parameters used to configure the model
#     training_params : TrainingParams
#         Parameters used to configure the training process
#     train_dir : str
#         The path to the training data directory
#     train_params : pydrobert.data.ContextWindowDataSetParams
#         Parameters describing the training data
#     val_dir : str
#         The path to the validation data directory
#     val_params : pydrobert.data.ContextWindowDataSetParams
#         Parameters describing the validation data
#     state_dir : str, optional
#         If set, model and optimizer states will be stored in this directory
#     state_csv : str, optional
#         If set, training history will be read and written from this file.
#         Training will resume from the last epoch, if applicable.
#     weight : FloatTensor, optional
#         Relative weights to assign to each class.
#         `train_params.weigh_training_samples` must also be ``True`` to use
#         during training
#     train_num_data_workers : int, optional
#         The number of worker threads to spawn to serve training data. 0 means
#         data are served on the main thread. The default is one fewer than the
#         number of CPUs available
#     print_epochs : bool, optional
#         Print the results of each epoch, and their timings, to stdout
#     batch_callbacks : sequence, optional
#         A sequence of functions that accepts two positional arguments: the
#         first is a batch index; the second is the batch loss. The functions
#         are called after a batch's loss has been propagated
#     epoch_callbacks : sequence, optional
#         A list of functions that accepts a dictionary as a positional argument,
#         containing:
#         - 'epoch': the current epoch
#         - 'train_loss': the training loss for the epoch
#         - 'val_loss': the validation loss for the epoch
#         - 'model': the model trained to this point
#         - 'optimizer': the optimizer at this point
#         - 'controller': the underlying training state controller
#         - 'will_stop': whether the controller thinks training should stop
#         The callback occurs after the controller has been updated for the epoch

#     Returns
#     -------
#     model : AcousticModel
#         The trained model
#     """
#     if train_params.context_left != val_params.context_left:
#         raise ValueError("context_left does not match for train_params and val_params")
#     if train_params.context_right != val_params.context_right:
#         raise ValueError("context_right does not match for train_params and val_params")
#     if weight is not None and len(weight) != model_params.target_dim:
#         raise ValueError("weight tensor does not match model_params.target_dim")
#     device = torch.device(device)
#     train_data = data.ContextWindowTrainingDataLoader(
#         train_dir,
#         train_params,
#         pin_memory=(device.type == "cuda"),
#         num_workers=train_num_data_workers,
#     )
#     assert len(train_data)
#     val_data = data.ContextWindowEvaluationDataLoader(
#         val_dir, val_params, pin_memory=(device.type == "cuda"),
#     )
#     assert len(val_data)
#     model = models.AcousticModel(
#         model_params, 1 + train_params.context_left + train_params.context_right,
#     )
#     if training_params.optimizer == "adam":
#         optimizer = torch.optim.Adam
#     elif training_params.optimizer == "adadelta":
#         optimizer = torch.optim.Adadelta
#     elif training_params.optimizer == "adagrad":
#         optimizer = torch.optim.Adagrad
#     else:
#         optimizer = torch.optim.SGD
#     optimizer_kwargs = {
#         "weight_decay": training_params.weight_decay,
#     }
#     if training_params.log10_learning_rate is not None:
#         optimizer_kwargs["lr"] = 10 ** training_params.log10_learning_rate
#     optimizer = optimizer(model.parameters(), **optimizer_kwargs)
#     controller = training.TrainingStateController(
#         training_params, state_csv_path=state_csv, state_dir=state_dir,
#     )
#     controller.load_model_and_optimizer_for_epoch(
#         model, optimizer, controller.get_last_epoch()
#     )
#     min_epoch = controller.get_last_epoch() + 1
#     num_epochs = training_params.num_epochs
#     if num_epochs is None:
#         epoch_it = icount(min_epoch)
#         if not training_params.early_stopping_threshold:
#             warnings.warn(
#                 "Neither a maximum number of epochs nor an early stopping "
#                 "threshold have been set. Training will continue indefinitely"
#             )
#     else:
#         epoch_it = range(min_epoch, num_epochs + 1)
#     for epoch in epoch_it:
#         epoch_start = time.time()
#         train_loss = train_am_for_epoch(
#             model,
#             train_data,
#             optimizer,
#             training_params,
#             epoch=epoch,
#             device=device,
#             weight=weight,
#             batch_callbacks=batch_callbacks,
#         )
#         val_loss = get_am_alignment_cross_entropy(
#             model, val_data, device=device, weight=weight
#         )
#         if print_epochs:
#             print(
#                 "epoch {:03d} ({:.03f}s): train={:e} val={:e}".format(
#                     epoch, time.time() - epoch_start, train_loss, val_loss
#                 )
#             )
#         will_stop = (
#             not controller.update_for_epoch(model, optimizer, train_loss, val_loss)
#             or epoch == num_epochs
#         )
#         if epoch_callbacks:
#             callback_dict = {
#                 "epoch": epoch,
#                 "train_loss": train_loss,
#                 "val_loss": val_loss,
#                 "model": model,
#                 "optimizer": optimizer,
#                 "controller": controller,
#                 "will_stop": will_stop,
#             }
#             for callback in epoch_callbacks:
#                 callback(callback_dict)
#         if will_stop:
#             break
#     return model
