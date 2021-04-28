"""Functions involved in running the models"""

# import os

# import time
# import warnings

# from itertools import count as icount

from typing import Any, Callable, Optional, Sequence, Tuple
import torch
import param
import pydrobert.torch.training as training
import pydrobert.torch.util as util

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


# FIXME(sdrobert): use pydrobert-pytorch copy when PR merged
def ctc_greedy_search(
    logits: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    blank_idx: int = -1,
    batch_first: bool = False,
    is_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.dim() != 3:
        raise RuntimeError("logits must be 3-dimensional")
    V = logits.size(2)
    if blank_idx < -V or blank_idx > (V - 1):
        raise IndexError(
            "Blank index out of range (expected to be in the range of "
            f"[-{V},{V-1}], but got {blank_idx})"
        )
    blank_idx = (blank_idx + V) % V
    if not is_probs:
        # normalize
        logits = logits.log_softmax(2)
    max_, argmax = logits.max(2)
    keep_mask = argmax != blank_idx
    if batch_first:
        keep_mask[:, 1:] &= argmax[:, 1:] != argmax[:, :-1]
        seq_dim = 1
    else:
        keep_mask[1:] &= argmax[1:] != argmax[:-1]
        seq_dim = 0
    seq_size = argmax.size(seq_dim)
    if in_lens is not None:
        in_len_mask = torch.arange(seq_size, device=argmax.device).unsqueeze(
            1 - seq_dim
        ) < in_lens.unsqueeze(seq_dim)
        keep_mask = keep_mask & in_len_mask
        if is_probs:
            max_ = max_.masked_fill(~in_len_mask, 1.0)
        else:
            max_ = max_.masked_fill(~in_len_mask, 0.0)
        del in_len_mask
    out_lens = keep_mask.long().sum(seq_dim)
    data = argmax.masked_select(keep_mask)
    out_len_mask = torch.arange(seq_size, device=argmax.device).unsqueeze(
        1 - seq_dim
    ) < out_lens.unsqueeze(seq_dim)
    if is_probs:
        max_ = max_.prod(seq_dim)
    else:
        max_ = max_.sum(seq_dim)
    return max_, argmax.masked_scatter_(out_len_mask, data), out_lens


def greedy_decode_am(
    model: models.AcousticModel, loader: data.SpectEvaluationDataLoader,
) -> float:
    """Determine average error rate on eval set using greedy decoding

    This should only be used in the training process for the dev set. A prefix search
    outputting trn files should be preferred for the test set.
    """

    if loader.batch_first:
        raise ValueError("data loader batch_first must be false")

    device = model.lift.log_tau.device
    non_blocking = device.type == "cpu" or loader.pin_memory

    model.eval()

    total_errs = 0.0
    total_refs = 0

    with torch.no_grad():
        for feats, _, refs, feat_lens, ref_lens, _ in loader:
            feats = feats.to(device, non_blocking=non_blocking)
            refs = refs.to(device, non_blocking=non_blocking)
            if refs.dim() == 3:
                refs = refs[..., 0]
            feat_lens = feat_lens.to(device, non_blocking=non_blocking)
            ref_lens = ref_lens.to(device, non_blocking=non_blocking)
            logits, lens = model(feats, feat_lens)
            _, hyps, lens_ = ctc_greedy_search(logits, lens)
            ref_len_mask = (
                torch.arange(refs.size(0), device=device).unsqueeze(1) >= ref_lens
            )
            refs.masked_fill_(ref_len_mask, -1)
            hyp_len_mask = (
                torch.arange(hyps.size(0), device=device).unsqueeze(1) >= lens_
            )
            hyps.masked_fill_(hyp_len_mask, -1)
            er = util.error_rate(refs, hyps, -1)
            total_errs += er.sum().item()
            total_refs += ref_lens.sum().item()
            del (
                feats,
                refs,
                feat_lens,
                ref_lens,
                logits,
                lens,
                hyps,
                lens_,
                ref_len_mask,
                hyp_len_mask,
                er,
            )

    return total_errs / total_refs


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
