"""Functions involved in running the models"""

import os
import sys

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import param
import pydrobert.torch.training as training
import pydrobert.torch.util as util

import pydrobert.torch.data as data
import cnn_mellin.models as models

from tqdm import tqdm


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
    max_time_warp = param.Number(
        80.0,
        bounds=(0.0, None),
        doc="SpecAugment max time dimension warp during training",
    )
    max_freq_warp = param.Number(
        0.0,
        bounds=(0.0, None),
        doc="SpecAugment max frequency dimension warp during training",
    )
    max_time_mask = param.Integer(
        100,
        bounds=(0, None),
        doc="SpecAugment max number of sequential frames in time to mask",
    )


def train_am_for_epoch(
    model: models.AcousticModel,
    loader: data.SpectTrainingDataLoader,
    optimizer: torch.optim.Optimizer,
    controller: training.TrainingStateController,
    params: Optional[MyTrainingStateParams] = None,
    epoch: Optional[int] = None,
    quiet: bool = True,
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

    if epoch == 1 or (controller.state_dir and controller.state_csv_path):
        controller.load_model_and_optimizer_for_epoch(model, optimizer, epoch - 1, True)

    model.dropout = params.dropout_prob
    model.train()

    loss_fn = torch.nn.CTCLoss(blank=model.target_dim - 1, zero_infinity=True)

    if not quiet:
        loader = tqdm(loader)

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
    if not batch_first:
        # the masked_fill/scatter_ logic won't work if it isn't batch_first
        logits = logits.transpose(0, 1)
    max_, argmax = logits.max(2)
    keep_mask = argmax != blank_idx
    keep_mask[:, 1:] &= argmax[:, 1:] != argmax[:, :-1]
    seq_size = argmax.size(1)
    if in_lens is not None:
        in_len_mask = torch.arange(seq_size, device=argmax.device).unsqueeze(
            0
        ) < in_lens.unsqueeze(1)
        keep_mask = keep_mask & in_len_mask
        if is_probs:
            max_ = max_.masked_fill(~in_len_mask, 1.0)
        else:
            max_ = max_.masked_fill(~in_len_mask, 0.0)
        del in_len_mask
    out_lens = keep_mask.long().sum(1)
    data = argmax.masked_select(keep_mask)
    out_len_mask = torch.arange(seq_size, device=argmax.device).unsqueeze(
        0
    ) < out_lens.unsqueeze(1)
    if is_probs:
        max_ = max_.prod(1)
    else:
        max_ = max_.sum(1)
    argmax = argmax.masked_scatter_(out_len_mask, data)
    if not batch_first:
        argmax = argmax.t()
    return max_, argmax, out_lens


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
            refs = refs.masked_fill_(ref_len_mask, -1)
            hyp_len_mask = (
                torch.arange(hyps.size(0), device=device).unsqueeze(1) >= lens_
            )
            hyps = hyps.masked_fill_(hyp_len_mask, -1)
            er = util.error_rate(refs, hyps, eos=-1, norm=False)
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


def get_filts_and_classes(train_dir: str) -> Tuple[int, int]:
    """Given the training partition directory, determine the number of filters/classes

    Always use training partition info! Number of filts in test partition might be the
    same, but maybe not the number of classes.

    Returns
    -------
    num_filts, num_classes : int, int
    """
    part_name = os.path.basename(train_dir)
    ext_file = os.path.join(os.path.dirname(train_dir), "ext", f"{part_name}.info.ark")
    if not os.path.isfile(ext_file):
        raise ValueError(f"Could not find '{ext_file}'")
    dict_ = dict()
    with open(ext_file) as file_:
        for line in file_:
            k, v = line.strip().split()
            dict_[k] = v
    return int(dict_["num_filts"]), int(dict_["max_ref_class"]) + 1


def train_am(
    model_params: models.AcousticModelParams,
    training_params: MyTrainingStateParams,
    data_params: data.SpectDataSetParams,
    train_dir: str,
    dev_dir: str,
    model_dir: Optional[str] = None,
    device: Union[torch.device, str] = "cpu",
    num_data_workers: int = os.cpu_count() - 1,
    epoch_callbacks: Sequence[Callable[[int, float, float], Any]] = tuple(),
    quiet: bool = True,
) -> Tuple[models.AcousticModel, float]:

    device = torch.device(device)

    num_filts, num_classes = get_filts_and_classes(train_dir)

    model = models.AcousticModel(num_filts, num_classes + 1, model_params)
    model.to(device)
    optimizer = training_params.optimizer(model.parameters(), lr=1e-4)

    if model_dir is not None:
        state_dir = os.path.join(model_dir, "training")
        state_csv = os.path.join(model_dir, "hist.csv")
    else:
        state_dir = state_csv = None

    controller = training.TrainingStateController(
        training_params, state_csv, state_dir, warn=not quiet
    )

    train_loader = data.SpectTrainingDataLoader(
        train_dir,
        data_params,
        batch_first=False,
        seed=training_params.seed,
        pin_memory=device.type == "cuda",
        num_workers=num_data_workers,
    )
    dev_loader = data.SpectEvaluationDataLoader(
        dev_dir, data_params, batch_first=False, num_workers=num_data_workers
    )

    dev_err = float("inf")
    epoch = controller.get_last_epoch() + 1
    while controller.continue_training(epoch - 1):
        if not quiet:
            print(f"Training epoch {epoch}...", file=sys.stderr)
        train_loss = train_am_for_epoch(
            model, train_loader, optimizer, controller, training_params, epoch, quiet
        )
        if not quiet:
            print("Epoch completed. Determining average error rate...", file=sys.stderr)
        dev_err = greedy_decode_am(model, dev_loader)
        controller.update_for_epoch(model, optimizer, train_loss, dev_err, epoch)
        if not quiet:
            print(
                f"Train loss: {train_loss:e}, dev err: {dev_err:.1%}", file=sys.stderr
            )
        for callback in epoch_callbacks:
            callback(epoch, train_loss, dev_err)
        epoch += 1

    if not quiet:
        print(f"Finished training at epoch {epoch - 1}", file=sys.stderr)

    if model_dir is not None:
        epoch = controller.get_best_epoch()
        if not quiet:
            print(
                f"Best epoch was {epoch}. Returning that model (and that error)",
                file=sys.stderr,
            )
        controller.load_model_for_epoch(model, epoch)
        dev_err = controller.get_info(epoch)["val_met"]
    elif not quiet:
        print(
            f"No history kept. Returning model from last epoch ({epoch - 1})",
            file=sys.stderr,
        )

    return model, dev_err
