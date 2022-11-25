"""Functions involved in running the models"""

import os
import sys
import warnings

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import param
import pydrobert.torch.training as training
import pydrobert.torch.util as util
import pydrobert.torch.modules as layers

import pydrobert.torch.data as data
import models

from common import get_num_avail_cores
from layers import ScaledGaussianNoise
from tqdm import tqdm


# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False


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
    autocast = param.Boolean(
        False, doc="Whether to perform mixed-precision training. Only valid for CUDA"
    )
    noise_eps = param.Magnitude(
        1e-3,
        softbounds=(1e-5, 1e-1),
        doc="The proportion of gaussian noise per coefficient to add to the input",
    )
    dropout_prob = param.Magnitude(
        0.05, softbounds=(0.0, 0.2), doc="The model dropout probability for all layers",
    )
    convolutional_dropout_2d = param.Boolean(
        True, doc="If true, zero out channels instead of individual coefficients"
    )
    max_time_warp = param.Number(
        20.0,
        bounds=(0.0, None),
        softbounds=(0.0, 20.0),
        doc="SpecAugment max time dimension warp during training",
    )
    max_freq_warp = param.Number(
        0.0,
        bounds=(0.0, None),
        softbounds=(0.0, 25.0),
        doc="SpecAugment max frequency dimension warp during training",
    )
    max_time_mask = param.Integer(
        100,
        bounds=(1, None),
        softbounds=(1, 200),
        doc="SpecAugment absolute upper bound on sequential frames in time to mask per "
        "mask",
    )
    max_freq_mask = param.Integer(
        27,
        bounds=(1, None),
        softbounds=(1, 40),
        doc="SpecAgument max number of coefficients in frequency to mask per mask",
    )
    max_time_mask_proportion = param.Magnitude(
        0.04,
        softbounds=(0.01, 0.1),
        inclusive_bounds=(False, False),
        doc="SpecAugment relative upper bound on the number of sequential frames in "
        "time to mask per mask",
    )
    num_time_mask = param.Integer(
        2,
        bounds=(0, None),
        softbounds=(0, 40),
        doc="SpecAgument absolute upper bound on the number of temporal masks to apply",
    )
    num_time_mask_proportion = param.Magnitude(
        0.04,
        softbounds=(0.01, 0.1),
        inclusive_bounds=(False, False),
        doc="SpecAugment relative upper bound on the number of temporal masks to apply",
    )
    num_freq_mask = param.Integer(
        2,
        bounds=(0, None),
        softbounds=(0, 5),
        doc="SpecAugment maximum number of frequency masks to apply",
    )
    max_shift_proportion = param.Magnitude(
        0.05,
        softbounds=(0.0, 0.2),
        doc="Randomly shift audio by up to this proportion of the sequence length on "
        "either side of the sequence (total possible proportion is twice this value)",
    )

    @classmethod
    def get_tunable(cls):
        return super().get_tunable() | {
            "convolutional_dropout_2d",
            "dropout_prob",
            "max_freq_mask",
            "max_freq_warp",
            "max_shift_proportion",
            "max_time_mask_proportion",
            "max_time_mask",
            "max_time_warp",
            "noise_eps",
            "num_freq_mask",
            "num_time_mask_proportion",
            "num_time_mask",
            "optimizer",
        }

    @classmethod
    def suggest_params(cls, trial, base, only, prefix):
        if only is None:
            only = cls.get_tunable()
        only = set(only)
        params = super().suggest_params(trial, base=base, only=only, prefix=prefix)
        pdict = params.param.params()
        softbound_scale_factor = 1
        if "raw" in trial.user_attrs or (
            hasattr(trial, "study") and "raw" in trial.study.user_attrs
        ):
            params.max_freq_warp = params.max_time_warp = 0.0
            params.max_freq_mask = 1
            params.num_freq_mask = 0
            for tunable in {
                "max_time_warp",
                "max_freq_warp",
                "max_freq_mask",
                "num_freq_mask",
            }:
                if tunable in only:
                    warnings.warn(
                        f"Removing '{tunable}' from list of tunable hyperparameters "
                        "because input is raw"
                    )
                    only.remove(tunable)
            # defaults are based on 10ms shift; with 16000 samps/sec, the same length
            # is 100 times
            softbound_scale_factor = 100

        def check_and_set(
            name, use_scale_factor=False, use_log=False, low=None, high=None
        ):
            if name not in only:
                return
            entry = pdict[name]
            deft = getattr(params, name)
            if isinstance(entry, param.Number):
                if low is None:
                    low = entry.get_soft_bounds()[0]
                    assert low is not None
                if high is None:
                    high = entry.get_soft_bounds()[1]
                if use_scale_factor:
                    low *= softbound_scale_factor
                    high *= softbound_scale_factor
                if isinstance(deft, int):
                    val = trial.suggest_int(prefix + name, low, high, log=use_log)
                else:
                    val = trial.suggest_float(prefix + name, low, high, log=use_log)
            elif isinstance(entry, param.Boolean):
                val = trial.suggest_categorical(prefix + name, (True, False))
            elif isinstance(entry, param.ObjectSelector):
                range_ = entry.get_range()
                key = trial.suggest_categorical(prefix + name, tuple(range_))
                val = range_[key]
            else:
                assert False
            setattr(params, name, val)

        check_and_set("optimizer")
        check_and_set("noise_eps", False, True)
        check_and_set("dropout_prob")
        check_and_set("max_time_warp", True)
        check_and_set("max_freq_warp")
        check_and_set("num_time_mask", True)
        check_and_set("num_freq_mask")
        check_and_set("max_shift_proportion")
        if params.num_time_mask:
            check_and_set("num_time_mask_proportion")
            check_and_set("max_time_mask", True)
            check_and_set("max_time_mask_proportion")
        if params.num_freq_mask:
            check_and_set("max_freq_mask")
        if params.dropout_prob:
            check_and_set("convolutional_dropout_2d")

        return params


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

    if not loader.batch_first:
        raise ValueError("data loader batch_first must be true")

    device = next(model.parameters()).device
    non_blocking = device.type == "cpu" or loader.pin_memory
    autocast = params.autocast and device.type == "cuda"

    if epoch == 1 or (controller.state_dir and controller.state_csv_path):
        controller.load_model_and_optimizer_for_epoch(model, optimizer, epoch - 1, True)

    model.train()

    loss_fn = torch.nn.CTCLoss(blank=model.target_dim - 1, zero_infinity=True)

    noise = ScaledGaussianNoise(1, params.noise_eps)

    spec_augment = (
        layers.SpecAugment(
            params.max_time_warp,
            params.max_freq_warp,
            params.max_time_mask,
            params.max_freq_mask,
            params.max_time_mask_proportion,
            params.num_time_mask,
            params.num_time_mask_proportion,
            params.num_freq_mask,
        )
        .train()
        .to(device)
    )

    random_shift = layers.RandomShift(params.max_shift_proportion)

    if not quiet:
        loader = tqdm(loader)

    if params.seed is not None:
        torch.manual_seed(params.seed * epoch + epoch)

    total_loss = 0.0
    total_batches = 0
    scaler = torch.cuda.amp.grad_scaler.GradScaler() if autocast else None
    for feats, _, refs, feat_lens, ref_lens in loader:
        feats = feats.to(device, non_blocking=non_blocking)
        refs = refs.to(device, non_blocking=non_blocking)
        feat_lens = feat_lens.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(autocast):
            if refs.dim() == 3:
                refs = refs[..., 0]
            feats, lens = random_shift(spec_augment(noise(feats), feat_lens), feat_lens)
            logits, lens = model(
                feats, feat_lens, params.dropout_prob, params.convolutional_dropout_2d
            )
            logits = torch.nn.functional.log_softmax(logits, 2)
            loss = loss_fn(logits, refs, lens, ref_lens)
        if autocast:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss = loss.item()
        del feats, refs, feat_lens, ref_lens, lens, logits
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

    if not loader.batch_first:
        raise ValueError("data loader batch_first must be true")

    device = next(model.parameters()).device
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
            refs = refs.t()
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
    train_data_params: data.SpectDataSetParams,
    train_dir: str,
    dev_dir: str,
    ckpt_dir: Optional[str] = None,
    state_csv: Optional[str] = None,
    device: Union[torch.device, str] = "cpu",
    num_data_workers: int = get_num_avail_cores() - 1,
    epoch_callbacks: Sequence[Callable[[int, float, float], Any]] = tuple(),
    quiet: bool = True,
    dev_data_params: Optional[data.SpectDataSetParams] = None,
) -> Tuple[models.AcousticModel, float]:

    if (
        training_params.num_epochs is None
        and not training_params.early_stopping_threshold
    ):
        raise ValueError(
            "Number of epochs not set with no early stopping threshold. Training would "
            "continue forever. If this is what you want, set num_epochs to something "
            "really high"
        )

    device = torch.device(device)

    num_filts, num_classes = get_filts_and_classes(train_dir)

    model = models.AcousticModel(num_filts, num_classes + 1, model_params)
    model.to(device)
    optimizer = training_params.optimizer(model.parameters(), lr=1e-4)

    if state_csv is None and ckpt_dir is not None:
        state_csv = os.path.join(ckpt_dir, "hist.csv")

    controller = training.TrainingStateController(
        training_params, state_csv, ckpt_dir, warn=not quiet
    )

    train_loader = data.SpectTrainingDataLoader(
        train_dir,
        train_data_params,
        batch_first=True,
        seed=training_params.seed,
        pin_memory=False,  # https://github.com/pytorch/pytorch/issues/57273 for now
        num_workers=num_data_workers,
    )
    dev_loader = data.SpectEvaluationDataLoader(
        dev_dir,
        train_data_params if dev_data_params is None else dev_data_params,
        batch_first=True,
        num_workers=num_data_workers,
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
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not quiet:
        print(f"Finished training at epoch {epoch - 1}", file=sys.stderr)

    if ckpt_dir is not None:
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


@torch.no_grad()
def compute_logits(
    model_params: models.AcousticModelParams,
    data_params: data.SpectDataParams,
    model_pt: Any,
    data_dir: str,
    logit_dir: str,
    device: Union[torch.device, str] = "cpu",
    quiet: bool = True,
) -> None:

    device = torch.device(device)

    state_dict = torch.load(model_pt, map_location="cpu")
    target_dim, freq_dim = state_dict.pop("target_dim"), state_dict.pop("freq_dim")
    model = models.AcousticModel(freq_dim, target_dim, model_params)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ds = data.SpectDataSet(
        data_dir, params=data_params, suppress_alis=True, suppress_uttids=False,
    )
    if not quiet:
        try:
            name = model_pt.name
        except:
            name = model_pt
        print(
            f"Computing logits for '{data_dir}' with '{name}' and storing in "
            f"'{logit_dir}'...",
            file=sys.stderr,
        )
        ds = tqdm(ds)

    os.makedirs(logit_dir, exist_ok=True)

    for feats, _, utt_id in ds:
        T = feats.size(0)
        feats = feats.to(device).unsqueeze(0)
        lens = torch.tensor([T]).to(device)
        logits, lens = model(feats, lens)
        torch.save(logits.cpu(), f"{logit_dir}/{utt_id}.pt")
        del logits, lens, feats

    if not quiet:
        print(
            f"Computed logits for '{data_dir}' with '{name}' and stored in "
            f"'{logit_dir}'",
            file=sys.stderr,
        )


class DirectoryDataset(torch.utils.data.Dataset):

    data_dir: str
    suffix: str
    utt_ids: Sequence[str]

    def __init__(self, data_dir: str, suffix: str = ".pt") -> None:
        super().__init__()
        if not os.path.isdir(data_dir):
            raise ValueError(f"'{data_dir}' is not a directory")
        self.data_dir, self.suffix = data_dir, suffix
        self.utt_ids = sorted(
            x[: len(x) - len(suffix)]
            for x in os.listdir(data_dir)
            if x.endswith(suffix)
        )

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        utt_id = self.utt_ids[idx]
        return utt_id, torch.load(os.path.join(self.data_dir, utt_id + self.suffix))


@torch.no_grad()
def decode_am(
    logit_dir: str,
    decode_dir: str,
    device: Union[torch.device, str] = "cpu",
    beam_width: int = 1,
    lm_beta: float = 0.0,
    lm_pt: Optional[str] = None,
    quiet: bool = True,
) -> None:

    device = torch.device(device)

    if lm_beta > 0:
        if lm_pt is None:
            raise ValueError("--lm-pt not specified")
        state_dict = torch.load(lm_pt)
        vocab_size = state_dict.pop("vocab_size")
        sos = state_dict.pop("sos")
        lm = layers.LookupLanguageModel(vocab_size, sos)
        lm.load_state_dict(state_dict)
        lm.to(device)
    else:
        lm = None

    search = layers.CTCPrefixSearch(beam_width, lm_beta, lm)

    ds = DirectoryDataset(logit_dir)
    if not quiet:
        print(
            f"Decoding from '{logit_dir}' with beam width {beam_width} and lm beta"
            f"{lm_beta} and storing in '{decode_dir}'...",
            file=sys.stderr,
        )
        ds = tqdm(ds)

    os.makedirs(decode_dir, exist_ok=True)

    for utt_id, logits in ds:
        logits = logits.to(device).log_softmax(-1)
        hyp, lens, _ = search(logits)
        hyp, lens = hyp[..., 0], lens[..., 0]
        hyp = hyp.flatten()[: lens.flatten()].cpu()
        torch.save(hyp, f"{decode_dir}/{utt_id}.pt")

    if not quiet:
        print(f"Decoded into '{decode_dir}'", file=sys.stderr)

