"""Command line hooks for cnn_mellin"""

from logging import warn
import os
import argparse
import warnings
import sys
import re

from typing import Any, Optional, Sequence, Text, Union

import torch
import cnn_mellin.models as models
import cnn_mellin.running as running
import pydrobert.torch.data as data
import pydrobert.param.argparse as pargparse
import pydrobert.param.optuna as poptuna

from cnn_mellin import construct_default_param_dict

try:
    import cnn_mellin.optim as optim
except ImportError:
    optim = None


def readable_dir(str_) -> str:
    if not os.path.isdir(str_):
        raise argparse.ArgumentTypeError(f"'{str_}' is not a directory")
    return os.path.abspath(str_)


def writable_dir(str_) -> str:
    if not os.path.isdir(str_):
        try:
            os.makedirs(str_)
        except OSError:
            raise argparse.ArgumentTypeError(f"Cannot create directory at '{str_}'")
    return os.path.abspath(str_)


def get_bounded_number_type(
    type_: type, bounds=(-float("inf"), float("inf")), exclusive=True
):
    low, high = bounds

    def _to_val(str_):
        try:
            val = type_(str_)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Cannot cast '{str_}' to {type_}")
        if exclusive:
            if low >= val or val >= high:
                raise argparse.ArgumentTypeError(
                    f"'{str_}' must be between ({low}, {high})"
                )
        else:
            if low > val or val > high:
                raise argparse.ArgumentTypeError(
                    f"'{str_}' must be between [{low}, {high}]"
                )
        return val

    return _to_val


def hhmmss_type(str_, allow_zero=False) -> str:
    err = argparse.ArgumentTypeError(
        f"Expected string of format '[[HH:]MM:]SS.xxx', got {str_}"
    )
    vals = str_.split(":")
    if len(vals) > 3:
        raise err
    try:
        vals = [float(x) for x in vals]
    except TypeError:
        raise err
    val = vals.pop()
    if val < 0.0 or vals and val >= 60.0:
        raise err
    factor = 60.0
    while vals:
        new = vals.pop()
        if new < 0.0 or new % 1 or (vals and new >= 60.0):
            raise err
        val += new * factor
        factor *= 60
    if not allow_zero and val == 0.0:
        raise argparse.ArgumentTypeError("Cannot be 0")
    return val


class RegexChoices(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Union[Text, Sequence[Any], None],
        option_string: Optional[Text],
    ) -> None:
        all_ = poptuna.get_param_dict_tunable(construct_default_param_dict())
        kept = set()
        for value in values:
            try:
                pattern = re.compile(value)
            except SyntaxError:
                raise argparse.ArgumentTypeError(
                    f"'{value}' could not be compiled into regex"
                )
            for choice in all_:
                if pattern.fullmatch(choice):
                    kept.add(choice)
        if option_string == "--whitelist" and not kept:
            raise argparse.ArgumentTypeError(
                f"Regular expressions '{values}' matched no options"
            )
        setattr(namespace, self.dest, kept)


def parse_args(args: Optional[Sequence[str]], param_dict: dict):
    parser = argparse.ArgumentParser(
        description="Training, decoding, or hyperparameter optimization"
    )
    parser.add_argument(
        "--model-dir",
        type=writable_dir,
        default=None,
        help="Where to save/load models and training history to/from",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default="cpu",
        help="Which device to perform operations on",
    )
    pargparse.add_parameterized_print_group(parser, parameterized=param_dict)
    pargparse.add_parameterized_read_group(parser, parameterized=param_dict)

    subparsers = parser.add_subparsers(required=True, dest="command")
    train_parser = subparsers.add_parser("train", help="Train an acoustic model")
    train_parser.add_argument(
        "train_dir", type=readable_dir, help="Training data directory"
    )
    train_parser.add_argument(
        "dev_dir",
        nargs="?",
        type=readable_dir,
        help="Validation directory. If unset, uses training directory",
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress output to stderr",
    )
    train_parser.add_argument(
        "--num-data-workers",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of background workers for training data loader. 0 is serial. "
        "Defaults to one fewer than the number of virtual cores on the machine.",
    )

    if optim is not None:
        all_ = poptuna.get_param_dict_tunable(param_dict)
        optim_parser = subparsers.add_parser(
            "optim", help="Hyperparameter optimization of an acoustic model"
        )
        optim_parser.add_argument(
            "db_url",
            type=optim.sqlalchemy.engine.url.make_url,
            help="RFC-1738 URL pointing to database",
        )
        optim_parser.add_argument(
            "--study-name",
            default=None,
            help="Under what name to store the hyperparameter optimization results in the "
            "database. Defaults to the basename of the database url",
        )
        optim_subparsers = optim_parser.add_subparsers(
            required=True, dest="optim_command"
        )

        optim_init_subparser = optim_subparsers.add_parser(
            "init", help="Initialize the Optuna database to begin optimization"
        )
        optim_init_subparser.add_argument(
            "train_dir", type=readable_dir, help="Where training data are located"
        )
        optim_init_subparser.add_argument(
            "--dev-proportion",
            metavar="(0, 1)",
            type=get_bounded_number_type(float, (0.0, 1.0)),
            default=0.1,
            help="The proportion of the training directory devoted to the dev set",
        )
        optim_init_subparser.add_argument(
            "--mem-limit-bytes",
            metavar="N",
            type=get_bounded_number_type(int, (1, float("inf"))),
            default=10 * (1024 ** 3),
            help="If optimizing model parameters, the number of bytes to limit the "
            "forward-backward pass of the largest batch to on CPU. Models above this "
            "number will be pruned.",
        )

        blacklist_whitelist_group = optim_init_subparser.add_mutually_exclusive_group()
        blacklist_whitelist_group.add_argument(
            "--blacklist",
            metavar="PATTERN",
            nargs="+",
            action=RegexChoices,
            default=set(),
            help="Regexes of hyperparameters to exclude from search",
        )
        blacklist_whitelist_group.add_argument(
            "--whitelist",
            metavar="PATTERN",
            nargs="+",
            action=RegexChoices,
            default=all_,
            help="Regexes of hyperparameters to include in search",
        )

        optim_run_subparser = optim_subparsers.add_parser(
            "run", help="Run a Study from a previously initialized optuna database"
        )
        optim_run_subparser.add_argument(
            "--sampler",
            default="tpe",
            choices=["tpe", "random", "cmaes", "nsgaii"],
            help="Which sampler to use in hyperparameter optimization. See "
            "https://optuna.readthedocs.io/en/stable/reference/samplers.html for more "
            "info",
        )

        end_group = optim_run_subparser.add_mutually_exclusive_group()
        end_group.add_argument(
            "--num-trials",
            type=get_bounded_number_type(int, (0, float("inf"))),
            default=None,
            help="Number of trials to run before quitting. This is the total number "
            "to run in *this* process - it ignores any previous or simultaneous trials",
        )
        end_group.add_argument(
            "--timeout",
            type=hhmmss_type,
            default=None,
            metavar="[[HH:]MM:]:SS.xxx",
            help="Length of time to run the study before quitting. This is the total "
            "length of *this* process - it ignores any previous or simultaneous trials",
        )

    return parser.parse_args(args)


def train(options, param_dict):
    if options.model_dir is None:
        warnings.warn("--model-dir was not set! Will not save anything!")
    model, _ = running.train_am(
        param_dict["model"],
        param_dict["training"],
        param_dict["data"],
        options.train_dir,
        options.train_dir if options.dev_dir is None else options.dev_dir,
        options.model_dir,
        options.device,
        options.num_data_workers,
        tuple(),
        options.quiet,
    )
    if options.model_dir is not None:
        model_pt = os.path.join(options.model_dir, "model.pt")
        if not options.quiet:
            print(f"Saving final model to '{model_pt}'", file=sys.stderr)
        # add the number of classes and number of filters to the state dict so that
        # we don't have to keep track of the
        state_dict = model.state_dict()
        assert "target_dim" not in state_dict
        assert "freq_dim" not in state_dict
        state_dict["target_dim"] = model.target_dim
        state_dict["freq_dim"] = model.freq_dim
        torch.save(state_dict, model_pt)


def optim_init(options, param_dict):
    optim.init_study(
        options.train_dir,
        param_dict,
        options.db_url,
        options.whitelist - options.blacklist,
        options.study_name,
        options.device,
        options.dev_proportion,
        options.mem_limit_bytes,
    )


def optim_run(options, param_dict):
    study_name = options.study_name
    if study_name is None:
        study_name = os.path.basename(options.db_url.database).split(".")[0]
    if options.sampler == "tpe":
        sampler = optim.optuna.samplers.TPESampler()
    elif options.sampler == "random":
        sampler = optim.optuna.samplers.RandomSampler()
    elif options.sampler == "cmaes":
        sampler = optim.optuna.samplers.CmaEsSampler()
    elif options.sampler == "nsgaii":
        sampler = optim.optuna.samplers.NSGAIISampler()
    else:
        assert False
    study = optim.optuna.load_study(study_name, str(options.db_url), sampler)
    if study.user_attrs["device"] != str(options.device):
        warnings.warn(
            f"Device passed by command line ({options.device}) differs from the device "
            f"the study was initialized with ({study.user_attrs['device']}). Will use "
            "the latter."
        )
    study.optimize(optim.objective, options.num_trials, options.timeout)


def cnn_mellin(args: Optional[Sequence[str]] = None):
    param_dict = construct_default_param_dict()
    options = parse_args(args, param_dict)

    if options.command == "train":
        train(options, param_dict)
    elif options.command == "optim":
        if options.optim_command == "init":
            optim_init(options, param_dict)
        elif options.optim_command == "run":
            optim_run(options, param_dict)
