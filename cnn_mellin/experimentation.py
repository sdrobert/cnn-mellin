'''Functions and classes involved in experimentation'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

from csv import DictReader, writer

import torch

from cnn_mellin.util import optimizer_to

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2018 Sean Robertson"


class TrainingStateController(object):
    '''Controls the state of training a model

    This class is used to help both control and persist experiment information
    like the current epoch, the model parameters, and model error. It assumes
    that the values stored in `params` have not changed when resuming a run.

    Parameters
    ----------
    params : TrainingParams
    state_csv_path : str, optional
        A path to where training state information is stored. It stores in
        comma-separated-values format the following information. Note that
        stored values represent the state *after* updates due to epoch
        results, such as the learning rate. That way, an experiment can be
        resumed without worrying about updating the loaded results
        1. "epoch": the epoch associated with this row of information
        2. "es_resume_cd": the number of epochs left before the early
           stopping criterion begins/resumes
        3. "es_patience_cd": the number of epochs left that must pass
           without much improvement before training halts due to early stopping
        3. "rlr_resume_cd": the number of epochs left before the
           criterion for reducing the learning rate begins/resumes
        4. "rlr_patience_cd": the number of epochs left that must pass
           without much improvement before the learning rate is reduced
        5. "lr": the learning rate of the optimizer after any updates
        6. "train_ent": mean training cross-entropy in exponent format
        7. "val_ent": mean validation cross-entropy in exponent format
        If unset, the history will not be stored/loaded
    state_dir : str, optional
        A path to a directory to store/load model and optimizer states. If
        unset, the information will not be stored/loaded

    Attributes
    ----------
    params : TrainingParams
    state_csv_path : str or None
    state_dir : str or None
    cache_hist : dict
        A dictionary of cached results per epoch. Is not guaranteed to be
        up-to-date with `state_csv_path`
    '''

    def __init__(self, params, state_csv_path=None, state_dir=None):
        super(TrainingStateController, self).__init__()
        self.params = params
        self.state_csv_path = state_csv_path
        self.state_dir = state_dir
        self.cache_hist = dict()

    def update_cache(self):
        '''Update the cache with history stored in state_csv_path'''
        if 0 not in self.cache_hist:
            # add a dummy entry for epoch "0" just to make logic easier. We
            # won't save it
            self.cache_hist[0] = {
                'epoch': 0,
                'es_resume_cd': self.params.early_stopping_burnin,
                'es_patience_cd': self.params.early_stopping_patience,
                'rlr_resume_cd': self.params.reduce_lr_burnin,
                'rlr_patience_cd': self.params.reduce_lr_patience,
                'train_ent': float('inf'),
                'val_ent': float('inf'),
                'lr': None,
            }
            if self.params.log10_learning_rate is not None:
                self.cache_hist[0]['lr'] = (
                    10 ** self.params.log10_learning_rate)
        if (self.state_csv_path is None or
                not os.path.exists(self.state_csv_path)):
            return
        with open(self.state_csv_path) as f:
            reader = DictReader(f)
            for row in reader:
                self.cache_hist[int(row['epoch'])] = {
                    'epoch': int(row['epoch']),
                    'es_resume_cd': int(row['es_resume_cd']),
                    'es_patience_cd': int(row['es_patience_cd']),
                    'rlr_resume_cd': int(row['rlr_resume_cd']),
                    'rlr_patience_cd': int(row['rlr_patience_cd']),
                    'lr': float(row['lr']),
                    'train_ent': float(row['train_ent']),
                    'val_ent': float(row['val_ent']),
                }

    def get_last_epoch(self):
        '''int : last finished epoch from training, or 0 if no history'''
        self.update_cache()
        return max(self.cache_hist)

    def get_best_epoch(self, train_ent=False):
        '''Get the epoch that has lead to the best validation entropy so far

        The "best" is the lowest recorded.

        Parameters
        ----------
        train_ent : bool, optional
            If ``True`` look for the best training entropy instead

        Returns
        -------
        epoch : int
            The corresponding 'best' epoch, or 0 if no epochs have run
        '''
        ent = 'train_ent' if train_ent else 'val_ent'
        self.update_cache()
        min_epoch = 0
        min_ent = self.cache_hist[0][ent]
        for info in self.cache_hist.values():
            if min_ent > info[ent]:
                min_epoch = info['epoch']
                min_ent = info[ent]
        return min_epoch

    def load_model_and_optimizer_for_epoch(self, model, optimizer, epoch=0):
        '''Load up model and optimizer states, or initialize them

        If `epoch` is not specified or 0, the model and optimizer are
        initialized with states for the beginning of the experiment. Otherwise,
        we look for appropriately named files in ``self.state_dir``
        '''
        model_device = next(model.parameters()).device
        if not epoch:
            # reset on cpu. Different devices can randomize differently
            model.cpu().reset_parameters()
            optim_defaults = dict(optimizer.defaults)
            if self.params.log10_learning_rate is not None:
                optim_defaults['lr'] = 10 ** self.params.log10_learning_rate
            else:
                del optim_defaults['lr']
            if self.params.seed is not None:
                torch.manual_seed(self.params.seed)
            new_optimizer = type(optimizer)(
                model.parameters(),
                **optim_defaults
            )
            model.to(model_device)
            optimizer_to(optimizer, model_device)
            optimizer.load_state_dict(new_optimizer.state_dict())
        elif self.state_dir is not None:
            epoch_info = self[epoch]
            model_basename = self.params.saved_model_fmt.format(**epoch_info)
            optimizer_basename = self.params.saved_optimizer_fmt.format(
                **epoch_info)
            model_state_dict = torch.load(
                os.path.join(self.state_dir, model_basename),
                map_location=model_device
            )
            model.load_state_dict(model_state_dict)
            optimizer_state_dict = torch.load(
                os.path.join(self.state_dir, optimizer_basename),
                map_location=model_device
            )
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            print(
                'Unable to load optimizer for epoch {}. No state dict!'
                ''.format(epoch))

    def delete_model_and_optimizer_for_epoch(self, epoch):
        '''Delete state dicts for model and epoch off of disk, if they exist

        This method does nothing if the epoch records or the files do not
        exist. It is called during ``update_for_epoch`` if the parameter
        ``keep_last_and_best_only`` is ``True``

        Parameters
        ----------
        epoch : int
            The epoch in question
        '''
        if self.state_dir is None:
            return
        epoch_info = self.get_info(epoch, None)
        if epoch_info is None:
            return
        model_basename = self.params.saved_model_fmt.format(**epoch_info)
        optimizer_basename = self.params.saved_optimizer_fmt.format(
            **epoch_info)
        try:
            os.remove(os.path.join(self.state_dir, model_basename))
        except OSError as e:
            pass
        try:
            os.remove(os.path.join(self.state_dir, optimizer_basename))
        except OSError:
            pass

    def get_info(self, epoch, *default):
        '''Get history entries for a specific epoch

        If there's an entry present for `epoch`, return it. The value is a
        dictionary with the keys "epoch", "es_resume_cd", "es_patience_cd",
        "rlr_resume_cd", "rlr_patience_cd", "lr", "train_ent", and "val_ent".

        If there's no entry for `epoch`, and no additional arguments were
        passed to this method, it raises a ``KeyError``. If an additional
        argument was passed to this method, return it.
        '''
        if len(default) > 1:
            raise TypeError('expected at most 2 arguments, got 3')
        if epoch in self.cache_hist:
            return self.cache_hist[epoch]
        self.update_cache()
        return self.cache_hist.get(epoch, *default)

    def __getitem__(self, epoch):
        return self.get_info(epoch)

    def save_model_and_optimizer_with_info(self, model, optimizer, info):
        '''Save model and optimizer state dictionaries to file given epoch info

        This is called automatically during ``update_for_epoch``. Does not save
        if there is no directory to save to (i.e. ``self.state_dir is None``).
        Format strings from ``self.params`` are formatted with the values from
        `info` to construct the base names of each file

        Parameters
        ----------
        model : AcousticModel
        optimizer : torch.optim.Optimizer
        info : dict
            A dictionary with the entries "epoch", "es_resume_cd",
            "es_patience_cd", "rlr_resume_cd", "rlr_patience_cd", "lr",
            "train_ent", and "val_ent"
        '''
        if self.state_dir is None:
            return
        os.makedirs(self.state_dir, exist_ok=True)
        model_basename = self.params.saved_model_fmt.format(**info)
        optimizer_basename = self.params.saved_optimizer_fmt.format(
            **info)
        model_state_dict = model.state_dict()
        # we always save on the cpu
        for key, val in model_state_dict.items():
            model_state_dict[key] = val.cpu()
        optimizer_to(optimizer, 'cpu')
        torch.save(
            model.state_dict(),
            os.path.join(self.state_dir, model_basename),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(self.state_dir, optimizer_basename),
        )

    def save_info_to_hist(self, info):
        '''Append history entries to the history csv

        This is called automatically during ``update_for_epoch``. Does not save
        if there is no file to save to (i.e. ``self.state_csv_path is None``).
        Values are appended to the end of the csv file - no checking is
        performed for mantaining a valid history.

        Parameters
        ----------
        info : dict
            A dictionary with the entries "epoch", "es_resume_cd",
            "es_patience_cd", "rlr_resume_cd", "rlr_patience_cd", "lr",
            "train_ent", and "val_ent"
        '''
        self.cache_hist[info['epoch']] = info
        if self.state_csv_path is None:
            return
        if not self.params.num_epochs:
            epoch_fmt_str = '{:010d}'
        else:
            epoch_fmt_str = '{{:0{}d}}'.format(
                int(math.log10(self.params.num_epochs)) + 1)
        es_resume_cd_fmt_str = '{{:0{}d}}'.format(
            int(math.log10(max(
                self.params.early_stopping_burnin,
                1,
                ))) + 1
        )
        es_patience_cd_fmt_str = '{{:0{}d}}'.format(
            int(math.log10(max(
                self.params.early_stopping_patience,
                1,
                ))) + 1
        )
        rlr_resume_cd_fmt_str = '{{:0{}d}}'.format(
            int(math.log10(max(
                self.params.reduce_lr_cooldown,
                self.params.reduce_lr_burnin,
                1,
            ))) + 1
        )
        rlr_patience_cd_fmt_str = '{{:0{}d}}'.format(
            int(math.log10(max(
                self.params.reduce_lr_patience,
                1,
            ))) + 1
        )
        lr_fmt_str = train_ent_fmt_str = val_ent_fmt_str = '{:10e}'
        write_header = not os.path.exists(self.state_csv_path)
        with open(self.state_csv_path, 'a') as f:
            wr = writer(f)
            if write_header:
                wr.writerow([
                    'epoch',
                    'es_resume_cd',
                    'es_patience_cd',
                    'rlr_resume_cd',
                    'rlr_patience_cd',
                    'lr',
                    'train_ent',
                    'val_ent',
                ])
            wr.writerow([
                epoch_fmt_str.format(info['epoch']),
                es_resume_cd_fmt_str.format(info['es_resume_cd']),
                es_patience_cd_fmt_str.format(info['es_patience_cd']),
                rlr_resume_cd_fmt_str.format(info['rlr_resume_cd']),
                rlr_patience_cd_fmt_str.format(info['rlr_patience_cd']),
                lr_fmt_str.format(info['lr']),
                train_ent_fmt_str.format(info['train_ent']),
                val_ent_fmt_str.format(info['val_ent']),
            ])

    def update_for_epoch(
            self, model, optimizer, train_ent, val_ent, epoch=None):
        '''Update history and optimizer after latest epoch results

        Parameters
        ----------
        model : AcousticModel
        optimizer : torch.optim.Optimizer
        train_ent : float
            Mean training cross entropy for the epoch
        val_ent : float
            Mean validation cross entropy for the epoch
        epoch : int, optional
            The epoch that just finished. If unset, it is inferred to be one
            after the last epoch in the history

        Returns
        -------
        continue_training : bool
            Whether to continue training. This can be set to ``False`` either
            by hitting the max number of epochs or by early stopping
        '''
        if epoch is None:
            epoch = self.get_last_epoch() + 1
        if not self.params.num_epochs:
            continue_training = True
        else:
            continue_training = epoch < self.params.num_epochs
        info = dict(self.get_info(epoch - 1, None))
        if info is None:
            raise ValueError(
                'no entry for the previous epoch {}, so unable to update'
                ''.format(epoch))
        last_best = self.get_best_epoch()
        best_info = self[last_best]
        if info['lr'] is None:
            # can only happen during the first epoch. We don't know the
            # optimizer defaults, so we get them now
            info['lr'] = optimizer.defaults['lr']
        if info["es_resume_cd"]:
            info["es_resume_cd"] -= 1
        elif (max(best_info['val_ent'] - val_ent, 0) <
                self.params.early_stopping_threshold):
            info["es_patience_cd"] -= 1
            if not info["es_patience_cd"]:
                continue_training = False
        else:
            info["es_patience_cd"] = self.params.early_stopping_patience
        if info["rlr_resume_cd"]:
            info["rlr_resume_cd"] -= 1
        elif (max(best_info['val_ent'] - val_ent, 0) <
                self.params.reduce_lr_threshold):
            info["rlr_patience_cd"] -= 1
            if not info["rlr_patience_cd"]:
                old_lr = info['lr']
                new_lr = old_lr * self.params.reduce_lr_factor
                rlr_epsilon = 10 ** self.params.reduce_lr_log10_epsilon
                if old_lr - new_lr > rlr_epsilon:
                    info['lr'] = new_lr
                    for param_group in optimizer.param_groups:
                        # just assume that the user knows what's what if
                        # the optimizer's lr doesn't match the old one
                        param_group['lr'] = new_lr
                info["rlr_resume_cd"] = self.params.reduce_lr_cooldown
                info["rlr_patience_cd"] = self.params.reduce_lr_patience
        else:
            info["rlr_patience_cd"] = self.params.reduce_lr_patience
        info["epoch"] = epoch
        info["val_ent"] = val_ent
        info["train_ent"] = train_ent
        self.save_model_and_optimizer_with_info(model, optimizer, info)
        self.save_info_to_hist(info)
        cur_best = self.get_best_epoch()
        if self.params.keep_last_and_best_only and cur_best != epoch - 1:
            self.delete_model_and_optimizer_for_epoch(epoch - 1)
        if self.params.keep_last_and_best_only and cur_best != last_best:
            self.delete_model_and_optimizer_for_epoch(last_best)
        return continue_training
