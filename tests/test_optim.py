# __author__ = "Sean Robertson"
# __email__ = "sdrobert@cs.toronto.edu"
# __license__ = "Apache 2.0"
# __copyright__ = "Copyright 2019 Sean Robertson"

# import os

# import torch
# import pytest
# import cnn_mellin.optim as optim
# import cnn_mellin.models as models
# import cnn_mellin.running as running
# import pydrobert.torch.data as data
# import optuna


# @pytest.mark.parametrize('partition_style', ['round-robin', 'average', 'last'])
# @pytest.mark.parametrize('val_partition', [True, False])
# @pytest.mark.parametrize('partitions', [2, 3])
# @pytest.mark.parametrize('intermediate_hist_size', [4, 5, 6])
# def test_optimize_am(
#         temp_dir, populate_torch_dir, partition_style, val_partition,
#         partitions, device, intermediate_hist_size):
#     data_dir = os.path.join(temp_dir, 'data')
#     history_db = os.path.join(temp_dir, 'hist.db')
#     populate_torch_dir(data_dir, 100)
#     partitions += 1 if val_partition else 0
#     optim_params = optim.CNNMellinOptimParams(
#         partition_style=partition_style,
#         max_samples=10,
#         to_optimize=['weight_decay'],
#         seed=2,
#     )
#     base_model_params = models.AcousticModelParams(
#         freq_dim=5,
#         target_dim=11,
#         kernel_sizes=[],
#         hidden_sizes=[],
#     )
#     base_training_params = running.TrainingParams(
#         num_epochs=2,
#     )
#     base_data_params = data.ContextWindowDataSetParams(
#         context_left=0,
#         context_right=0,
#         batch_size=10,
#     )
#     weight = torch.rand(11)
#     _, training_params, _ = optim.optimize_am(
#         data_dir, partitions, optim_params, base_model_params,
#         base_training_params, base_data_params,
#         val_partition=val_partition,
#         weight=weight,
#         device=device,
#         verbose=True,
#     )
#     first_decay = training_params.weight_decay
#     optim_params.max_samples = intermediate_hist_size
#     optim.optimize_am(
#         data_dir, partitions, optim_params, base_model_params,
#         base_training_params, base_data_params,
#         val_partition=val_partition,
#         weight=weight,
#         device=device,
#         history_url='sqlite:///' + history_db,
#     )
#     optim_params.max_samples = 10
#     _, training_params, _ = optim.optimize_am(
#         data_dir, partitions, optim_params, base_model_params,
#         base_training_params, base_data_params,
#         val_partition=val_partition,
#         weight=weight,
#         device=device,
#         history_url='sqlite:///' + history_db,
#     )
#     assert os.path.exists(history_db)
#     study = optuna.create_study(
#         storage='sqlite:///' + history_db,
#         study_name=optim_params.study_name,
#         load_if_exists=True,
#     )
#     assert (
#         abs(
#             training_params.weight_decay -
#             study.best_params['weight_decay']) <= 1e-5)
