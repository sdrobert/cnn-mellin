# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import os

# import torch
# import pytest
# import pydrobert.torch.data as data
# import cnn_mellin.running as running

# __author__ = "Sean Robertson"
# __email__ = "sdrobert@cs.toronto.edu"
# __license__ = "Apache 2.0"
# __copyright__ = "Copyright 2018 Sean Robertson"


# class DummyAM(torch.nn.Module):
#     def __init__(self, num_filts, num_classes, seed=1):
#         super(DummyAM, self).__init__()
#         self.seed = seed
#         torch.manual_seed(seed)
#         self.fc = torch.nn.Linear(num_filts, num_classes)
#         self.drop = torch.nn.Dropout(p=0.5)

#     def forward(self, x):
#         x = self.fc(x)
#         return x.sum(1)  # sum out the context window

#     def reset_parameters(self):
#         torch.manual_seed(self.seed)
#         self.fc.reset_parameters()

#     @property
#     def dropout(self):
#         return self.drop.p

#     @dropout.setter
#     def dropout(self, p):
#         self.drop.p = p


# def test_get_am_alignment_cross_entropy(temp_dir, device, populate_torch_dir):
#     populate_torch_dir(temp_dir, 50)
#     params = data.ContextWindowDataSetParams(
#         context_left=1,
#         context_right=1,
#         batch_size=5,
#         seed=2,
#         drop_last=True,
#     )
#     data_loader = data.ContextWindowEvaluationDataLoader(temp_dir, params)
#     model = DummyAM(5, 11)
#     loss_a = running.get_am_alignment_cross_entropy(
#         model, data_loader, device=device)
#     assert loss_a != 0.  # highly unlikely that it would be zero
#     loss_b = running.get_am_alignment_cross_entropy(
#         model, data_loader, device=device)
#     assert abs(loss_a - loss_b) < 1e-5


# def test_write_am_pdfs(temp_dir, device, populate_torch_dir):
#     populate_torch_dir(temp_dir, 50, include_ali=False)
#     params = data.ContextWindowDataSetParams(
#         context_left=2,
#         context_right=2,
#         batch_size=6,
#         seed=3,
#         drop_last=False,
#     )
#     data_loader = data.ContextWindowEvaluationDataLoader(temp_dir, params)
#     log_prior = torch.rand(11)
#     log_prior /= log_prior.sum()
#     model = DummyAM(5, 11)
#     running.write_am_pdfs(model, data_loader, log_prior, device=device)
#     file_list = os.listdir(os.path.join(temp_dir, 'pdfs'))
#     assert len(file_list) == 50
#     pdfs_a = dict()
#     for file_name in file_list:
#         feat = torch.load(os.path.join(temp_dir, 'feats', file_name))
#         pdf = torch.load(os.path.join(temp_dir, 'pdfs', file_name))
#         assert tuple(pdf.size()) == (feat.size()[0], 11)
#         assert torch.allclose(
#             torch.exp(pdf + log_prior).sum(1), torch.tensor(1.))
#         pdfs_a[file_name] = pdf
#     running.write_am_pdfs(
#         model, data_loader, log_prior,
#         device=device, pdfs_dir=os.path.join(temp_dir, 'zoo'))
#     for file_name in file_list:
#         pdf = torch.load(os.path.join(temp_dir, 'zoo', file_name))
#         assert torch.allclose(pdf, pdfs_a[file_name])


# def test_train_am_for_epoch(temp_dir, device, populate_torch_dir):
#     populate_torch_dir(temp_dir, 50)
#     spect_p = data.ContextWindowDataSetParams(
#         context_left=1,
#         context_right=1,
#         batch_size=5,
#         seed=2,
#         drop_last=True,
#     )
#     data_loader = data.ContextWindowTrainingDataLoader(
#         temp_dir, spect_p, num_workers=4)
#     train_p = running.TrainingEpochParams(
#         num_epochs=10,
#         seed=3,
#         dropout_prob=.5,
#     )
#     model = DummyAM(5, 11)
#     # important! Use optimizer without history
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#     loss_a = running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device=device)
#     assert loss_a != 0
#     loss_b = running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device=device)
#     assert loss_a > loss_b  # we learned something, maybe?
#     optimizer.zero_grad()
#     # important! We have to initialize parameters on the same device to get the
#     # same results!
#     model.cpu().reset_parameters()
#     loss_c = running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, epoch=0, device=device)
#     assert abs(loss_a - loss_c) < 1e-5


# @pytest.mark.gpu
# def test_train_am_for_epoch_changing_devices(temp_dir, populate_torch_dir):
#     populate_torch_dir(temp_dir, 50)
#     spect_p = data.ContextWindowDataSetParams(
#         context_left=1,
#         context_right=1,
#         batch_size=5,
#         seed=2,
#         drop_last=True,
#     )
#     data_loader = data.ContextWindowTrainingDataLoader(
#         temp_dir, spect_p, num_workers=4)
#     model = DummyAM(5, 11)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#     train_p = running.TrainingEpochParams(seed=3)
#     running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device='cuda')
#     running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device='cpu')
#     running.train_am_for_epoch(
#         model, data_loader, optimizer, train_p, device='cuda')
