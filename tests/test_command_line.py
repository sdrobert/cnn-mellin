# import os

# from collections import namedtuple

# import pytest
# import torch

# import cnn_mellin.command_line as command_line


# def test_read_and_write_print_parameters_as_ini(capsys, temp_dir):
#     assert not command_line.print_parameters_as_ini([])
#     s = capsys.readouterr()
#     assert s.out.find("[data]") != -1
#     a_ini = s.out[s.out.find("[model]") : s.out.find("[training]")]
#     # this doesn't replace the original value, just appends to it
#     a_ini = a_ini.replace("kernel_sizes = ", "kernel_sizes = (1000, 1000, 1), ")
#     a_ini = a_ini.replace("time_factor = ", "time_factor = 1000")
#     with open(os.path.join(temp_dir, "a.ini"), "w") as f:
#         f.write(a_ini + "\n")
#     b_ini = "[model]\ntime_factor = 30\n"
#     with open(os.path.join(temp_dir, "b.ini"), "w") as f:
#         f.write(b_ini)
#     command_line.print_parameters_as_ini(
#         [os.path.join(temp_dir, "a.ini"), os.path.join(temp_dir, "b.ini")]
#     )
#     s = capsys.readouterr()
#     assert s.out.find("[data]") != -1
#     assert s.out.find("kernel_sizes = (1000, 1000, 1), ") != -1
#     assert s.out.find("time_factor = 1000") == -1
#     assert s.out.find("time_factor = 30") != -1


# def test_target_count_info_to_tensor(temp_dir):
#     info_file = os.path.join(temp_dir, "info.ark")
#     with open(info_file, "w") as f:
#         f.write(
#             "num_utterances 10\n"
#             "num_filts 4\n"
#             "total_frames 100\n"
#             "count_01 4\n"
#             "count_02 0\n"
#             "count_03 50\n"
#             "count_04 20\n"
#             "count_05 26\n"
#         )
#     weight_file = os.path.join(temp_dir, "weight.pt")
#     assert not command_line.target_count_info_to_tensor(
#         ["--min-count", "0", info_file, "inv_weight", weight_file]
#     )
#     exp_weight_tensor = torch.FloatTensor([0, 4, 0, 50, 20, 26])
#     exp_weight_tensor = (100.0 - exp_weight_tensor) / 100.0
#     act_weight_tensor = torch.load(weight_file)
#     assert torch.allclose(exp_weight_tensor, act_weight_tensor)
#     prior_file = os.path.join(temp_dir, "prior.pt")
#     assert not command_line.target_count_info_to_tensor(
#         ["--num-targets", "7", info_file, "log_prior", prior_file]
#     )
#     exp_prior_tensor = torch.FloatTensor([1, 4, 1, 50, 20, 26, 1])
#     exp_prior_tensor /= exp_prior_tensor.sum()
#     exp_prior_tensor = torch.log(exp_prior_tensor)
#     act_prior_tensor = torch.load(prior_file)
#     assert torch.allclose(exp_prior_tensor, act_prior_tensor)


# @pytest.fixture
# def DummyAM():
#     class _DummyAM(torch.nn.Module):
#         def __init__(self, params, *args):
#             super(_DummyAM, self).__init__()
#             self.params = params
#             self.fc = torch.nn.Linear(params.freq_dim, params.target_dim)
#             self.reset_parameters()

#         def reset_parameters(self):
#             if self.params.seed is not None:
#                 torch.manual_seed(self.params.seed)
#             self.fc.reset_parameters()

#         def forward(self, x):
#             x = x.sum(1)
#             return self.fc(x)

#     import cnn_mellin.models as models

#     old = models.AcousticModel
#     models.AcousticModel = _DummyAM
#     yield _DummyAM
#     models.AcousticModel = old


# @pytest.fixture
# def DummyTrainingController():
#     class _DummyTrainingController(object):
#         def __init__(self, params, state_csv_path=None, state_dir=None):
#             self.params = params
#             self.state_csv_path = state_csv_path
#             self.state_dir = state_dir

#         def save(self, epoch, state_dict):
#             pth = os.path.join(
#                 self.state_dir, self.params.saved_model_fmt.format(epoch=epoch)
#             )
#             if not os.path.isdir(self.state_dir):
#                 os.makedirs(self.state_dir)
#             torch.save(state_dict, pth)

#         def get_best_epoch(self):
#             return 1

#         def get_last_epoch(self):
#             return 2

#         def get_info(self, epoch, *default):
#             return {"epoch": epoch}

#     import pydrobert.torch.training as training

#     old = training.TrainingStateController
#     training.TrainingStateController = _DummyTrainingController
#     yield _DummyTrainingController
#     training.TrainingStateController = old


# @pytest.mark.gpu
# def test_forward_acoustic_model(
#     temp_dir, populate_torch_dir, DummyAM, DummyTrainingController
# ):
#     torch.manual_seed(10)
#     prior = torch.rand(11)
#     prior.clamp_(1e-4)
#     prior /= prior.sum()
#     log_prior = torch.log(prior)
#     log_prior_file = os.path.join(temp_dir, "log_prior.pt")
#     torch.save(log_prior, log_prior_file)
#     pdfs_a_dir = os.path.join(temp_dir, "pdfs")
#     pdfs_b_dir = os.path.join(temp_dir, "pdfs_")
#     populate_torch_dir(temp_dir, 10, num_filts=4)
#     state_dir = os.path.join(temp_dir, "states")
#     state_csv = os.path.join(temp_dir, "hist.csv")
#     c_params = namedtuple("a", "saved_model_fmt")(saved_model_fmt="foo-{epoch}.pt")
#     controller = DummyTrainingController(
#         c_params, state_csv_path=state_csv, state_dir=state_dir
#     )
#     MParams = namedtuple("MP", "freq_dim target_dim seed")
#     m_params_1 = MParams(freq_dim=4, target_dim=11, seed=1)
#     m_1 = DummyAM(m_params_1)
#     m_params_2 = MParams(freq_dim=4, target_dim=11, seed=2)
#     m_2 = DummyAM(m_params_2)
#     controller.save(1, m_1.state_dict())
#     controller.save(2, m_2.state_dict())
#     assert os.path.isfile(os.path.join(state_dir, "foo-1.pt"))
#     ini_path = os.path.join(temp_dir, "a.ini")
#     with open(ini_path, "w") as f:
#         f.write(
#             "[model]\n"
#             "freq_dim = 4\n"
#             "target_dim = 11\n"
#             "[training]\n"
#             "saved_model_fmt = foo-{epoch}.pt\n"
#         )
#     assert not command_line.acoustic_model_forward_pdfs(
#         [
#             "--config",
#             ini_path,
#             "--device",
#             "cpu",
#             log_prior_file,
#             temp_dir,
#             "path",
#             os.path.join(state_dir, "foo-1.pt"),
#         ]
#     )
#     file_names = os.listdir(pdfs_a_dir)
#     assert len(file_names) == 10
#     assert not command_line.acoustic_model_forward_pdfs(
#         [
#             "--config",
#             ini_path,
#             "--device",
#             "cuda",
#             "--pdfs-dir",
#             pdfs_b_dir,
#             log_prior_file,
#             temp_dir,
#             "history",
#             state_dir,
#             state_csv,
#         ]
#     )
#     assert len(os.listdir(pdfs_b_dir)) == 10
#     for file_name in file_names:
#         pdf_a = torch.load(os.path.join(pdfs_a_dir, file_name), map_location="cpu")
#         pdf_b = torch.load(os.path.join(pdfs_b_dir, file_name), map_location="cpu")
#         assert torch.allclose(pdf_a, pdf_b, atol=1e-4)
#     assert not command_line.acoustic_model_forward_pdfs(
#         [
#             "--config",
#             ini_path,
#             "--device",
#             "cpu",
#             "--pdfs-dir",
#             pdfs_b_dir,
#             log_prior_file,
#             temp_dir,
#             "history",
#             state_dir,
#             state_csv,
#             "--last",
#         ]
#     )
#     for file_name in file_names:
#         pdf_a = torch.load(os.path.join(pdfs_a_dir, file_name), map_location="cpu")
#         pdf_b = torch.load(os.path.join(pdfs_b_dir, file_name), map_location="cpu")
#         assert not torch.allclose(pdf_a, pdf_b, atol=1e-4)


# @pytest.mark.gpu
# def test_train_acoustic_model(temp_dir, populate_torch_dir):
#     torch.manual_seed(10)
#     w = torch.rand(11)
#     weight_pt_file = os.path.join(temp_dir, "label_weights.pt")
#     torch.save(w, weight_pt_file)
#     ini_path = os.path.join(temp_dir, "a.ini")
#     with open(ini_path, "w") as f:
#         f.write(
#             "[model]\n"
#             "freq_dim = 3\n"
#             "target_dim = 11\n"
#             "hidden_sizes = \n"
#             "kernel_sizes = \n"
#             "seed = 1\n"
#             "[data]\n"
#             "context_left = 1\n"
#             "context_right = 1\n"
#             "[training]\n"
#             "log10_learning_rate = -2\n"
#             "num_epochs = 1\n"
#             "seed = 2\n"
#             "saved_model_fmt = foo-{epoch}.pt\n"
#             "[train_data]\n"
#             "seed = 3\n"
#         )
#     state_dir = os.path.join(temp_dir, "states")
#     train_dir = os.path.join(temp_dir, "train")
#     val_dir = os.path.join(temp_dir, "dev")
#     populate_torch_dir(train_dir, 10, num_filts=3)
#     populate_torch_dir(val_dir, 5, num_filts=3)
#     assert not command_line.train_acoustic_model(
#         [
#             "--weight-tensor-file",
#             weight_pt_file,
#             "--config",
#             ini_path,
#             "--device",
#             "cpu",
#             state_dir,
#             train_dir,
#             val_dir,
#         ]
#     )
#     params_a = torch.load(os.path.join(state_dir, "foo-1.pt"), map_location="cpu")
#     os.remove(os.path.join(state_dir, "foo-1.pt"))
#     with open(ini_path, "w") as f:
#         f.write(
#             "[model]\n"
#             "freq_dim = 3\n"
#             "target_dim = 11\n"
#             "hidden_sizes = \n"
#             "kernel_sizes = \n"
#             "seed = 1\n"
#             "[data]\n"
#             "context_left = 1\n"
#             "context_right = 1\n"
#             "[training]\n"
#             "log10_learning_rate = -2\n"
#             "num_epochs = 2\n"
#             "seed = 2\n"
#             "saved_model_fmt = foo-{epoch}.pt\n"
#             "keep_last_and_best_only = false\n"
#             "[train_data]\n"
#             "seed = 3\n"
#         )
#     assert not command_line.train_acoustic_model(
#         [
#             "--weight-tensor-file",
#             weight_pt_file,
#             "--config",
#             ini_path,
#             "--device",
#             "cuda",
#             state_dir,
#             train_dir,
#             val_dir,
#         ]
#     )
#     params_b = torch.load(os.path.join(state_dir, "foo-1.pt"), map_location="cpu")
#     for key in params_a.keys():
#         assert torch.allclose(params_a[key], params_b[key], atol=1e-4)
#     assert os.path.isfile(os.path.join(state_dir, "foo-2.pt"))


# @pytest.mark.gpu
# def test_optimize_acoustic_model(temp_dir, populate_torch_dir):
#     data_dir = os.path.join(temp_dir, "data")
#     in_ini = os.path.join(temp_dir, "in.ini")
#     out_ini = os.path.join(temp_dir, "out.ini")
#     history_db = os.path.join(temp_dir, "history.db")
#     utt_ids = sorted(populate_torch_dir(data_dir, 20, num_filts=3)[-1])
#     with open(in_ini, "w") as f:
#         f.write(
#             "[model]\n"
#             "freq_dim = 3\n"
#             "target_dim = 11\n"
#             "hidden_sizes = \n"
#             "kernel_sizes = \n"
#             "[data]\n"
#             "context_left = 0\n"
#             "context_right = 0\n"
#             "[training]\n"
#             "num_epochs = 5\n"
#             "log10_learning_rate = -300\n"
#             "[train_data]\n"
#             "batch_size = 100\n"
#             "[optim]\n"
#             "to_optimize = log10_learning_rate,batch_size\n"
#             "max_samples = 5\n"
#         )
#     partitions = [os.path.join(temp_dir, x) for x in ("p1.txt", "p2.txt", "p3.txt")]
#     partition_size = len(utt_ids) // 4
#     for partition in partitions:
#         with open(partition, "w") as f:
#             f.write("\n".join(utt_ids[:partition_size]))
#         utt_ids = utt_ids[partition_size:]
#     assert not command_line.optimize_acoustic_model(
#         [
#             "--config",
#             in_ini,
#             "--device",
#             "cpu",
#             "--history-url",
#             "sqlite:///" + history_db,
#             data_dir,
#             str(len(partitions)),
#             out_ini,
#         ]
#     )
#     with open(out_ini) as f:
#         lines = f.read()
#     train_data_idx = lines.index("[train_data]")
#     batch_size_idx = lines.index("batch_size", train_data_idx)
#     batch_line = lines[batch_size_idx:].split("\n")[0]
#     batch_size = int(batch_line.split("=")[1].strip())
#     assert batch_size != 30
#     log10_lr_idx = lines.index("log10_learning_rate")
#     log10_lr_line = lines[log10_lr_idx:].split("\n")[0]
#     log10_lr = float(log10_lr_line.split("=")[1].strip())
#     assert log10_lr != -300
#     assert not command_line.optimize_acoustic_model(
#         ["--config", in_ini, "--device", "cuda", data_dir] + partitions + [out_ini]
#     )

