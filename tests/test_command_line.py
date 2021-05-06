import os

import torch
import cnn_mellin.command_line as command_line
from pydrobert.torch.command_line import get_torch_spect_data_dir_info


def test_train_command(temp_dir, device, populate_torch_dir):
    C, F, V = 100, 10, 10
    train_dir = os.path.join(temp_dir, "train")
    populate_torch_dir(train_dir, C, num_filts=F, max_class=V - 1)
    ext_dir = os.path.join(temp_dir, "ext")
    os.makedirs(ext_dir)
    model_1_dir = os.path.join(temp_dir, "models_1")
    ini_file = os.path.join(temp_dir, "cfg.ini")
    with open(ini_file, "w") as file_:
        file_.write("[model]\n")
        file_.write("seed=1\n")
        file_.write("[training]\n")
        file_.write("dropout_prob=0.05\n")
        file_.write("num_epochs=2\n")
        file_.write("seed=2\n")
    assert not get_torch_spect_data_dir_info(
        [train_dir, os.path.join(ext_dir, "train.info.ark")]
    )
    args = [
        "--read-ini",
        ini_file,
        "--model-dir",
        model_1_dir,
        "--device",
        str(device),
        "train",
        train_dir,
        "--num-data-workers",
        "0",
        "--quiet",
    ]
    assert not command_line.cnn_mellin(args)
    model_1_hist_dir = os.path.join(model_1_dir, "training")
    assert os.path.isdir(model_1_hist_dir)
    assert len(os.listdir(model_1_hist_dir)) == 4  # 2 model files, 2 optim files
    model_1_path = os.path.join(model_1_dir, "model.pt")
    assert os.path.isfile(model_1_path)
    model_2_dir = os.path.join(temp_dir, "models_2")
    args[3] = model_2_dir
    assert not command_line.cnn_mellin(args)
    model_2_hist_dir = os.path.join(model_2_dir, "training")
    assert os.path.isdir(model_2_hist_dir)
    assert len(os.listdir(model_2_hist_dir)) == 4
    model_2_path = os.path.join(model_2_dir, "model.pt")
    assert os.path.isfile(model_2_path)
    model_1_state_dict = torch.load(model_1_path)
    model_2_state_dict = torch.load(model_2_path)
    assert set(model_1_state_dict) == set(model_2_state_dict)
    for key in model_1_state_dict:
        v1, v2 = model_1_state_dict[key], model_2_state_dict[key]
        if isinstance(v1, torch.Tensor):
            assert torch.allclose(v1, v2), key
        else:
            assert v1 == v2, key
