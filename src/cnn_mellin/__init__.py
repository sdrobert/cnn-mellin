__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2021 Sean Robertson"


__all__ = ["layers", "models", "running", "construct_default_param_dict"]


def construct_default_param_dict():
    import cnn_mellin.models as models
    import cnn_mellin.running as running
    import pydrobert.torch.data as data
    from collections import OrderedDict

    return OrderedDict(
        (
            ("model", models.AcousticModelParams(name="model")),
            ("training", running.MyTrainingStateParams(name="training")),
            ("data", data.SpectDataSetParams(name="data")),
        )
    )
