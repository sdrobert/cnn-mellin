def get_num_avail_cores() -> int:
    import os

    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        return os.cpu_count()


def construct_default_param_dict():
    import models as models
    import running as running
    import pydrobert.torch.data as data
    from collections import OrderedDict

    return OrderedDict(
        (
            ("model", models.AcousticModelParams(name="model")),
            ("training", running.MyTrainingStateParams(name="training")),
            ("data", data.SpectDataSetParams(name="data")),
        )
    )
