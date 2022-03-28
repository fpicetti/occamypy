import torch
import numpy as np
from GPUtil import getFirstAvailable


__all__ = [
    "set_backends",
    "set_seed_everywhere",
    "get_device",
    "get_device_name",
]


def set_backends():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_seed_everywhere(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device(devID: int = None) -> torch.device:
    """
    Decide
    :param devID: device id to be used (None for CPU, -1 for max free memory)
    :return: torch.device object
    """
    if devID is None:
        dev = "cpu"
    elif devID == -1:
        dev = getFirstAvailable(order='memory')[0]
    else:
        dev = int(devID)
        if dev > torch.cuda.device_count():
            dev = "cpu"
            raise UserWarning("The selected device is not available, switched to CPU.")

    return torch.device(dev)


def get_device_name(devID: int = None) -> str:
    if devID is None or isinstance(torch.cuda.get_device_name(devID), torch.NoneType):
        return "CPU"
    else:
        return "GPU %d - %s" % (devID, torch.cuda.get_device_name(devID))
    