import torch
import numpy as np


def set_backends():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_seed_everywhere(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


__all__ = [
    "set_backends",
    "set_seed_everywhere",
]
