import os

import numpy
import torch

try:
    import cupy
except ModuleNotFoundError:
    pass

__all__ = [
    "set_seed_everywhere",
    "get_backend",
    "get_vector_type",
]


def set_seed_everywhere(seed=0):
    """Set random seed on every computation engine"""
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUPY_SEED"] = str(seed)
    

def get_backend(vector):
    """
    Get the vector content backend
    Args:
        vector: vector to analyze

    Returns: package (one of numpy, cupy, torch)
    """
    if vector.whoami == "VectorTorch":
        backend = torch
    elif vector.whoami == "VectorNumpy":
        backend = numpy
    elif vector.whoami == "VectorCupy":
        backend = cupy
        
    return backend


def get_vector_type(vector):
    """
    Get the vector content original classs
    Args:
        vector: vector to analyze

    Returns: array class (numpy.ndarray, cupy.ndarray, torch.Tensor)
    """
    if vector.whoami == "VectorTorch":
        return torch.Tensor
    elif vector.whoami == "VectorNumpy":
        return numpy.ndarray
    elif vector.whoami == "VectorCupy":
        return cupy.ndarray
