import numpy
import torch

try:
    import cupy
except ModuleNotFoundError:
    pass

__all__ = [
    "get_backend",
    "get_vector_type",
]


def get_backend(vector):
    
    if vector.whoami == "VectorTorch":
        backend = torch
    elif vector.whoami == "VectorNumpy":
        backend = numpy
    elif vector.whoami == "VectorCupy":
        backend = cupy
        
    return backend


def get_vector_type(vector):
    if vector.whoami == "VectorTorch":
        return torch.Tensor
    elif vector.whoami == "VectorNumpy":
        return numpy.ndarray
    elif vector.whoami == "VectorCupy":
        return cupy.ndarray
