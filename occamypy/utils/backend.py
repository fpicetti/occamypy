import numpy
import torch

try:
    import cupy
except ModuleNotFoundError:
    pass

__all__ = [
    "get_backend",
]


def get_backend(vector):
    
    if vector.whoami == "VectorTorch":
        backend = torch
    elif vector.whoami == "VectorNumpy":
        backend = numpy
    elif vector.whoami == "VectorCupy":
        backend = cupy
        
    return backend
