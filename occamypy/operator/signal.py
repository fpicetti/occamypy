from typing import Tuple, Union

from occamypy.utils.os import CUPY_ENABLED

from occamypy.numpy import operator as np
from occamypy.torch import operator as tc
if CUPY_ENABLED:
    from occamypy.cupy import operator as cp


__all__ = [
    "FFT",
    "ConvND",
    "GaussianFilter",
    "Padding",
    "ZeroPad",
]


def FFT(domain, axes=None, nfft=None, sampling=None):
    """
    N-dimensional Fast Fourier Transform
    
    Args:
        domain: domain vector
        axes: dimension along which FFT is computed (all by default)
        nfft: number of samples in Fourier Transform for each direction (same as domain by default)
        sampling: sampling steps on each axis (1. by default)
    """
    if domain.whoami == "VectorNumpy":
        return np.FFT(domain=domain, axes=axes, nfft=nfft, sampling=sampling)
    elif domain.whoami == "VectorTorch":
        return tc.FFT(domain=domain, axes=axes, nfft=nfft, sampling=sampling)
    elif domain.whoami == "VectorCupy":
        raise NotImplementedError("FFT operator for VectorCupy is not implemented yet")
    else:
        raise TypeError("Domain vector not recognized")
    
    
def ConvND(domain, kernel, method: str = 'auto'):
    """
    ND convolution square operator in the domain space
    
    Args:
        domain: domain vector
        kernel: kernel vector
        method: how to compute the convolution [auto, direct, fft]
    """
    if domain.whoami == "VectorNumpy":
        return np.ConvND(domain=domain, kernel=kernel, method=method)
    elif domain.whoami == "VectorTorch":
        return tc.ConvND(domain=domain, kernel=kernel)
    elif domain.whoami == "VectorCupy":
        return cp.ConvND(domain=domain, kernel=kernel, method=method)
    else:
        raise TypeError("Domain vector not recognized")


def GaussianFilter(domain, sigma: Tuple[float]):
    """
    Gaussian smoothing operator
    
    Args:
        domain: domain vector
        sigma: standard deviation along the domain directions
    """
    if domain.whoami == "VectorNumpy":
        return np.GaussianFilter(domain=domain, sigma=sigma)
    elif domain.whoami == "VectorTorch":
        return tc.GaussianFilter(domain=domain, sigma=sigma)
    elif domain.whoami == "VectorCupy":
        return cp.GaussianFilter(domain=domain, sigma=sigma)
    else:
        raise TypeError("Domain vector not recognized")


def Padding(domain, pad: Union[int, Tuple[int]], mode: str = "constant"):
    """
    Padding operator

    Notes:
        To pad 2 values to each side of the first dim, and 3 values to each side of the second dim, use:
            pad=((2,2), (3,3))

    Args:
        domain: domain vector
        pad: number of samples to be added at each end of the dimension, for each dimension
        mode: padding mode
    """
    if domain.whoami == "VectorNumpy":
        return np.Padding(domain=domain, pad=pad, mode=mode)
    elif domain.whoami == "VectorTorch":
        return tc.Padding(domain=domain, pad=pad, mode=mode)
    elif domain.whoami == "VectorCupy":
        return cp.Padding(domain=domain, pad=pad, mode=mode)
    else:
        raise TypeError("Domain vector not recognized")


def ZeroPad(domain, pad):
    """
    Zero-Padding operator

    Notes:
        To pad 2 values to each side of the first dim, and 3 values to each side of the second dim, use:
            pad=((2,2), (3,3))

    Args:
        domain: domain vector
        pad: number of samples to be added at each end of the dimension, for each dimension
    """
    return Padding(domain=domain, pad=pad)
