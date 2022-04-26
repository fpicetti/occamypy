from typing import Union

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve, correlate

from occamypy.operator.base import Operator, Dstack
from occamypy.numpy.vector import VectorNumpy
from occamypy.vector.base import superVector


class GaussianFilter(Operator):
    """Gaussian smoothing operator using scipy smoothing"""

    def __init__(self, domain, sigma):
        """
        GaussianFilter (numpy) constructor

        Args:
            domain: domain vector
            sigma: standard deviation along the domain directions
        """
        self.sigma = sigma
        self.scaling = np.sqrt(np.prod(np.array(self.sigma) / np.pi))  # in order to have the max amplitude 1
        super(GaussianFilter, self).__init__(domain, domain, name="GaussFilt")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        data_arr[:] += self.scaling * gaussian_filter(model_arr, sigma=self.sigma)
        return
    
    def adjoint(self, add, model, data):
        self.forward(add, data, model)
        return


class ConvND(Operator):
    """ND convolution square operator in the domain space"""

    def __init__(self, domain: VectorNumpy, kernel: Union[VectorNumpy, np.ndarray], method='auto'):
        """
        ConvND (numpy) constructor

        Args:
            domain: domain vector
            kernel: kernel vector
            method: how to compute the convolution [auto, direct, fft]
        """
        if isinstance(kernel, VectorNumpy):
            self.kernel = kernel.clone().getNdArray()
        elif isinstance(kernel, np.ndarray):
            self.kernel = kernel.copy()
        else:
            raise ValueError("kernel has to be either a vector or a numpy.ndarray")
        
        # Padding array to avoid edge effects
        pad_width = []
        for len_filt in self.kernel.shape:
            half_len = int(len_filt / 2)
            if np.mod(len_filt, 2):
                padding = (half_len, half_len)
            else:
                padding = (half_len, half_len - 1)
            pad_width.append(padding)
        self.kernel = np.pad(self.kernel, pad_width, mode='constant')
        
        if len(domain.shape) != len(self.kernel.shape):
            raise ValueError("Domain and kernel number of dimensions mismatch")
        
        if method not in ["auto", "direct", "fft"]:
            raise ValueError("method has to be auto, direct or fft")
        self.method = method
        
        super(ConvND, self).__init__(domain=domain, range=domain, name="Convolve")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()[:]
        dataNd += convolve(modelNd, self.kernel, mode='same', method=self.method)
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()[:]
        modelNd += correlate(dataNd, self.kernel, mode='same', method=self.method)
        return


def Padding(domain, pad, mode: str = "constant"):
    """
    Padding operator

    Notes:
        To pad 2 values to each side of the first dim, and 3 values to each side of the second dim, use:
            pad=((2,2), (3,3))

    Args:
        domain: domain vector
        pad: number of samples to be added at each end of the dimension, for each dimension
        mode: padding mode (see https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
    """
    if isinstance(domain, VectorNumpy):
        return _Padding(domain, pad, mode)
    elif isinstance(domain, superVector):
        return Dstack([_Padding(v, pad, mode) for v in domain.vecs])
    else:
        raise ValueError("ERROR! Provided domain has to be either vector or superVector")


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
    return Padding(domain, pad, mode="constant")


def _pad_VectorNumpy(vec, pad):
    if not isinstance(vec, VectorNumpy):
        raise ValueError("ERROR! Provided vector must be a VectorNumpy")
    if len(vec.shape) != len(pad):
        raise ValueError("Dimensions of vector and padding mismatch!")
    
    vec_new_shape = tuple(np.asarray(vec.shape) + [sum(pad[_]) for _ in range(len(pad))])
    return VectorNumpy(np.empty(vec_new_shape, dtype=vec.getNdArray().dtype))


class _Padding(Operator):
    
    def __init__(self, domain: VectorNumpy, pad, mode: str = "constant"):

        self.dims = domain.shape
        pad = [(pad, pad)] * len(self.dims) if isinstance(pad, int) else list(pad)
        if (np.array(pad) < 0).any():
            raise ValueError('Padding must be positive or zero')
        self.pad = pad
        self.mode = mode
        super(_Padding, self).__init__(domain, _pad_VectorNumpy(domain, self.pad), name="Padding")
    
    def forward(self, add, model, data):
        """Pad the domain"""
        self.checkDomainRange(model, data)
        if add:
            temp = data.clone()
        y = np.pad(model.arr, self.pad, mode=self.mode)
        data.arr = y
        if add:
            data.scaleAdd(temp, 1., 1.)
        return
    
    def adjoint(self, add, model, data):
        """Extract original subsequence"""
        self.checkDomainRange(model, data)
        if add:
            temp = model.clone()
        x = data.clone().arr
        for ax, pad in enumerate(self.pad):
            x = np.take(x, pad[0] + np.arange(self.dims[ax]), axis=ax)
        model.arr = x
        if add:
            model.scaleAdd(temp, 1., 1.)
        return
