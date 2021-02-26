from __future__ import division, print_function, absolute_import
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
try:
    from cusignal.convolution import convolve, correlate
except ModuleNotFoundError:
    raise ModuleNotFoundError("cuSIGNAL is not installed. Please install it")
from occamypy import Operator, Dstack, Vector, superVector
from occamypy.cupy import VectorCupy


class GaussianFilter(Operator):
    def __init__(self, model, sigma):
        """
        Gaussian smoothing operator using scipy smoothing:
        :param model : vector class;
            domain vector
        :param sigma : scalar or sequence of scalars;
            standard deviation along the model directions
        """
        self.sigma = sigma
        self.scaling = np.sqrt(np.prod(np.array(self.sigma) / cp.pi))  # in order to have the max amplitude 1
        
        super(GaussianFilter, self).__init__(model, model)
        return
    
    def __str__(self):
        return "GausFilt"
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        data_arr[:] += self.scaling * gaussian_filter(model_arr, sigma=self.sigma)
        return
    
    def adjoint(self, add, model, data):
        """Self-adjoint operator"""
        self.forward(add, data, model)
        return


class ConvND(Operator):
    """
    ND convolution square operator in the domain space

    :param domain   : [no default] - vector class; domain vector
    :param kernel   : [no default] - vector class; kernel vector
    :param method   : [auto] - str; how to compute the convolution [auto, direct, fft]
    :return         : Convolution Operator
    """
    
    def __init__(self, domain, kernel, method='auto'):
        
        if isinstance(kernel, Vector):
            self.kernel = kernel.clone().getNdArray()
        elif isinstance(kernel, cp.ndarray):
            self.kernel = kernel.copy()
        else:
            raise ValueError("kernel has to be either a vector or a cupy.ndarray")
        
        # Padding array to avoid edge effects
        pad_width = []
        for len_filt in self.kernel.shape:
            half_len = int(len_filt / 2)
            if np.mod(len_filt, 2):
                padding = (half_len, half_len)
            else:
                padding = (half_len, half_len - 1)
            pad_width.append(padding)
        self.kernel = cp.padding(self.kernel, pad_width, mode='constant')
        
        if len(domain.shape()) != len(self.kernel.shape):
            raise ValueError("Domain and kernel number of dimensions mismatch")
        
        assert method in ["auto", "direct", "fft"], "method has to be auto, direct or fft"
        self.method = method
        
        super(ConvND, self).__init__(domain, domain)
    
    def __str__(self):
        return " ConvOp "
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()[:]
        dataNd += convolve(modelNd, self.kernel)
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()[:]
        modelNd += correlate(dataNd, self.kernel)
        return


def ZeroPad(domain, pad):
    if isinstance(domain, VectorCupy):
        return _ZeroPadIC(domain, pad)
    elif isinstance(domain, superVector):
        # TODO add the possibility to have different padding for each sub-vector
        return Dstack([_ZeroPadIC(v, pad) for v in domain.vecs])
    else:
        raise ValueError("ERROR! Provided domain has to be either vector or superVector")


def _pad_vectorIC(vec, pad):
    if not isinstance(vec, VectorCupy):
        raise ValueError("ERROR! Provided vector must be of vectorCcupy type")
    assert len(vec.shape) == len(pad), "Dimensions of vector and padding mismatch!"
    
    vec_new_shape = tuple(cp.asarray(vec.shape) + [sum(pad[_]) for _ in range(len(pad))])
    if isinstance(vec, VectorCupy):
        return VectorCupy(cp.empty(vec_new_shape, dtype=vec.getNdArray().dtype))
    else:
        raise ValueError("ERROR! For now only vectorCupy is supported!")


class _ZeroPadIC(Operator):
    
    def __init__(self, domain, pad):
        """ Zero Pad operator.

        To pad 2 values to each side of the first dim, and 3 values to each side of the second dim, use:
            pad=((2,2), (3,3))
        :param domain: vectorIC class
        :param pad: scalar or sequence of scalars
            Number of samples to pad in each dimension.
            If a single scalar is provided, it is assigned to every dimension.
        """
        if isinstance(domain, VectorCupy):
            self.dims = domain.shape
            pad = [(pad, pad)] * len(self.dims) if pad is cp.isscalar else list(pad)
            if (cp.array(pad) < 0).any():
                raise ValueError('Padding must be positive or zero')
            self.pad = pad
            super(_ZeroPadIC, self).__init__(domain, _pad_vectorIC(domain, self.pad))
    
    def __str__(self):
        return "ZeroPad "
    
    def forward(self, add, model, data):
        """Zero padding"""
        self.checkDomainRange(model, data)
        if add:
            temp = data.clone()
        y = cp.padding(model.getNdArray(), self.pad, mode='constant')
        data.getNdArray()[:] = y
        if add:
            data.scaleAdd(temp, 1., 1.)
        return
    
    def adjoint(self, add, model, data):
        """Extract non-zero subsequence"""
        self.checkDomainRange(model, data)
        if add:
            temp = model.clone()
        x = data.clone().getNdArray()
        for ax, pad in enumerate(self.pad):
            x = cp.take(x, pad[0] + cp.arange(self.dims[ax]), axis=ax)
        model.arr = x
        if add:
            model.scaleAdd(temp, 1., 1.)
        return
