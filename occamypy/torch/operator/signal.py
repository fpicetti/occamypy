from itertools import accumulate, product
from typing import Union, List, Tuple

import numpy as np
import torch

from occamypy.operator.base import Operator, Dstack
from occamypy.torch.back_utils import set_backends
from occamypy.torch.vector import VectorTorch
from occamypy.vector.base import superVector

set_backends()


def _gaussian_kernel1d(sigma: float, order: int = 0, truncate: float = 4.) -> torch.Tensor:
    """Computes a 1-D Gaussian convolution kernel"""

    radius = int(truncate * sigma + 0.5)
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = torch.arange(order + 1)
    sigma2 = sigma * sigma
    x = torch.arange(-radius, radius+1)
    phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = torch.zeros(order + 1)
        q[0] = 1
        D = torch.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = torch.diag(torch.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x
    

class ConvND(Operator):
    """ND convolution square operator in the domain space"""
    
    def __init__(self, domain: VectorTorch, kernel: Union[VectorTorch, torch.Tensor]):
        """
        ConvND (torch) constructor

        Args:
            domain: domain vector
            kernel: kernel vector or tensor
        """
        if isinstance(kernel, VectorTorch):
            self.kernel = kernel.getNdArray().clone()
        elif isinstance(kernel, torch.Tensor):
            self.kernel = kernel.clone()
        
        if domain.ndim != self.kernel.ndim:
            raise ValueError("Domain and kernel number of dimensions mismatch")
        
        if domain.device != self.kernel.device:
            raise ValueError("Domain and kernel has to live in the same device")
        
        self.kernel_size = tuple(self.kernel.shape)
        self.pad_size = tuple([k // 2 for k in self.kernel_size])

        if domain.ndim == 1:
            corr = torch.nn.functional.conv1d
        elif domain.ndim == 2:
            corr = torch.nn.functional.conv2d
        elif domain.ndim == 3:
            corr = torch.nn.functional.conv3d
        else:
            raise ValueError
        
        # torch.nn functions require batch and channel dimensions
        self.conv = lambda x: corr(x.unsqueeze(0).unsqueeze(0),
                                   torch.flip(self.kernel, dims=tuple(range(self.kernel.ndim))).unsqueeze(0).unsqueeze(0),
                                   padding=self.pad_size).flatten(end_dim=2)

        self.corr = lambda x: corr(x.unsqueeze(0).unsqueeze(0),
                                   self.kernel.unsqueeze(0).unsqueeze(0),
                                   padding=self.pad_size).flatten(end_dim=2)
        
        super(ConvND, self).__init__(domain, domain, name="Convolve")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        data[:] += self.conv(model.getNdArray())
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        model[:] += self.corr(data.getNdArray())
        return
    

class GaussianFilter(ConvND):
    """Gaussian smoothing operator"""
    
    def __init__(self, domain, sigma):
        """
        GaussianFilter (torch) constructor
        
        Args:
            domain: domain vector
            sigma: standard deviation along the domain directions
        """
        self.sigma = [sigma] if isinstance(sigma, (float, int)) else sigma
        if not isinstance(self.sigma, (list, tuple)):
            raise TypeError("sigma has to be either a list or a tuple")
        self.scaling = np.sqrt(np.prod(np.array(self.sigma) / np.pi))
        kernels = [_gaussian_kernel1d(s) for s in self.sigma]
        
        self.kernel = [*accumulate(kernels, lambda a, b: torch.outer(a.flatten(), b))][-1]
        self.kernel.reshape([k.numel() for k in kernels])
        super(GaussianFilter, self).__init__(domain, self.kernel.to(domain.device))
        self.name = "GaussFilt"


def Padding(domain, pad, mode="constant"):
    """
    Padding operator

    Notes:
        To pad 2 values to each side of the first dim, and 3 values to each side of the second dim, use:
            pad=((2,2), (3,3))

    Args:
        domain: domain vector
        pad: number of samples to be added at each end of the dimension, for each dimension
        mode: padding mode (see https://pytorch.org/docs/1.10/generated/torch.nn.functional.pad.html)
    """
    if isinstance(domain, VectorTorch):
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


class _Padding(Operator):
    
    def __init__(self, domain: VectorTorch, pad: Union[int, Tuple[int]], mode: str = "constant"):
        
        nd = domain.ndim
        
        if isinstance(pad, int):
            pad = [int(pad), int(pad)] * nd
        else:
            pad = np.asarray(pad).ravel().tolist()
            if len(pad) != 2 * nd:
                raise ValueError("len(pad) has to be 2*nd")
            
        if (np.array(pad) < 0).any():
            raise ValueError('Padding must be positive or zero')
        
        self.pad = list(pad)
        
        self.padded_shape = tuple(np.asarray(domain.shape) + [self.pad[i] + self.pad[i + 1] for i in range(0, 2 * nd, 2)])
        
        super(_Padding, self).__init__(domain, VectorTorch(self.padded_shape, device=domain.device.index), name="Paddding")

        self.inner_idx = [list(torch.arange(start=self.pad[0:-1:2][i], end=self.range.shape[i]-pad[1::2][i])) for i in range(nd)]
        self.mode = mode
        
    def forward(self, add, model, data):
        """Pad the domain"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # torch counts the axes in reverse order
        if self.mode == "constant":
            padded = torch.nn.functional.pad(model.getNdArray(), self.pad[::-1], mode=self.mode)
        else:
            padded = torch.nn.functional.pad(model.getNdArray()[None], self.pad[::-1], mode=self.mode).squeeze(0)
        data[:] += padded
        return
    
    def adjoint(self, add, model, data):
        """Extract original subsequence"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        x = torch.Tensor([data[coord] for coord in product(*self.inner_idx)]).reshape(self.domain.shape).to(model.device)
        model[:] += x
        return
