from itertools import product

import torch
import torch.fft as fft
from numpy.fft import fftfreq

from occamypy.operator.base import Operator
from occamypy.torch.back_utils import set_backends
from occamypy.torch.vector import VectorTorch

set_backends()

__all__ = [
    "FFT"
]


class FFT(Operator):
    """N-dimensional Fast Fourier Transform for complex input"""
    
    def __init__(self, domain, axes=None, nfft=None, sampling=None):
        """
        FFT (torch) constructor

        Args:
            domain: domain vector
            axes: dimension along which FFT is computed (all by default)
            nfft: number of samples in Fourier Transform for each direction (same as domain by default)
            sampling: sampling steps on each axis (1. by default)
        """
        if axes is None:
            axes = tuple(range(domain.ndim))
        elif not isinstance(axes, tuple) and domain.ndim == 1:
            axes = (axes,)
        if nfft is None:
            nfft = domain.shape
        elif not isinstance(nfft, tuple) and domain.ndim == 1:
            nfft = (nfft,)
        if sampling is None:
            sampling = tuple([1.] * domain.ndim)
        elif not isinstance(sampling, tuple) and domain.ndim == 1:
            sampling = (sampling,)
        
        if len(axes) != len(nfft) != len(sampling):
            raise ValueError('axes, nffts, and sampling must have same number of elements')
        
        self.axes = axes
        self.nfft = nfft
        self.sampling = sampling
        self.fs = [fftfreq(n, d=s) for n, s in zip(self.nfft, self.sampling)]
        
        dims_fft = list(domain.shape)
        for a, n in zip(self.axes, self.nfft):
            dims_fft[a] = n
        self.inner_idx = [torch.arange(0, domain.shape[i]) for i in range(len(dims_fft))]
        
        super(FFT, self).__init__(domain=VectorTorch(torch.zeros(domain.shape).type(torch.double)),
                                  range=VectorTorch(torch.zeros(dims_fft).type(torch.complex128)),
                                  name="FFT")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        data[:] += fft.fftn(model.getNdArray(), s=self.nfft, dim=self.axes, norm='ortho')
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        # compute IFFT
        x = fft.ifftn(data.getNdArray(), s=self.nfft, dim=self.axes, norm='ortho').type(model.getNdArray().dtype)
        # handle nfft > model.shape
        x = torch.Tensor([x[coord] for coord in product(*self.inner_idx)]).reshape(self.domain.shape).to(model.device)
        model[:] += x
        return
