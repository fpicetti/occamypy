import numpy as np

from occamypy.operator.base import Operator
from occamypy.numpy.vector import VectorNumpy


class FFT(Operator):
    """N-dimensional Fast Fourier Transform for complex input"""

    def __init__(self, domain, axes=None, nfft=None, sampling=None):
        """
        FFT (numpy) constructor

        Args:
            domain: domain vector
            axes: index of axes on which the FFT is computed
            nfft: number of frequency bins for each axis
            sampling: sampling step on each axis
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
        
        self.fs = [np.fft.fftfreq(n, d=s) for n, s in zip(nfft, sampling)]
        
        dims_fft = np.asarray(domain.shape)
        for a, n in zip(self.axes, self.nfft):
            dims_fft[a] = n
        
        super(FFT, self).__init__(domain=VectorNumpy(np.zeros(domain.shape, dtype=complex)),
                                  range=VectorNumpy(np.zeros(shape=dims_fft, dtype=complex)),
                                  name="FFT")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        data[:] += np.fft.fftn(model[:], s=self.nfft, axes=self.axes, norm='ortho')
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        # here we need to separate the computation and use np.take for handling nfft > model.shape
        temp = np.fft.ifftn(data[:], s=self.nfft, axes=self.axes, norm='ortho')
        for a in self.axes:
            temp = np.take(temp, range(self.domain.shape[a]), axis=a)
        model[:] += temp
        return
