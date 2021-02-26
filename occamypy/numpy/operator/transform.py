import numpy as np

from occamypy import Operator
from ..vector import VectorNumpy


class FFT(Operator):
    """
    N-dimensional Fast Fourier Transform for complex input.
    
    :param model : :obj:`Vector`
        Domain vector
    :param axes : :obj:`tuple`, optional
        Dimension along which FFT is computed (all by default)
    :param nfft : :obj:`tuple`, optional
        Number of samples in Fourier Transform for each direction (same as model by default)
    :param sampling : :obj:`tuple`, optional
        Sampling steps on each axis (1. by default)
    """
    
    def __init__(self, model, axes=None, nfft=None, sampling=None):
        
        if axes is None:
            axes = tuple(range(model.ndim))
        elif not isinstance(axes, tuple) and model.ndim == 1:
            axes = (axes,)
        if nfft is None:
            nfft = model.shape
        elif not isinstance(nfft, tuple) and model.ndim == 1:
            nfft = (nfft,)
        if sampling is None:
            sampling = tuple([1.] * model.ndim)
        elif not isinstance(sampling, tuple) and model.ndim == 1:
            sampling = (sampling,)
        
        if len(axes) != len(nfft) != len(sampling):
            raise ValueError('axes, nffts, and sampling must have same number of elements')

        self.axes = axes
        self.nfft = nfft
        self.sampling = sampling
        
        self.fs = [np.fft.fftfreq(n, d=s) for n, s in zip(nfft, sampling)]
        
        dims_fft = np.asarray(model.shape)
        for a, n in zip(self.axes, self.nfft):
            dims_fft[a] = n
        
        super(FFT, self).__init__(domain=VectorNumpy(np.zeros(model.shape, dtype=np.complex)),
                                  range=VectorNumpy(np.zeros(shape=dims_fft, dtype=np.complex)))
    
    def __str__(self):
        return 'numpyFFT'
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()
        dataNd += np.fft.fftn(modelNd, s=self.nfft, axes=self.axes, norm='ortho')
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()
        # here we need to separate the computation and use np.take for handling nfft > model.shape
        temp = np.fft.ifftn(dataNd, s=self.nfft, axes=self.axes, norm='ortho')
        for a in self.axes:
            temp = np.take(temp, range(self.domain.shape[a]), axis=a)
        modelNd += temp
        return
