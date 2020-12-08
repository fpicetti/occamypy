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


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    DIM = 1
    
    if DIM == 1:
        # 1D
        dt = 0.005
        nt = 100
        t = VectorNumpy(np.arange(nt) * dt)
        f0 = 10
        nfft = 2 ** 10
        x = VectorNumpy(np.sin(2 * np.pi * f0 * t.getNdArray()))
        
        F = FFT(x, nfft=nfft, sampling=dt)
        F.dotTest(True)
        
        spectrum = F * x
        x_inv = F.H * spectrum
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(t.getNdArray(), x.getNdArray(), 'k', lw=2, label='True')
        axs[0].plot(t.getNdArray(), x_inv.getNdArray().real, '--r', lw=2, label='Inverted')
        axs[0].legend()
        axs[0].set_title('Signal')
        
        axs[1].plot(F.fs[0][:F.nfft[0] // 2], np.abs(spectrum.getNdArray()[:F.nfft[0] // 2]), 'k', lw=2)
        axs[1].set_title('Fourier Transform')
        axs[1].set_xlim(0, 3 * f0)
        plt.show()
    
    elif DIM == 3:
        # 3D
        dt, dx, dy = 0.005, 5, 3
        nt, nx, ny = 30, 21, 11
        t = np.arange(nt) * dt
        x = np.arange(nx) * dx
        y = np.arange(nx) * dy
        f0 = 10
        nfftt = 64
        nfftk = 32
    
        d = np.outer(np.sin(2 * np.pi * f0 * t), np.arange(nx) + 1)
        d = np.tile(d[:, :, np.newaxis], [1, 1, ny])
        
        d = VectorNumpy(d.astype(np.complex))
        F = FFT(model=d, nfft=(nfftt, nfftk, nfftk), sampling=(dt, dx, dy))
        F.dotTest(True)
        
        D = F * d
        dinv = F.H * D
    
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0][0].imshow(d.getNdArray()[:, :, ny // 2].real,
                         vmin=-20, vmax=20, cmap='seismic')
        axs[0][0].set_title('Signal')
        axs[0][0].axis('tight')
        axs[0][1].imshow(np.abs(np.fft.fftshift(D.getNdArray(), axes=1)[:20, :, nfftk // 2]),
                         cmap='seismic')
        axs[0][1].set_title('Fourier Transform')
        axs[0][1].axis('tight')
        axs[1][0].imshow(dinv.getNdArray()[:, :, ny // 2].real,
                         vmin=-20, vmax=20, cmap='seismic')
        axs[1][0].set_title('Inverted')
        axs[1][0].axis('tight')
        axs[1][1].imshow(d.getNdArray()[:, :, ny // 2].real - dinv.getNdArray()[:, :, ny // 2].real,
                         vmin=-20, vmax=20, cmap='seismic')
        axs[1][1].set_title('Error')
        axs[1][1].axis('tight')
        fig.tight_layout()
        plt.show()
    
    print('')
