from typing import Union
import numpy as np
import torch
from occamypy import superVector, Operator, Dstack
from ..vector import VectorTorch


def gaussian_kernel(std: float, ntaps: int = None, sym=True) -> torch.Tensor:
    if ntaps is None:
        ntaps = int(10 * std)
    assert ntaps > 1
    odd = ntaps % 2
    if not sym and not odd:
        ntaps = ntaps + 1
    n = torch.arange(0, ntaps) - (ntaps - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


class GaussianFilter(Operator):
    def __init__(self, model: VectorTorch, sigma: float):
        """
        Gaussian smoothing operator:
        :param model : vector class;
            domain vector
        :param sigma : scalar or sequence of scalars;
            standard deviation along the model directions
        """
        assert model.ndim < 4, "GaussianFilter supports at max 3D operations"
        self.sigma = sigma
        self.scaling = np.sqrt(np.prod(np.array(self.sigma) / np.pi))  # in order to have the max amplitude 1
        super(GaussianFilter, self).__init__(model, model)
        
        # 1D gaussian kernel has an odd number of taps that is:
        # - 10*sigma in order to get a good kernel, or
        # - min(model.shape) in order to have a kernel constrained for the convolution
        kernel_size = min(*model.shape, int(10*sigma))
        if not kernel_size % 2:
            kernel_size += 1
        self.kernel = gaussian_kernel(sigma, kernel_size)
        
        if model.ndim == 1:
            self.conv = torch.nn.ConvTranspose1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
            # self.conv = torch.nn.functional.conv_transpose1d
            w = self.kernel
        elif model.ndim == 2:
            self.conv = torch.nn.ConvTranspose2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
            # self.conv = torch.nn.functional.conv_transpose2d
            w = torch.outer(self.kernel, self.kernel)
        elif model.ndim == 3:
            self.conv = torch.nn.ConvTranspose3d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
            # self.conv = torch.nn.functional.conv_transpose3d
            w = torch.outer(self.kernel, torch.outer(self.kernel, self.kernel))
        else:
            raise ValueError

        self.conv.weight = torch.nn.Parameter(w.unsqueeze(0).unsqueeze(0))
        self.conv.weight.requires_grad = False
        
    def __str__(self):
        return "GausFilt"
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays and adding the batch and channel dimensions
        model_arr = model.getNdArray().clone().unsqueeze(0).unsqueeze(0)
        # compute convolution
        data_arr = self.conv(model_arr)
        # remove batch and channel dimensions
        data_arr = torch.flatten(data_arr, end_dim=2)
        # write on data vector
        data[:] += self.scaling * data_arr
        return
    
    def adjoint(self, add, model, data):
        """Self-adjoint operator"""
        self.forward(add, data, model)
        return
    

class ConvND(Operator):
    """
    ND convolution square operator in the domain space

    :param model  : [no default] - vector class; domain vector
    :param kernel : [no default] - vector class; kernel vector
    :return       : Convolution Operator
    """
    
    def __init__(self, model: VectorTorch,  kernel: Union[VectorTorch, torch.Tensor]):
        
        if isinstance(kernel, VectorTorch):
            self.kernel = kernel.getNdArray().clone()
        elif isinstance(kernel, torch.Tensor):
            self.kernel = kernel.clone()
        
        if model.ndim != self.kernel.ndim:
            raise ValueError("Domain and kernel number of dimensions mismatch")

        self.kernel_size = tuple(self.kernel.shape)
        self.pad_size = tuple([k // 2 for k in self.kernel_size])

        if model.ndim == 1:
            conv = torch.nn.functional.conv_transpose1d
            corr = torch.nn.functional.conv1d
        elif model.ndim == 2:
            conv = torch.nn.functional.conv_transpose2d
            corr = torch.nn.functional.conv2d
        elif model.ndim == 3:
            conv = torch.nn.functional.conv_transpose3d
            corr = torch.nn.functional.conv3d
        else:
            raise ValueError
        
        # torch.nn functions require batch and channel dimensions
        self.conv = lambda x: conv(x.unsqueeze(0).unsqueeze(0),
                                   self.kernel.unsqueeze(0).unsqueeze(0),
                                   padding=self.pad_size).flatten(end_dim=2)

        self.corr = lambda x: corr(x.unsqueeze(0).unsqueeze(0),
                                   self.kernel.unsqueeze(0).unsqueeze(0),
                                   padding=self.pad_size).flatten(end_dim=2)
        
        super(ConvND, self).__init__(model, model)
    
    def __str__(self):
        return " ConvND "
    
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


def ZeroPad(model, pad):
    if isinstance(model, VectorTorch):
        return _ZeroPadIC(model, pad)
    elif isinstance(model, superVector):
        # TODO add the possibility to have different padding for each sub-vector
        return Dstack([_ZeroPadIC(v, pad) for v in model.vecs])
    else:
        raise ValueError("ERROR! Provided domain has to be either vector or superVector")


def _pad_vectorIC(vec, pad):
    if not isinstance(vec, VectorTorch):
        raise ValueError("ERROR! Provided vector must be a VectorTorch")
    assert len(vec.shape) == len(pad), "Dimensions of vector and padding mismatch!"
    
    vec_new_shape = tuple(np.asarray(vec.shape) + [sum(pad[_]) for _ in range(len(pad))])
    return VectorTorch(torch.empty(vec_new_shape).type(vec.getNdArray().dtype))


class _ZeroPadIC(Operator):
    
    def __init__(self, model, pad):
        """ Zero Pad operator.

        To pad 2 values to each side of the first dim, and 3 values to each side of the second dim, use:
            pad=((2,2), (3,3))
        :param model: vectorIC class
        :param pad: scalar or sequence of scalars
            Number of samples to pad in each dimension.
            If a single scalar is provided, it is assigned to every dimension.
        """
        if isinstance(model, VectorTorch):
            self.dims = model.shape
            pad = [(pad, pad)] * len(self.dims) if isinstance(pad, (int, float)) else list(pad)
            if (np.array(pad) < 0).any():
                raise ValueError('Padding must be positive or zero')
            self.pad = pad
            super(_ZeroPadIC, self).__init__(model, _pad_vectorIC(model, self.pad))
    
    def __str__(self):
        return "ZeroPad "
    
    def forward(self, add, model, data):
        """Zero padding"""
        self.checkDomainRange(model, data)
        if add:
            temp = data.clone()
        y = np.pad(model.arr, self.pad, mode='constant')
        data.arr = y
        if add:
            data.scaleAdd(temp, 1., 1.)
        return
    
    def adjoint(self, add, model, data):
        """Extract non-zero subsequence"""
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


if __name__ == "__main__":
    import occamypy
    import torch
    import matplotlib.pyplot as plt
    
    x = occamypy.VectorTorch((25, 25)).set(0.)
    x[12, 12] = 1.
    # plt.imshow(x.getNdArray()), plt.show()
    #
    # G = occamypy.torch.GaussianFilter(x, 2.)
    # plt.plot(G.kernel), plt.show()
    #
    # y = G*x
    # plt.imshow(y.getNdArray()), plt.show()
    
    g = occamypy.torch.operator.signal.gaussian_kernel(2., 21)
    g = torch.outer(g,g)
    plt.imshow(g), plt.show()

    C = occamypy.torch.ConvND(x, g)
    C.dotTest(True)
    y = C * x
    # plt.imshow(y.getNdArray()), plt.show()
