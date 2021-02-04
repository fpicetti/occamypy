from typing import Union, List
from itertools import accumulate
import numpy as np
import torch
from occamypy import superVector, Operator, Dstack
from ..vector import VectorTorch
from ..back_utils import set_backends

set_backends()


def _gaussian_kernel1d(sigma: float, order: int = 0, truncate: float = 4.) -> torch.Tensor:
    """
    Computes a 1-D Gaussian convolution kernel.
    """
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
        
        if model.device != self.kernel.device:
            raise ValueError("Domain and kernel has to live in the same device")
        
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
    

class GaussianFilter(ConvND):
    def __init__(self, model, sigma):
        """
        Gaussian smoothing operator using scipy smoothing:
        :param model : vector class;
            domain vector
        :param sigma : scalar or sequence of scalars;
            standard deviation along the model directions
        """
        self.sigma = [sigma] if isinstance(sigma, (float, int)) else sigma
        assert isinstance(self.sigma, (list, tuple))
        self.scaling = np.sqrt(np.prod(np.array(self.sigma) / np.pi))
        kernels = [_gaussian_kernel1d(s) for s in self.sigma]
        
        self.kernel = [*accumulate(kernels, lambda a, b: torch.outer(a.flatten(), b))][-1]
        self.kernel.reshape([k.numel() for k in kernels])
        super(GaussianFilter, self).__init__(model, self.kernel.to(model.device))
    
    def __str__(self):
        return "GausFilt"


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
    plt.imshow(x.getNdArray()), plt.show()
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
    plt.imshow(y.getNdArray()), plt.show()
