from __future__ import division, print_function, absolute_import
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve, correlate

from occamypy import Operator, Vstack, Dstack, Vector, superVector
from occamypy.vector import sep
from . import VectorCupy


class MatrixOp(Operator):
    """Operator built upon a matrix"""
    
    def __init__(self, matrix, domain, range, outcore=False):
        """Class constructor
        :param matrix   : matrix to use
        :param domain   : domain vector
        :param range    : range vector
        :param outcore  : use outcore sep operators
        """
        if not isinstance(domain, Vector):
            raise TypeError("ERROR! Domain vector not a vector object")
        if not isinstance(range, Vector):
            raise TypeError("ERROR! Range vector not a vector object")
        # Setting domain and range of operator and matrix to use during application of the operator
        self.setDomainRange(domain, range)
        if not (isinstance(matrix, np.ndarray) or isinstance(matrix, cp.ndarray)):
            raise ValueError("ERROR! matrix has to be a numpy/cupy ndarray")
        self.M = matrix
        self.outcore = outcore
        self.arr_module = cp.get_array_module(matrix)
    
    def __str__(self):
        return "MatrixOp"
    
    def forward(self, add, model, data):
        """d = A * m"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        model_arr = model.getNdArray()
        if self.outcore:
            [data_arr, data_axis] = sep.read_file(data.vecfile)
            data_arr += self.arr_module.matmul(self.M, model_arr.ravel()).reshape(data_arr.shape)
            sep.write_file(data.vecfile, data_arr, data_axis)
        else:
            data_arr = data.getNdArray()
            data_arr += self.arr_module.matmul(self.M, model_arr.ravel()).reshape(data_arr.shape)
        return
    
    def adjoint(self, add, model, data):
        """m = A' * d"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        data_arr = data.getNdArray()
        if self.outcore:
            [model_arr, model_axis] = sep.read_file(model.vecfile)
            model_arr += self.arr_module.matmul(self.M.T.conj(), data_arr.ravel()).reshape(model_arr.shape)
            sep.write_file(model.vecfile, model_arr, model_axis)
        else:
            model_arr = model.getNdArray()
            model_arr += self.arr_module.matmul(self.M.T.conj(), data_arr.ravel()).reshape(model_arr.shape)
        return
    
    def getNdArray(self):
        return self.arr_module.array(self.M)


class FirstDerivative(Operator):
    def __init__(self, model, sampling=1., axis=0, kind='centered'):
        r"""
        First Derivative with a stencil
            1) 2nd order centered:

            .. math::
                y[i] = 0.5 (x[i+1] - x[i-1]) / dx

            2) 1st order forward:

            .. math::
                y[i] = (x[i+1] - x[i]) / dx

            1) 1st order backward:

            .. math::
                y[i] = 0.5 (x[i] - x[i-1]) / dx

        :param model    : vector class; domain vector
        :param sampling : scalar; sampling step [1.]
        :param axis     : int; axis along which to compute the derivative [0]
        :param kind     : str; derivative kind (centered, forward, backward)
        """
        self.sampling = sampling
        self.dims = model.getNdArray().shape
        self.axis = axis if axis >= 0 else len(self.dims) + axis
        self.kind = kind
        
        if self.kind == 'centered':
            self.forward = self._forwardC
            self.adjoint = self._adjointC
        elif self.kind == 'backward':
            self.forward = self._forwardB
            self.adjoint = self._adjointB
        elif self.kind == 'forward':
            self.forward = self._forwardF
            self.adjoint = self._adjointF
        else:
            raise ValueError("Derivative kind must be centered, forward or backward")
        
        super(FirstDerivative, self).__init__(model, model)
    
    def __str__(self):
        return "1stDer_%d" % self.axis
    
    def _forwardF(self, add, model, data):
        """Forward operator for the 1st order forward stencil"""
        self.checkDomainRange(model, data)
        if add:
            data_tmp = data.clone()
        data.zero()
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = cp.get_array_module(x).swapaxes(x, self.axis, 0)
        y = cp.get_array_module(x).zeros(x.shape)
        
        y[:-1] = (x[1:] - x[:-1]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = cp.get_array_module(x).swapaxes(y, 0, self.axis)
        data.getNdArray()[:] = y
        if add:
            data.scaleAdd(data_tmp)
        return
    
    def _adjointF(self, add, model, data):
        """Adjoint operator for the 1st order forward stencil"""
        self.checkDomainRange(model, data)
        if add:
            model_temp = model.clone()
        model.zero()
        # Getting Ndarrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = cp.get_array_module(y).swapaxes(y, self.axis, 0)
        x = cp.get_array_module(y).zeros(y.shape)
        
        x[:-1] -= y[:-1] / self.sampling
        x[1:] += y[:-1] / self.sampling
        
        if self.axis > 0:
            x = cp.get_array_module(x).swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(model_temp)
        return
    
    def _forwardC(self, add, model, data):
        """Forward operator for the 2nd order centered stencil"""
        self.checkDomainRange(model, data)
        if add:
            data_tmp = data.clone()
        data.zero()
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = cp.get_array_module(x).swapaxes(x, self.axis, 0)
        y = cp.get_array_module(x).zeros(x.shape)
        
        y[1:-1] = (.5 * x[2:] - 0.5 * x[:-2]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = cp.get_array_module(x).swapaxes(y, 0, self.axis)
        data.getNdArray()[:] = y
        if add:
            data.scaleAdd(data_tmp)
        return
    
    def _adjointC(self, add, model, data):
        """Adjoint operator for the 2nd order centered stencil"""
        self.checkDomainRange(model, data)
        if add:
            model_temp = model.clone()
        model.zero()
        # Getting Ndarrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = cp.get_array_module(y).swapaxes(y, self.axis, 0)
        x = cp.get_array_module(y).zeros(y.shape)
        
        x[:-2] -= 0.5 * y[1:-1] / self.sampling
        x[2:] += 0.5 * y[1:-1] / self.sampling
        
        if self.axis > 0:
            x = cp.get_array_module(x).swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(model_temp)
        return
    
    def _forwardB(self, add, model, data):
        """Forward operator for the 1st order backward stencil"""
        self.checkDomainRange(model, data)
        if add:
            data_tmp = data.clone()
        data.zero()
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = cp.get_array_module(x).swapaxes(x, self.axis, 0)
        y = cp.get_array_module(x).zeros(x.shape)
        
        y[1:] = (x[1:] - x[:-1]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = cp.get_array_module(y).swapaxes(y, 0, self.axis)
        data.getNdArray()[:] = y
        if add:
            data.scaleAdd(data_tmp)
        return
    
    def _adjointB(self, add, model, data):
        """Adjoint operator for the 1st order backward stencil"""
        self.checkDomainRange(model, data)
        if add:
            model_temp = model.clone()
        model.zero()
        # Getting Ndarrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = cp.get_array_module(y).swapaxes(y, self.axis, 0)
        x = cp.get_array_module(y).zeros(y.shape)
        
        x[:-1] -= y[1:] / self.sampling
        x[1:] += y[1:] / self.sampling
        
        if self.axis > 0:
            x = cp.get_array_module(x).swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(model_temp)
        return


class SecondDerivative(Operator):
    def __init__(self, model, sampling=1., axis=0):
        r"""
        Compute 2nd order second derivative

        .. math::
            y[i] = (x[i+1] - 2x[i] + x[i-1]) / dx^2

        :param model    : vector class; domain vector
        :param sampling : scalar; sampling step [1.]
        :param axis     : int; axis along which to compute the derivative [0]
        """
        self.sampling = sampling
        self.data_tmp = model.clone().zero()
        self.dims = model.getNdArray().shape
        self.axis = axis if axis >= 0 else len(self.dims) + axis
        super(SecondDerivative, self).__init__(model, model)
    
    def __str__(self):
        return "2ndDer_%d" % self.axis
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if add:
            self.data_tmp.copy(data)
        data.zero()
        
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = cp.get_array_module(x).swapaxes(x, self.axis, 0)
        y = cp.get_array_module(x).zeros_like(x)
        
        y[1:-1] = (x[0:-2] - 2 * x[1:-1] + x[2:]) / self.sampling ** 2
        
        if self.axis > 0:  # reset axis order
            y = cp.get_array_module(y).swapaxes(y, 0, self.axis)
        data.getNdArray()[:] = y
        if add:
            data.scaleAdd(self.data_tmp)
        return
    
    def adjoint(self, add, model, data):
        """Adjoint operator"""
        self.checkDomainRange(model, data)
        if add:
            self.data_tmp.copy(model)
        model.zero()
        
        # Getting numpy arrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = cp.get_array_module(y).swapaxes(y, self.axis, 0)
        x = cp.get_array_module(y).zeros_like(y)
        
        x[0:-2] += y[1:-1] / self.sampling ** 2
        x[1:-1] -= 2 * y[1:-1] / self.sampling ** 2
        x[2:] += y[1:-1] / self.sampling ** 2
        
        if self.axis > 0:
            x = cp.get_array_module(x).swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(self.data_tmp)
        return


class Gradient(Operator):
    def __init__(self, model, sampling=None):
        r"""
        N-Dimensional Gradient operator.

        :param model    : vector class; domain vector
        :param sampling : tuple; sampling step [1]
        """
        self.dims = model.getNdArray().shape
        self.sampling = sampling if sampling is not None else tuple([1] * len(self.dims))
        
        assert len(self.sampling) != 0, "There is something wrong with the dimensions"
        
        self.op = Vstack([FirstDerivative(model, sampling=self.sampling[d], axis=d)
                               for d in range(len(self.dims))])
        
        super(Gradient, self).__init__(domain=self.op.domain, range=self.op.range)
    
    def __str__(self):
        return "Gradient"
    
    def forward(self, add, model, data):
        return self.op.forward(add, model, data)
    
    def adjoint(self, add, model, data):
        return self.op.adjoint(add, model, data)
    
    def merge_directions(self, add, model, data, iso=True):
        """
        Merge the different directional contributes, using the L2 norm (iso=True) or the simple sum (iso=False)
        """
        self.range.checkSame(model)
        if not add:
            data.zero()
        
        if iso:
            for v in model.vecs:
                data.scaleAdd(v.clone().pow(2), 1., 1.)
                data.pow(.5)
        else:
            for v in model.vecs:
                data.scaleAdd(v, 1., 1.)


class Laplacian(Operator):
    def __init__(self, model, axis=None, weights=None, sampling=None):
        r"""
        Laplacian operator.
        The input parameters are tailored for >2D, but it works also for 1D.

        :param model    : vector class; domain vector
        :param axis     : tuple; axis along which to compute the derivative [all]
        :param weights  : tuple; scalar weights for the axis [1 for each model axis]
        :param sampling : tuple; sampling step [1 for each model axis]
        """
        self.dims = model.getNdArray().shape
        self.axis = axis if axis is not None else tuple(range(len(self.dims)))
        self.sampling = sampling if sampling is not None else tuple([1] * len(self.dims))
        self.weights = weights if weights is not None else tuple([1] * len(self.dims))
        
        assert len(self.axis) == len(self.weights) == len(self.sampling) != 0, \
            "There is something wrong with the dimensions"
        
        self.data_tmp = model.clone().zero()
        
        self.op = self.weights[0] * SecondDerivative(model, sampling=self.sampling[0], axis=self.axis[0])
        for d in range(1, len(self.axis)):
            self.op += self.weights[d] * SecondDerivative(model, sampling=self.sampling[d], axis=self.axis[d])
        super(Laplacian, self).__init__(model, model)
    
    def __str__(self):
        return "Laplace "
    
    def forward(self, add, model, data):
        return self.op.forward(add, model, data)
    
    def adjoint(self, add, model, data):
        return self.op.adjoint(add, model, data)


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
        self.kernel = cp.pad(self.kernel, pad_width, mode='constant')
        
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
        y = cp.pad(model.getNdArray(), self.pad, mode='constant')
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



