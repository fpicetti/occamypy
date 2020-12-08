import numpy as np
from occamypy import Operator, Vstack


class FirstDerivative(Operator):
    def __init__(self, model, sampling=1., axis=0, stencil='centered'):
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
        :param stencil  : str; derivative kind (centered, forward, backward)
        """
        self.sampling = sampling
        self.dims = model.getNdArray().shape
        self.axis = axis if axis >= 0 else len(self.dims) + axis
        self.stencil = stencil
        
        if self.stencil == 'centered':
            self.forward = self._forwardC
            self.adjoint = self._adjointC
        elif self.stencil == 'backward':
            self.forward = self._forwardB
            self.adjoint = self._adjointB
        elif self.stencil == 'forward':
            self.forward = self._forwardF
            self.adjoint = self._adjointF
        else:
            raise ValueError("Derivative stencil must be centered, forward or backward")
        
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
            x = np.swapaxes(x, self.axis, 0)
        y = np.zeros(x.shape)
        
        y[:-1] = (x[1:] - x[:-1]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = np.swapaxes(y, 0, self.axis)
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
            y = np.swapaxes(y, self.axis, 0)
        x = np.zeros(y.shape)
        
        x[:-1] -= y[:-1] / self.sampling
        x[1:] += y[:-1] / self.sampling
        
        if self.axis > 0:
            x = np.swapaxes(x, 0, self.axis)
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
            x = np.swapaxes(x, self.axis, 0)
        y = np.zeros(x.shape)
        
        y[1:-1] = (.5 * x[2:] - 0.5 * x[:-2]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = np.swapaxes(y, 0, self.axis)
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
            y = np.swapaxes(y, self.axis, 0)
        x = np.zeros(y.shape)
        
        x[:-2] -= 0.5 * y[1:-1] / self.sampling
        x[2:] += 0.5 * y[1:-1] / self.sampling
        
        if self.axis > 0:
            x = np.swapaxes(x, 0, self.axis)
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
            x = np.swapaxes(x, self.axis, 0)
        y = np.zeros(x.shape)
        
        y[1:] = (x[1:] - x[:-1]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = np.swapaxes(y, 0, self.axis)
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
            y = np.swapaxes(y, self.axis, 0)
        x = np.zeros(y.shape)
        
        x[:-1] -= y[1:] / self.sampling
        x[1:] += y[1:] / self.sampling
        
        if self.axis > 0:
            x = np.swapaxes(x, 0, self.axis)
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
            x = np.swapaxes(x, self.axis, 0)
        y = np.zeros(x.shape)
        
        y[1:-1] = (x[0:-2] - 2 * x[1:-1] + x[2:]) / self.sampling ** 2
        
        if self.axis > 0:  # reset axis order
            y = np.swapaxes(y, 0, self.axis)
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
            y = np.swapaxes(y, self.axis, 0)
        x = np.zeros(y.shape)
        
        x[0:-2] += (y[1:-1]) / self.sampling ** 2
        x[1:-1] -= (2 * y[1:-1]) / self.sampling ** 2
        x[2:] += (y[1:-1]) / self.sampling ** 2
        
        if self.axis > 0:
            x = np.swapaxes(x, 0, self.axis)
        model.getNdArray()[:] = x
        if add:
            model.scaleAdd(self.data_tmp)
        return


class Gradient(Operator):
    def __init__(self, model, sampling=None, stencil=None):
        r"""
        N-Dimensional Gradient operator

        :param model    : vector class; domain vector
        :param sampling : tuple; sampling step [1]
        :param stencil  : str or list of str; stencil kind for each direction ['centered']
        """
        self.dims = model.getNdArray().shape
        self.sampling = sampling if sampling is not None else tuple([1] * len(self.dims))
        
        if stencil is None:
            self.stencil = tuple(['centered'] * len(self.dims))
        elif isinstance(stencil, str):
            self.stencil = tuple([stencil] * len(self.dims))
        elif isinstance(stencil, tuple) or isinstance(stencil, list):
            self.stencil = stencil
        assert len(self.sampling) == len(self.stencil) != 0, "There is something wrong with the dimensions"
        
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
