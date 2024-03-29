from occamypy.utils import get_backend
from occamypy.operator.base import Operator, Vstack


class FirstDerivative(Operator):
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
    """
    def __init__(self, domain, sampling=1., axis=0, stencil='centered'):
        """
        FirstDerivative costructor
        
        Args:
            domain: domain vector
            sampling: sampling step along the differentiation axis
            axis: axis along which to compute the derivative [0]
            stencil: derivative kind (centered, forward, backward)
        """
        self.sampling = sampling
        self.dims = domain.getNdArray().shape
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
        
        self.backend = get_backend(domain)

        super(FirstDerivative, self).__init__(domain, domain, name="Der1")
    
    def _forwardF(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = self.backend.swapaxes(x, self.axis, 0)
        y = self.backend.zeros_like(x)
        
        y[:-1] = (x[1:] - x[:-1]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = self.backend.swapaxes(y, 0, self.axis)
        data[:] += y
        return
    
    def _adjointF(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        # Getting Ndarrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = self.backend.swapaxes(y, self.axis, 0)
        x = self.backend.zeros_like(y)
        
        x[:-1] -= y[:-1] / self.sampling
        x[1:] += y[:-1] / self.sampling
        
        if self.axis > 0:
            x = self.backend.swapaxes(x, 0, self.axis)
        model[:] += x
        return
    
    def _forwardC(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = self.backend.swapaxes(x, self.axis, 0)
        y = self.backend.zeros_like(x)
        
        y[1:-1] = (.5 * x[2:] - 0.5 * x[:-2]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = self.backend.swapaxes(y, 0, self.axis)
        data[:] += y
        return
    
    def _adjointC(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        # Getting Ndarrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = self.backend.swapaxes(y, self.axis, 0)
        x = self.backend.zeros_like(y)
        
        x[:-2] -= 0.5 * y[1:-1] / self.sampling
        x[2:] += 0.5 * y[1:-1] / self.sampling
        
        if self.axis > 0:
            x = self.backend.swapaxes(x, 0, self.axis)
        model[:] += x
        return
    
    def _forwardB(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = self.backend.swapaxes(x, self.axis, 0)
        y = self.backend.zeros_like(x)
        
        y[1:] = (x[1:] - x[:-1]) / self.sampling
        if self.axis > 0:  # reset axis order
            y = self.backend.swapaxes(y, 0, self.axis)
        data[:] += y
        return
    
    def _adjointB(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        # Getting Ndarrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = self.backend.swapaxes(y, self.axis, 0)
        x = self.backend.zeros_like(y)
        
        x[:-1] -= y[1:] / self.sampling
        x[1:] += y[1:] / self.sampling
        
        if self.axis > 0:
            x = self.backend.swapaxes(x, 0, self.axis)
        model[:] += x
        return


class SecondDerivative(Operator):
    r"""
    Compute 2nd order second derivative

    .. math::
        y[i] = (x[i+1] - 2x[i] + x[i-1]) / dx^2
    """
    def __init__(self, domain, sampling=1., axis=0):
        """
        SecondDerivative constructor

        Args:
            domain: domain vector
            sampling: sampling step along the differentiation axis
            axis: axis along which to compute the derivative
        """
        self.sampling = sampling
        self.data_tmp = domain.clone().zero()
        self.dims = domain.getNdArray().shape
        self.axis = axis if axis >= 0 else len(self.dims) + axis

        self.backend = get_backend(domain)
        
        super(SecondDerivative, self).__init__(domain, domain, name="Der2")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        
        # Getting Ndarrays
        x = model.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            x = self.backend.swapaxes(x, self.axis, 0)
        y = self.backend.zeros_like(x)
        
        y[1:-1] = (x[0:-2] - 2 * x[1:-1] + x[2:]) / self.sampling ** 2
        
        if self.axis > 0:  # reset axis order
            y = self.backend.swapaxes(y, 0, self.axis)
        data[:] += y
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        
        # Getting numpy arrays
        y = data.clone().getNdArray()
        if self.axis > 0:  # need to bring the dim. to derive to first dim
            y = self.backend.swapaxes(y, self.axis, 0)
        x = self.backend.zeros_like(y)
        
        x[0:-2] += (y[1:-1]) / self.sampling ** 2
        x[1:-1] -= (2 * y[1:-1]) / self.sampling ** 2
        x[2:] += (y[1:-1]) / self.sampling ** 2
        
        if self.axis > 0:
            x = self.backend.swapaxes(x, 0, self.axis)
        model[:] += x
        return


class Gradient(Operator):
    """N-Dimensional Gradient operator"""

    def __init__(self, domain, sampling=None, stencil=None):
        """
        Gradient constructor

        Args:
            domain: domain vector
            sampling: sampling steps
            stencil: stencil kind for each direction
        """
        self.dims = domain.getNdArray().shape
        self.sampling = sampling if sampling is not None else tuple([1] * len(self.dims))
        
        if stencil is None:
            self.stencil = tuple(['centered'] * len(self.dims))
        elif isinstance(stencil, str):
            self.stencil = tuple([stencil] * len(self.dims))
        elif isinstance(stencil, tuple) or isinstance(stencil, list):
            self.stencil = stencil
        if len(self.sampling) == 0:
            raise ValueError("Provide at least one sampling item")
        if len(self.stencil) == 0:
            raise ValueError("Provide at least one stencil item")
        if len(self.sampling) != len(self.stencil):
            raise ValueError("There is something wrong with the dimensions")
        
        self.op = Vstack([FirstDerivative(domain, sampling=self.sampling[d], axis=d)
                          for d in range(len(self.dims))])
        super(Gradient, self).__init__(domain=self.op.domain, range=self.op.range, name="Gradient")
    
    def __str__(self):
        return "Gradient"
    
    def forward(self, add, model, data):
        return self.op.forward(add, model, data)
    
    def adjoint(self, add, model, data):
        return self.op.adjoint(add, model, data)
    
    def merge_directions(self, grad_vector, iso=True):
        """
        Merge the different directional contributes, using the L2 norm (iso=True) or the simple sum (iso=False)
        """
        self.range.checkSame(grad_vector)
        
        data = self.domain.clone()
        
        if iso:
            for v in grad_vector.vecs:
                data.scaleAdd(v.clone().pow(2), 1., 1.)
                data.pow(.5)
        else:
            for v in grad_vector.vecs:
                data.scaleAdd(v, 1., 1.)
        
        return data


class Laplacian(Operator):
    """
    Laplacian operator.

    Notes:
        The input parameters are tailored for >2D, but it works also for 1D.
    """
    
    def __init__(self, domain, axis=None, weights=None, sampling=None):
        """
        Laplacian constructor

        Args:
            domain: domain vector
            axis: axes along which to compute the derivative
            weights: scalar weights for each axis
            sampling: sampling steps for each axis
        """
        self.dims = domain.getNdArray().shape
        self.axis = axis if axis is not None else tuple(range(len(self.dims)))
        self.sampling = sampling if sampling is not None else tuple([1] * len(self.dims))
        self.weights = weights if weights is not None else tuple([1] * len(self.dims))
        
        if not (len(self.axis) == len(self.weights) == len(self.sampling)):
            raise ValueError("There is something wrong with the dimensions")
        
        self.data_tmp = domain.clone().zero()
        
        self.op = self.weights[0] * SecondDerivative(domain, sampling=self.sampling[0], axis=self.axis[0])
        for d in range(1, len(self.axis)):
            self.op += self.weights[d] * SecondDerivative(domain, sampling=self.sampling[d], axis=self.axis[d])
        super(Laplacian, self).__init__(domain, domain, name="Laplacian")
    
    def forward(self, add, model, data):
        return self.op.forward(add, model, data)
    
    def adjoint(self, add, model, data):
        return self.op.adjoint(add, model, data)
