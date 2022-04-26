import numpy as np

from occamypy.operator.base import Operator

try:
    import pylops
except ImportError:
    raise UserWarning("PyLops is not installed. To use this feature please run: pip install pylops")

__all__ = [
    "ToPylops",
    "FromPylops",
]


class FromPylops(Operator):
    """Cast a pylops.LinearOperator to occamypy.Operator"""

    def __init__(self, domain, range, op):
        """
        FromPylops constructor

        Args:
            domain: domain vector
            range: range vector
            op: pylops LinearOperator
        """
        if not isinstance(op, pylops.LinearOperator):
            raise TypeError("op has to be a pylops.LinearOperator")
        if op.shape[0] != range.size:
            raise ValueError("Range and operator rows mismatch")
        if op.shape[1] != domain.size:
            raise ValueError("Domain and operator columns mismatch")
        
        self.op = op
        
        super(FromPylops, self).__init__(domain, range, name=op.__str__().replace("<", "").replace(">", ""))
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        x = model.getNdArray().ravel()
        y = self.op.matvec(x)
        data.getNdArray()[:] += y.reshape(data.shape)
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        y = data.getNdArray().ravel()
        x = self.op.rmatvec(y)
        model[:] += x.reshape(model.shape)
        return


class ToPylops(pylops.LinearOperator):
    """Cast an numpy-based occamypy.Operator to pylops.LinearOperator"""

    def __init__(self, op: Operator):
        """
        ToPylops constructor

        Args:
            op: occamypy.Operator
        """
        super(ToPylops, self).__init__(explicit=False)
        self.shape = (op.range.size, op.domain.size)
        self.dtype = op.domain.getNdArray().dtype
        
        if not isinstance(op, Operator):
            raise TypeError("op has to be an Operator")
        self.op = op
        
        # these are just temporary vectors, used by forward and adjoint computations
        self.domain = op.domain.clone()
        self.range = op.range.clone()
        self._name = op.name
    
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x_ = self.domain.clone()
        x_[:] = x.reshape(self.domain.shape).astype(self.dtype)
        y_ = self.range.clone()
        self.op.forward(False, x_, y_)
        return y_.getNdArray().ravel()
    
    def _rmatvec(self, y: np.ndarray) -> np.ndarray:
        y_ = self.range.clone()
        y_[:] = y.reshape(self.range.shape).astype(self.dtype)
        x_ = self.domain.clone()
        self.op.adjoint(False, x_, y_)
        return x_.getNdArray().ravel()
