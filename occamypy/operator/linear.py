import numpy as np
from occamypy.operator.base import Operator


class Zero(Operator):
    """Zero matrix operator; useful for Jacobian matrices that are zeros"""
    
    def __init__(self, domain, range):
        """
        Zero constructor

        Args:
            domain: domain vector
            range: range vector
        """
        super(Zero, self).__init__(domain, range, name="Zero")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()


class Identity(Operator):
    """Identity operator"""
    
    def __init__(self, domain):
        """
        Identity constructor

        Args:
            domain: domain vector
        """
        super(Identity, self).__init__(domain, domain, name="Identity")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if add:
            data.scaleAdd(model)
        else:
            data.copy(model)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if add:
            model.scaleAdd(data)
        else:
            model.copy(data)


class Scaling(Operator):
    """scalar multiplication operator"""
    
    def __init__(self, domain, scalar):
        """
        Scaling constructor

        Args:
            domain: domain vector
            scalar: scaling coefficient
        """
        super(Scaling, self).__init__(domain, domain, name="Scaling")
        if not np.isscalar(scalar):
            raise ValueError('scalar has to be (indeed) a scalar variable')
        self.scalar = scalar
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        data.scaleAdd(model, 1. if add else 0., self.scalar)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        model.scaleAdd(data, 1. if add else 0., self.scalar)


class Diagonal(Operator):
    """Diagonal operator for performing element-wise multiplication"""
    
    def __init__(self, diag):
        """
        Diagonal constructor

        Args:
            diag: vector to be stored on the diagonal
        """
        super(Diagonal, self).__init__(diag, diag, name="Diagonal")
        self.diag = diag
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        data.scaleAdd(model, 1. if add else 0.)
        data.multiply(self.diag)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        model.scaleAdd(data, 1. if add else 0.)
        model.multiply(self.diag)
