from typing import Union

import numpy as np

from occamypy.operator.base import Operator
from occamypy.vector.base import Vector


class Zero(Operator):
    """Zero matrix operator; useful for Jacobian matrices that are zeros"""
    
    def __init__(self, domain: Vector, range: Vector):
        """
        Zero constructor
        
        Args:
            domain: domain vector
            range: range vector
        """
        super(Zero, self).__init__(domain=domain, range=range)
        self.name = "Zero"
    
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
    
    def __init__(self, domain: Vector):
        """
        Identity constructor
        
        Args:
            domain: domain vector
        """
        super(Identity, self).__init__(domain=domain, range=domain)
        self.name = "Identity"
    
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
    
    def __init__(self, domain: Vector, scalar: Union[float, int]):
        """
        Scaling constructor
        
        Args:
            domain: domain vector
            scalar: scaling coefficient
        """
        super(Scaling, self).__init__(domain=domain, range=domain)
        if not np.isscalar(scalar):
            raise ValueError('scalar has to be (indeed) a scalar variable')
        self.scalar = scalar
        self.name = "Scaling"
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        data.scaleAdd(model, 1. if add else 0., self.scalar)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        model.scaleAdd(data, 1. if add else 0., self.scalar)


class Diagonal(Operator):
    """Diagonal operator for performing element-wise multiplication"""
    
    def __init__(self, diag: Vector):
        """
        Diagonal constructor
        
        Args:
            diag: vector to be stored on the diagonal
        """
        # if not isinstance(diag, vector):
        #     raise TypeError('diag has to be a vector')
        super(Diagonal, self).__init__(domain=diag, range=diag)
        self.diag = diag
        self.name = "Diagonal"
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        data.scaleAdd(model, 1. if add else 0.)
        data.multiply(self.diag)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        model.scaleAdd(data, 1. if add else 0.)
        model.multiply(self.diag)
