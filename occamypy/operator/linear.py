import numpy as np
from .base import Operator


class Zero(Operator):
    """Zero matrix operator; useful for Jacobian matrices that are zeros"""
    
    def __init__(self, domain, range):
        super(Zero, self).__init__(domain, range)
    
    def __str__(self):
        return "  Zero  "
    
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
        super(Identity, self).__init__(domain, domain)
    
    def __str__(self):
        return "Identity"
    
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
        super(Scaling, self).__init__(domain, domain)
        if not np.isscalar(scalar):
            raise ValueError('scalar has to be (indeed) a scalar variable')
        self.scalar = scalar
    
    def __str__(self):
        return "Scaling "
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        data.scaleAdd(model, 1. if add else 0., self.scalar)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        model.scaleAdd(data, 1. if add else 0., self.scalar)


class Diagonal(Operator):
    """Diagonal operator for performing element-wise multiplication"""
    
    def __init__(self, diag):
        # if not isinstance(diag, vector):
        #     raise TypeError('diag has to be a vector')
        super(Diagonal, self).__init__(diag, diag)
        self.diag = diag
    
    def __str__(self):
        return "Diagonal"
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        data.scaleAdd(model, 1. if add else 0.)
        data.multiply(self.diag)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        model.scaleAdd(data, 1. if add else 0.)
        model.multiply(self.diag)
