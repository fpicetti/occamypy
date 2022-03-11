from occamypy import Vector
from .base import Operator
from occamypy.utils import get_backend, get_vector_type


class Matrix(Operator):
    """Operator built upon a matrix"""
    
    def __init__(self, matrix: Vector, domain: Vector, range: Vector, outcore=False):
        """Class constructor
        :param matrix   : matrix to use
        :param domain   : domain vector
        :param range    : range vector
        :param outcore  : use outcore sep operators
        """
        if not (type(domain) == type(range) == type(matrix)):
            raise TypeError("ERROR! Domain, Range and Matrix have to be the same vector type")
          
        if matrix.shape[1] != domain.size:
            raise ValueError
        if matrix.shape[0] != range.size:
            raise ValueError
        
        super(Matrix, self).__init__(domain=domain, range=range)
        self.backend = get_backend(matrix)
        self.matrix_type = get_vector_type(matrix)
        
        self.matrix = matrix
        self.outcore = outcore
    
    def __str__(self):
        return "MatrixOp"
    
    def forward(self, add, model, data):
        """d = A * m"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        data[:] += self.backend.matmul(self.matrix[:], model[:].flatten()).reshape(data.shape)
        return
    
    def adjoint(self, add, model, data):
        """m = A' * d"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        model[:] += self.backend.matmul(self.matrix.hermitian()[:], data[:].flatten()).reshape(model.shape)
        return
    
    def getNdArray(self):
        return self.matrix.getNdArray()
