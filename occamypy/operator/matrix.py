from occamypy.vector.base import Vector
from occamypy.operator.base import Operator
from occamypy.utils.backend import get_backend, get_vector_type


class Matrix(Operator):
    """
    Linear Operator build upon an explicit matrix

    Attributes:
        matrix: Vector array that contains the matrix
    """
    def __init__(self, matrix: Vector, domain: Vector, range: Vector, outcore=False):
        """
        Matrix constructor

        Args:
            matrix: vector that contains the matrix
            domain: domain vector
            range: range vector
            outcore: whether to use out-of-core SEPlib operators
        """
        if not (type(domain) == type(range) == type(matrix)):
            raise TypeError("ERROR! Domain, Range and Matrix have to be the same vector type")
          
        if matrix.shape[1] != domain.size:
            raise ValueError
        if matrix.shape[0] != range.size:
            raise ValueError
        
        super(Matrix, self).__init__(domain=domain, range=range, name="Matrix")
        self.backend = get_backend(matrix)
        self.matrix_type = get_vector_type(matrix)
        
        self.matrix = matrix
        self.outcore = outcore
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        data[:] += self.backend.matmul(self.matrix[:], model[:].flatten()).reshape(data.shape)
        return
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        model[:] += self.backend.matmul(self.matrix.hermitian()[:], data[:].flatten()).reshape(model.shape)
        return
    
    def getNdArray(self):
        """Get the matrix vector"""
        return self.matrix.getNdArray()
