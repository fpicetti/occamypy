import numpy as np
from occamypy import Vector, Operator


class Matrix(Operator):
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
        super().__init__(domain, range)
        if not isinstance(matrix, np.ndarray):
            raise ValueError("ERROR! matrix has to be a numpy ndarray")
        self.M = matrix
        self.outcore = outcore
    
    def __str__(self):
        return "MatrixOp"
    
    def forward(self, add, model, data):
        """d = A * m"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        data_arr += np.matmul(self.M, model_arr.ravel()).reshape(data_arr.shape)
        return
    
    def adjoint(self, add, model, data):
        """m = A' * d"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        data_arr = data.getNdArray()
        model_arr = model.getNdArray()
        model_arr += np.matmul(self.M.T.conj(), data_arr.ravel()).reshape(model_arr.shape)
        return
    
    def getNdArray(self):
        return np.array(self.M)
