import torch
from occamypy import Operator
from occamypy.torch import VectorTorch


class Matrix(Operator):
    """Operator built upon a matrix"""
    
    def __init__(self, matrix, domain, range):
        """Class constructor
        :param matrix   : matrix to use
        :param domain   : domain vector
        :param range    : range vector
        """
        if not isinstance(domain, VectorTorch):
            raise TypeError("ERROR! Domain vector not a VectorTorch object")
        if not isinstance(range, VectorTorch):
            raise TypeError("ERROR! Range vector not a VectorTorch object")
        # Setting domain and range of operator and matrix to use during application of the operator
        super().__init__(domain, range)
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("ERROR! matrix has to be a torch Tensor")
        self.M = matrix
        assert matrix.device == domain.device == range.device, "ERROR! All the elements have to be in the same device"
    
    def __str__(self):
        return "MatrixOp"
    
    def forward(self, add, model, data):
        """d = A * m"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        data_arr += torch.matmul(self.M, model_arr.flatten()).reshape(data_arr.shape)
        return
    
    def adjoint(self, add, model, data):
        """m = A' * d"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        data_arr = data.getNdArray()
        model_arr = model.getNdArray()
        model_arr += torch.matmul(self.M.T.conj(), data_arr.flatten()).reshape(model_arr.shape)
        return
    
    def getNdArray(self):
        return self.M
