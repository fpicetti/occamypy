from math import sqrt

import torch
from numpy import ndarray

from occamypy.vector.base import Vector
from occamypy.torch.back_utils import get_device, get_device_name


class VectorTorch(Vector):
    """
    Vector class based on torch.Tensor
    
    Notes:
       tensors are stored in C-contiguous memory
    """
    def __init__(self, in_content, device: int = None, *args, **kwargs):
        """
        VectorTorch constructor
        
        Args:
            in_content: Vector, np.ndarray, torch.Tensor or tuple
            device: computation device (None for CPU, -1 for least used GPU)
            *args: list of arguments for Vector construction
            **kwargs: dict of arguments for Vector construction
        """
        super(VectorTorch, self).__init__(*args, **kwargs)

        if isinstance(in_content, Vector):
            try:
                self.arr = torch.from_numpy(in_content.getNdArray()).contiguous()
                self.ax_info = in_content.ax_info
            except:
                raise UserWarning("Torch cannot handle the input array type")
        elif isinstance(in_content, ndarray):  # Numpy array passed to constructor
            self.arr = torch.from_numpy(in_content).contiguous()
        elif isinstance(in_content, torch.Tensor):  # Tensor passed to constructor
            self.arr = in_content.contiguous()
        elif isinstance(in_content, tuple):  # Tuple size passed to constructor
            self.arr = torch.zeros(in_content)
        else:  # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")

        self.setDevice(device)
        
        self.shape = tuple(self.arr.shape)  # Number of elements per axis (tuple)
        self.ndim = self.arr.ndim           # Number of axes (integer)
        self.size = self.arr.numel()        # Total number of elements (integer)
        
    def _check_same_device(self, other):
        if not isinstance(self, VectorTorch):
            raise TypeError("The self vector has to be a VectorTorch")
        if not isinstance(other, VectorTorch):
            raise TypeError("The other vector has to be a VectorTorch")
        answer = self.device == other.device
        if not answer:
            raise Warning('The two vectors live in different devices: %s - %s' % (self.device, other.device))
        return answer
    
    @property
    def device(self):
        try:
            return self.arr.device
        except AttributeError:
            return None
    
    def setDevice(self, devID):
        if isinstance(devID, int):
            self.arr = self.arr.to(get_device(devID))
        elif isinstance(devID, torch.device):
            self.arr = self.arr.to(devID)
        else:
            ValueError("Device type not understood")
    
    @ property
    def deviceName(self):
        return get_device_name(self.device.index)
        
    def getNdArray(self):
        return self.arr
    
    def norm(self, N=2):
        norm = torch.linalg.norm(self.getNdArray().flatten(), ord=N)
        return norm.item()
    
    def zero(self):
        self.set(0)
        return self
    
    def max(self):
        max = self.getNdArray().max()
        return max.item()
    
    def min(self):
        min = self.getNdArray().min()
        return min.item()
    
    def set(self, val: float or int):
        self.getNdArray().fill_(val)
        return self
    
    def scale(self, sc: float or int):
        self.getNdArray()[:] *= sc
        return self
    
    def addbias(self, bias: float or int):
        self.getNdArray()[:] += bias
        return self
    
    def rand(self, low: float = -1., high: float = 1.):
        self.arr.uniform_(low, high)
        return self
    
    def randn(self, mean: float = 0., std: float = 1.):
        self.arr.normal_(mean, std)
        return self
    
    def clone(self):
        # If self is a Space vector, it is empty and it has only the shape, size and ndim attributes
        if self.getNdArray().numel() == 0:  # this is the shape of tensor!
            vec_clone = VectorTorch(torch.zeros(self.shape).type(self.getNdArray().dtype), ax_info=self.ax_info.copy())
        
        else:  # self is a data vector, just clone
            vec_clone = VectorTorch(self.getNdArray().clone(), ax_info=self.ax_info.copy())
        
        vec_clone.setDevice(self.device.index)
        return vec_clone
    
    def cloneSpace(self):
        vec_space = self.__class__(torch.empty(0).type(self.getNdArray().dtype))
        vec_space.setDevice(self.device.index)
        vec_space.ax_info = self.ax_info
        # Cloning space of input vector
        vec_space.ndim = self.ndim
        vec_space.shape = self.shape
        vec_space.size = self.size
        return vec_space
    
    def checkSame(self, other):
        return self.shape == other.shape
    
    def abs(self):
        self.getNdArray().abs_()
        return self
    
    def sign(self):
        self.getNdArray().sign_()
        return self
    
    def reciprocal(self):
        self.getNdArray().reciprocal_()
        return self
    
    def maximum(self, other):
        if isinstance(other, (int, float)):
            # create a vector filled with the scalar value
            other = self.clone().set(other)
        
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        self.getNdArray()[:] = torch.maximum(self.getNdArray(), other.getNdArray())
        return self
    
    def conj(self):
        self.getNdArray()[:] = torch.conj(self.getNdArray())
        return self
    
    def transpose(self):
        other = VectorTorch(self.getNdArray().T)
        other.getNdArray()[:] = other.getNdArray()[:].contiguous()
        return other
    
    def pow(self, power):
        self.getNdArray()[:].pow_(power)
        return self
    
    def real(self):
        self.getNdArray()[:] = torch.real(self.getNdArray())
        return self
    
    def imag(self):
        self.getNdArray()[:] = torch.imag(self.getNdArray())
        return self
    
    def copy(self, other):
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorTorch):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        # Element-wise copy of the input array
        self.getNdArray()[:].copy_(other.getNdArray())
        return self
    
    def scaleAdd(self, other, sc1=1.0, sc2=1.0):
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorTorch):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        # Performing scaling and addition
        self.scale(sc1)
        self.getNdArray()[:] += sc2 * other.getNdArray()
        return self
    
    def dot(self, other):
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorTorch):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("Vector size mismatching: self = %d; other = %d" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        return torch.vdot(self.getNdArray().flatten(), other.getNdArray().flatten())
    
    def multiply(self, other):
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorTorch):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("Vector size mismatching: self = %s; other = %s" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        # Performing element-wise multiplication
        self.getNdArray()[:].multiply_(other.getNdArray())
        return self
    
    def isDifferent(self, other):
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorTorch):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        isDiff = not torch.equal(self.getNdArray(), other.getNdArray())
        return isDiff
    
    def clip(self, low, high):
        if not isinstance(low, VectorTorch):
            raise TypeError("Provided input low vector not a %s!" % self.whoami)
        if not isinstance(high, VectorTorch):
            raise TypeError("Provided input high vector not a %s!" % self.whoami)
        self.getNdArray()[:] = torch.minimum(torch.maximum(low.getNdArray(), self.getNdArray()), high.getNdArray())
        return self
    
    def plot(self):
        return self.getNdArray().detach().cpu().numpy()
