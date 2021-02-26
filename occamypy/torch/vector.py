from math import sqrt

import torch
from GPUtil import getGPUs, getFirstAvailable
from numpy import ndarray

from occamypy import Vector


class VectorTorch(Vector):
    """In-core python vector class"""
    
    def __init__(self, in_content, device=None):
        """
        VectorTorch constructor: arr = torch.Tensor
        :param in_content: numpy ndarray or torch.Tensor or InCore Vector
        :param device: int - GPU id (None for CPU, -1 for most available memory)
        """
        if isinstance(in_content, Vector):
            try:
                self.arr = torch.from_numpy(in_content.getNdArray()).contiguous()
                self.ax_info = None
            except:
                raise UserWarning("Torch cannot handle the input array type")
        elif isinstance(in_content, ndarray):  # Numpy array passed to constructor
            self.arr = torch.from_numpy(in_content).contiguous()
            self.ax_info = None
        elif isinstance(in_content, torch.Tensor):  # Tensor passed to constructor
            self.arr = in_content.contiguous()
            self.ax_info = None
        elif isinstance(in_content, tuple):  # Tuple size passed to constructor
            self.arr = torch.empty(in_content)
            self.ax_info = None
        else:  # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")

        self.setDevice(device)
        
        super(VectorTorch, self).__init__()

        self.shape = tuple(self.arr.shape)  # Number of elements per axis (tuple)
        self.ndim = self.arr.ndim           # Number of axes (integer)
        self.size = self.arr.numel()        # Total number of elements (integer)
    
    def _check_same_device(self, other):
        assert isinstance(self, VectorTorch)
        assert isinstance(other, VectorTorch)
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

    def setDevice(self, devID=0):
        if devID is not None:  # Move to GPU
            if devID == -1:
                devID = getFirstAvailable(order='memory')[0]
            try:
                self.arr = self.arr.to(devID)
            except AttributeError:
                self.arr = self.arr.to('cpu')
                raise UserWarning("The selected device is not available, switched to CPU.")
        else:  # move to CPU
            if self.device is torch.device('cpu'):
                pass
            else:
                self.arr = self.arr.to('cpu')

    def printDevice(self):
        if self.device is torch.device('cpu'):
            print('CPU')
        else:
            name = getGPUs()[self.device.index].name
            print('GPU %d - %s' % (self.device.index, name))
        
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return self.arr
    
    def norm(self, N=2):
        """Function to compute vector N-norm"""
        return torch.linalg.norm(self.getNdArray().flatten(), ord=N).item()
    
    def zero(self):
        """Function to zero out a vector"""
        self.set(0)
        return self
    
    def max(self):
        """Function to obtain maximum value in the vector"""
        return self.getNdArray().max().item()
    
    def min(self):
        """Function to obtain minimum value in the vector"""
        return self.getNdArray().min().item()
    
    def set(self, val: float or int):
        """Function to set all values in the vector"""
        self.getNdArray().fill_(val)
        return self
    
    def scale(self, sc: float or int):
        """Function to scale a vector"""
        self.getNdArray()[:] *= sc
        return self
    
    def addbias(self, bias: float or int):
        self.getNdArray()[:] += bias
        return self
    
    def rand(self, snr: float or int = 1.):
        """Fill vector with random number (~U[1,-1]) with a given SNR"""
        rms = torch.sqrt(torch.mean(torch.square(self.getNdArray())))
        amp_noise = 1.0
        if rms != 0.:
            amp_noise = sqrt(3. / snr) * rms  # sqrt(3*Power_signal/SNR)
        self.getNdArray()[:].uniform_(-1, 1)
        self.scale(amp_noise)
        return self
    
    def randn(self, snr=1.):
        """Fill vector with random number (~N[0,-1]) with a given SNR"""
        rms = torch.sqrt(torch.mean(torch.square(self.getNdArray())))
        amp_noise = 1.0
        if rms != 0.:
            amp_noise = sqrt(3. / snr) * rms  # sqrt(3*Power_signal/SNR)
        self.getNdArray()[:].normal_(0, 1)
        self.scale(amp_noise)
        return self
    
    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        # If self is a Space vector, it is empty and it has only the shape, size and ndim attributes
        if self.getNdArray().numel() == 0:  # this is the shape of tensor!
            vec_clone = VectorTorch(torch.zeros(self.shape).type(self.getNdArray().dtype))
        
        else:  # self is a data vector, just clone
            vec_clone = VectorTorch(self.getNdArray().clone())
        
        vec_clone.setDevice(self.device.index)
        return vec_clone
    
    def cloneSpace(self):
        """Function to clone vector space only (vector without actual vector array by using empty array of size 0)"""
        vec_space = VectorTorch(torch.empty(0).type(self.getNdArray().dtype))
        vec_space.setDevice(self.device.index)
        # Cloning space of input vector
        vec_space.ndim = self.ndim
        vec_space.shape = self.shape
        vec_space.size = self.size
        return vec_space
    
    def checkSame(self, other):
        """Function to check dimensionality of vectors"""
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
        """Compute element-wise power of the vector"""
        self.getNdArray()[:].pow_(power)
        return self
    
    def real(self):
        """Return the real part of the vector"""
        self.getNdArray()[:] = torch.real(self.getNdArray())
        return self
    
    def imag(self, ):
        """Return the imaginary part of the vector"""
        self.getNdArray()[:] = torch.imag(self.getNdArray())
        return self
    
    def copy(self, other):
        """Function to copy vector from input vector"""
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
        """Function to scale a vector"""
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
        """Function to compute dot product between two vectors"""
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
        """Function to multiply element-wise two vectors"""
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
        """Function to check if two vectors are identical using built-in hash function"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorTorch):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        isDiff = not torch.equal(self.getNdArray(), other.getNdArray())
        return isDiff
    
    def clipVector(self, low, high):
        """Function to bound vector values based on input vectors low and high"""
        if not isinstance(low, VectorTorch):
            raise TypeError("Provided input low vector not a %s!" % self.whoami)
        if not isinstance(high, VectorTorch):
            raise TypeError("Provided input high vector not a %s!" % self.whoami)
        self.getNdArray()[:] = torch.minimum(torch.maximum(low.getNdArray(), self.getNdArray()), high.getNdArray())
        return self
    
    def plot(self):
        return self.getNdArray().detach().cpu().numpy()
