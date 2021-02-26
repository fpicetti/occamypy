from copy import deepcopy
from sys import version_info

import cupy as cp
import numpy as np
from GPUtil import getGPUs, getFirstAvailable

from occamypy import Vector, VectorNumpy


class VectorCupy(Vector):

    def __init__(self, in_content, device=None):
        """
        VectorCupy constructor
        :param in_content: if a cupy vector, the object is build upon the same vector (i.e., same device).
                       Otherwise, the object is created in the selected device.
        :param device: int - GPU id (None for CPU, -1 for most available memory)
        
        This class stores array with C memory order (i.e., row-wise sorting)
        """
        
        if isinstance(in_content, cp.ndarray) or isinstance(in_content, np.ndarray):
            if cp.isfortran(in_content):
                raise TypeError('Input array not a C contiguous array!')
            self.arr = cp.array(in_content, copy=False)
            self.ax_info = None
        elif isinstance(in_content, tuple):  # Tuple size passed to constructor
            # self.arr = cp.zeros(tuple(reversed(in_content)))
            self.arr = cp.empty(in_content)
            self.ax_info = None
        elif isinstance(in_content, VectorNumpy):
            self.arr = in_content.getNdArray().copy()
            self.ax_info = in_content.ax_info
        else:  # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")
        
        self.setDevice(device)
        
        super(VectorCupy, self).__init__()
        self.shape = self.arr.shape   # Number of elements per axis (tuple)
        self.ndim = self.arr.ndim    # Number of axes (integer)
        self.size = self.arr.size     # Total number of elements (integer)

    def _check_same_device(self, other):
        assert isinstance(self, VectorCupy)
        assert isinstance(other, VectorCupy)
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
                with cp.cuda.Device(devID):
                    self.arr = cp.asarray(self.arr)
            except AttributeError:
                self.arr.device = None
        else:  # move to CPU
            if self.device is None:  # already on CPU
                pass
            else:
                self.arr = cp.asnumpy(self.arr)
    
    def printDevice(self):
        if self.device is None:
            print('CPU')
        else:
            name = getGPUs()[self.device.id].name
            print('GPU %d - %s' % (self.device.id, name))

    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return self.arr
    
    def norm(self, N=2):
        """Function to compute vector N-norm using Numpy"""
        return cp.linalg.norm(self.getNdArray().ravel(), ord=N)

    def zero(self):
        """Function to zero out a vector"""
        self.getNdArray().fill(0)
        return self

    def max(self):
        """Function to obtain maximum value in the vector"""
        return self.getNdArray().max()

    def min(self):
        """Function to obtain minimum value in the vector"""
        return self.getNdArray().min()

    def set(self, val):
        """Function to set all values in the vector"""
        self.getNdArray().fill(val)
        return self

    def scale(self, sc):
        """Function to scale a vector"""
        self.getNdArray()[:] *= sc
        return self

    def addbias(self, bias):
        self.getNdArray()[:] += bias
        return self

    def rand(self, snr=1.):
        """Fill vector with random number (~U[1,-1]) with a given SNR"""
        rms = cp.sqrt(cp.mean(cp.square(self.getNdArray())))
        amp_noise = 1.0
        if rms != 0.:
            amp_noise = cp.sqrt(3. / snr) * rms  # sqrt(3*Power_signal/SNR)
        self.getNdArray()[:] = amp_noise * (2. * cp.random.random(self.getNdArray().shape) - 1.)
        return self

    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        vec_clone = deepcopy(self)  # Deep clone of vector
        # Checking if a vector space was provided
        if vec_clone.getNdArray().size == 0:
            vec_clone.arr = cp.zeros(vec_clone.shape, dtype=self.getNdArray().dtype)
        return vec_clone

    def cloneSpace(self):
        """Function to clone vector space only (vector without actual vector array by using empty array of size 0)"""
        arr = cp.empty(0, dtype=self.getNdArray().dtype)
        vec_space = VectorCupy(arr)
        # Cloning space of input vector
        vec_space.shape = self.shape
        vec_space.ndim = self.ndim
        vec_space.size = self.size
        return vec_space

    def checkSame(self, other):
        """Function to check dimensionality of vectors"""
        return self.shape == other.shape
    
    def abs(self):
        self.getNdArray()[:] = cp.abs(self.getNdArray())
        return self

    def sign(self):
        self.getNdArray()[:] = cp.sign(self.getNdArray())
        return self

    def reciprocal(self):
        self.getNdArray()[:] = 1. / self.getNdArray()
        return self

    def maximum(self, other):
        if cp.isscalar(other):
            self.getNdArray()[:] = cp.maximum(self.getNdArray(), other)
            return self
        elif isinstance(other, VectorCupy):
            if not self.checkSame(other):
                raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
            if not self._check_same_device(other):
                raise ValueError('Provided input has to live in the same device')
            self.getNdArray()[:] = cp.maximum(self.getNdArray(), other.getNdArray())
            return self
        else:
            raise TypeError("Provided input has to be either a scalar or a %s!" % self.whoami)

    def conj(self):
        self.getNdArray()[:] = cp.conj(self.getNdArray())
        return self

    def pow(self, power):
        """Compute element-wise power of the vector"""
        self.getNdArray()[:] = self.getNdArray() ** power
        return self

    def real(self):
        """Return the real part of the vector"""
        self.getNdArray()[:] = self.getNdArray().real
        return self

    def imag(self,):
        """Return the imaginary part of the vector"""
        self.getNdArray()[:] = self.getNdArray().imag
        return self

    def copy(self, other):
        """Function to copy vector from input vector"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: self = %s; other = %s" % (self.shape, other.shape))
        # Element-wise copy of the input array
        self.getNdArray()[:] = other.getNdArray()
        return self

    def scaleAdd(self, other, sc1=1.0, sc2=1.0):
        """Function to scale a vector"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: self = %s; other = %s" % (self.shape, other.shape))
        # Performing scaling and addition
        if not self._check_same_device(other):
            raise ValueError('Provided input has to live in the same device')
        self.getNdArray()[:] = sc1 * self.getNdArray() + sc2 * other.getNdArray()
        return self

    def dot(self, other):
        """Function to compute dot product between two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("Vector size mismatching: self = %d; other = %d" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: self = %s; other = %s" % (self.shape, other.shape))
        if not self._check_same_device(other):
            raise ValueError('Provided input has to live in the same device')
        return cp.vdot(self.getNdArray().flatten(), other.getNdArray().flatten())

    def multiply(self, other):
        """Function to multiply element-wise two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("Vector size mismatching: self = %d; other = %d" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: self = %s; other = %s" % (self.shape, other.shape))
        # Performing element-wise multiplication
        if not self._check_same_device(other):
            raise ValueError('Provided input has to live in the same device')
        self.getNdArray()[:] = cp.multiply(self.getNdArray(), other.getNdArray())
        return self

    def isDifferent(self, other):
        """Function to check if two vectors are identical using built-in hash function"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        if not self._check_same_device(other):
            raise ValueError('Provided input has to live in the same device')
        # Using Hash table for python2 and numpy built-in function array_equal otherwise
        if version_info[0] == 2:
            # First make both array buffers read-only
            self.arr.flags.writeable = False
            other.arr.flags.writeable = False
            chcksum1 = hash(self.getNdArray().data)
            chcksum2 = hash(other.getNdArray().data)
            # Remake array buffers writable
            self.arr.flags.writeable = True
            other.arr.flags.writeable = True
            isDiff = (chcksum1 != chcksum2)
        else:
            isDiff = (not cp.equal(self.getNdArray(), other.getNdArray()).all())
        return isDiff

    def clipVector(self, low, high):
        """Function to bound vector values based on input vectors low and high"""
        if not isinstance(low, VectorCupy):
            raise TypeError("Provided input low vector not a %s!" % self.whoami)
        if not isinstance(high, VectorCupy):
            raise TypeError("Provided input high vector not a %s!" % self.whoami)
        self.getNdArray()[:] = cp.minimum(cp.maximum(low.getNdArray(), self.getNdArray()), high.getNdArray())
        return self
    
    def plot(self):
        return self.getNdArray().get()
