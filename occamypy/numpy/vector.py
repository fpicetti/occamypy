from copy import deepcopy
from sys import version_info

import numpy as np

from occamypy import Vector


class VectorNumpy(Vector):
    """In-core python vector class"""
    
    def __init__(self, in_content):
        """
        VectorNumpy constructor: arr = np.array
        This class stores array with C memory order (i.e., row-wise sorting)
        """
        if isinstance(in_content, str):  # Header file name passed to constructor
            self.arr = np.load(in_content, allow_pickle=True)
        elif isinstance(in_content, np.ndarray):  # Numpy array passed to constructor
            if np.isfortran(in_content):
                raise TypeError('Input array not a C contiguous array!')
            self.arr = np.array(in_content, copy=False)
            self.ax_info = None
        elif isinstance(in_content, tuple):  # Tuple size passed to constructor
            # self.arr = np.zeros(tuple(reversed(in_vec)))
            self.arr = np.empty(in_content)
            self.ax_info = None
        else:  # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")
        
        super(VectorNumpy, self).__init__()
        # Number of elements per axis (tuple). Checking also the memory order
        self.shape = self.arr.shape  # If fortran the first axis is the "fastest"
        self.ndim = self.arr.ndim  # Number of axes integer
        self.size = self.arr.size  # Total number of elements
    
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return self.arr
    
    def norm(self, N=2):
        """Function to compute vector N-norm using Numpy"""
        return np.linalg.norm(self.getNdArray().ravel(), ord=N)
    
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
        rms = np.sqrt(np.mean(np.square(self.getNdArray())))
        amp_noise = 1.0
        if rms != 0.:
            amp_noise = np.sqrt(3. / snr) * rms  # sqrt(3*Power_signal/SNR)
        self.getNdArray()[:] = amp_noise * (2. * np.random.random(self.getNdArray().shape) - 1.)
        return self
    
    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        vec_clone = deepcopy(self)  # Deep clone of vector
        # Checking if a vector space was provided
        if vec_clone.getNdArray().size == 0:  # this is the shape of np.ndarray!
            vec_clone.arr = np.zeros(vec_clone.shape, dtype=self.getNdArray().dtype)
        return vec_clone
    
    def cloneSpace(self):
        """Function to clone vector space only (vector without actual vector array by using empty array of size 0)"""
        vec_space = VectorNumpy(np.empty(0, dtype=self.getNdArray().dtype))
        # Cloning space of input vector
        vec_space.ndim = self.ndim
        vec_space.shape = self.shape
        vec_space.size = self.size
        return vec_space
    
    def checkSame(self, other):
        """Function to check dimensionality of vectors"""
        return self.shape == other.shape
    
    def abs(self):
        self.getNdArray()[:] = np.abs(self.getNdArray())
        return self
    
    def sign(self):
        self.getNdArray()[:] = np.sign(self.getNdArray())
        return self
    
    def reciprocal(self):
        self.getNdArray()[:] = 1. / self.getNdArray()
        return self
    
    def maximum(self, other):
        if np.isscalar(other):
            self.getNdArray()[:] = np.maximum(self.getNdArray(), other)
            return self
        elif isinstance(other, VectorNumpy):
            if not self.checkSame(other):
                raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
            self.getNdArray()[:] = np.maximum(self.getNdArray(), other.getNdArray())
            return self
        else:
            raise TypeError("Provided input has to be either a scalar or a %s!" % self.whoami)
    
    def conj(self):
        self.getNdArray()[:] = np.conjugate(self.getNdArray())
        return self
    
    def transpose(self):
        other = VectorNumpy(tuple(reversed(self.shape)))
        other[:] = self.getNdArray().T
        return other
    
    def pow(self, power):
        """Compute element-wise power of the vector"""
        self.getNdArray()[:] = self.getNdArray() ** power
        return self
    
    def real(self):
        """Return the real part of the vector"""
        self.getNdArray()[:] = self.getNdArray().real
        return self
    
    def imag(self, ):
        """Return the imaginary part of the vector"""
        self.getNdArray()[:] = self.getNdArray().imag
        return self
    
    def copy(self, other):
        """Function to copy vector from input vector"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorNumpy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        # Element-wise copy of the input array
        self.getNdArray()[:] = other.getNdArray()
        return self
    
    def scaleAdd(self, other, sc1=1.0, sc2=1.0):
        """Function to scale a vector"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorNumpy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        # Performing scaling and addition
        self.getNdArray()[:] = sc1 * self.getNdArray() + sc2 * other.getNdArray()
        return self
    
    def dot(self, other):
        """Function to compute dot product between two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorNumpy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("Vector size mismatching: self = %d; other = %d" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        return np.vdot(self.getNdArray().ravel(), other.getNdArray().ravel())
    
    def multiply(self, other):
        """Function to multiply element-wise two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorNumpy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("Vector size mismatching: self = %s; other = %s" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        # Performing element-wise multiplication
        self.getNdArray()[:] = np.multiply(self.getNdArray(), other.getNdArray())
        return self
    
    def isDifferent(self, other):
        """Function to check if two vectors are identical using built-in hash function"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorNumpy):
            raise TypeError("Provided input vector not a %s!" % self.whoami)
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
            isDiff = (not np.array_equal(self.getNdArray(), other.getNdArray()))
        return isDiff
    
    def clipVector(self, low, high):
        """Function to bound vector values based on input vectors low and high"""
        if not isinstance(low, VectorNumpy):
            raise TypeError("Provided input low vector not a %s!" % self.whoami)
        if not isinstance(high, VectorNumpy):
            raise TypeError("Provided input high vector not a %s!" % self.whoami)
        self.getNdArray()[:] = np.minimum(np.maximum(low.getNdArray(), self.getNdArray()), high.getNdArray())
        return self
