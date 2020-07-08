import numpy as np
from copy import deepcopy
from sys import version_info
from . import Vector, VectorOC
from occamypy.vector import sep


class VectorIC(Vector):
    """In-core python vector class"""
    
    def __init__(self, in_vec):
        """
        VectorIC constructor: arr=np.array
        This class stores array with C memory order (i.e., row-wise sorting)
        """
        
        # Verify that input is a numpy array or header file or vectorOC
        if isinstance(in_vec, VectorOC):  # VectorOC passed to constructor
            self.arr, self.ax_info = sep.read_file(in_vec.vecfile)
        elif isinstance(in_vec, str):  # Header file passed to constructor
            self.arr, self.ax_info = sep.read_file(in_vec)
        elif isinstance(in_vec, np.ndarray):  # Numpy array passed to constructor
            if np.isfortran(in_vec):
                raise TypeError('Input array not a C contiguous array!')
            self.arr = np.array(in_vec, copy=False)
            self.ax_info = None
        elif isinstance(in_vec, tuple):  # Tuple size passed to constructor
            self.arr = np.zeros(tuple(reversed(in_vec)))
            self.ax_info = None
        else:  # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")
        
        # Number of elements per axis (tuple). Checking also the memory order
        self.shape = self.arr.shape  # If fortran the first axis is the "fastest"
        self.ndims = len(self.shape)  # Number of axes integer
        self.size = self.arr.size  # Total number of elements
        super(VectorIC, self).__init__()
    
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return self.arr
    
    def size(self):
        return self.getNdArray().size
    
    def norm(self, N=2):
        """Function to compute vector N-norm using Numpy"""
        return np.linalg.norm(self.getNdArray().flatten(), ord=N)
    
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
        if vec_clone.getNdArray().size == 0:
            vec_clone.arr = np.zeros(vec_clone.shape, dtype=self.getNdArray().dtype)
        return vec_clone
    
    def cloneSpace(self):
        """Function to clone vector space only (vector without actual vector array by using empty array of size 0)"""
        vec_space = VectorIC(np.empty(0, dtype=self.getNdArray().dtype))
        # Cloning space of input vector
        vec_space.ndims = self.ndims
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
    
    def maximum(self, vec2):
        if np.isscalar(vec2):
            self.getNdArray()[:] = np.maximum(self.getNdArray(), vec2)
            return self
        elif isinstance(vec2, VectorIC):
            if not self.checkSame(vec2):
                raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
            self.getNdArray()[:] = np.maximum(self.getNdArray(), vec2.getNdArray())
            return self
        else:
            raise TypeError('Provided input has to be either a scalar or a vectorIC')
    
    def conj(self):
        self.getNdArray()[:] = np.conjugate(self.getNdArray())
        return self
    
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
    
    def copy(self, vec2):
        """Function to copy vector from input vector"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, VectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
        # Element-wise copy of the input array
        self.getNdArray()[:] = vec2.getNdArray()
        return self
    
    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        """Function to scale a vector"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, VectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
        # Performing scaling and addition
        self.getNdArray()[:] = sc1 * self.getNdArray() + sc2 * vec2.getNdArray()
        return self
    
    def dot(self, vec2):
        """Function to compute dot product between two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, VectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking size (must have same number of elements)
        if self.size != vec2.size:
            raise ValueError("Vector size mismatching: vec1 = %d; vec2 = %d" % (self.size, vec2.size))
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
        return np.vdot(self.getNdArray().ravel(), vec2.getNdArray().ravel())
    
    def multiply(self, vec2):
        """Function to multiply element-wise two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, VectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking size (must have same number of elements)
        if self.size != vec2.size:
            raise ValueError("Vector size mismatching: vec1 = %s; vec2 = %s" % (self.size, vec2.size))
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
        # Performing element-wise multiplication
        self.getNdArray()[:] = np.multiply(self.getNdArray(), vec2.getNdArray())
        return self
    
    def isDifferent(self, vec2):
        """Function to check if two vectors are identical using built-in hash function"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, VectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Using Hash table for python2 and numpy built-in function array_equal otherwise
        if version_info[0] == 2:
            # First make both array buffers read-only
            self.arr.flags.writeable = False
            vec2.arr.flags.writeable = False
            chcksum1 = hash(self.getNdArray().data)
            chcksum2 = hash(vec2.getNdArray().data)
            # Remake array buffers writable
            self.arr.flags.writeable = True
            vec2.arr.flags.writeable = True
            isDiff = (chcksum1 != chcksum2)
        else:
            isDiff = (not np.array_equal(self.getNdArray(), vec2.getNdArray()))
        return isDiff
    
    def clipVector(self, low, high):
        """Function to bound vector values based on input vectors low and high"""
        if not isinstance(low, VectorIC):
            raise TypeError("Provided input low vector not a vectorIC!")
        if not isinstance(high, VectorIC):
            raise TypeError("Provided input high vector not a vectorIC!")
        self.getNdArray()[:] = np.minimum(np.maximum(low.getNdArray(), self.getNdArray()), high.getNdArray())
        return self