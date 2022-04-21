from copy import deepcopy
from sys import version_info

import cupy as cp
import numpy as np
from GPUtil import getGPUs, getFirstAvailable

from occamypy.vector.base import Vector
from occamypy.numpy.vector import VectorNumpy


class VectorCupy(Vector):
    """Vector class based on cupy.ndarray"""

    def __init__(self, in_content, device=None, *args, **kwargs):
        """
        VectorCupy constructor

        Args:
            in_content: numpy.ndarray, cupy.ndarray, tuple or VectorNumpy
            device: computation device (None for CPU, -1 for least used GPU)
            *args: list of arguments for Vector construction
            **kwargs: dict of arguments for Vector construction
        """
        if isinstance(in_content, cp.ndarray) or isinstance(in_content, np.ndarray):
            if cp.isfortran(in_content):
                raise TypeError('Input array not a C contiguous array!')
            self.arr = cp.array(in_content, copy=False)
        elif isinstance(in_content, tuple):  # Tuple size passed to constructor
            # self.arr = cp.zeros(tuple(reversed(in_content)))
            self.arr = cp.zeros(in_content)
        elif isinstance(in_content, VectorNumpy):
            self.arr = in_content.getNdArray().copy()
            self.ax_info = in_content.ax_info
        else:  # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")
        
        self.setDevice(device)
        
        super(VectorCupy, self).__init__(*args, **kwargs)
        self.shape = self.arr.shape   # Number of elements per axis (tuple)
        self.ndim = self.arr.ndim    # Number of axes (integer)
        self.size = self.arr.size     # Total number of elements (integer)

    def _check_same_device(self, other):
        if not isinstance(self, VectorCupy):
            raise TypeError("self vector has to be a VectorCupy")
        if not isinstance(other, VectorCupy):
            raise TypeError("other vector has to be a VectorCupy")
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
    
    @property
    def deviceName(self):
        if self.device is None:
            return "CPU"
        else:
            name = getGPUs()[self.device.id].name
            return "GPU %d - %s" % (self.device.id, name)

    def getNdArray(self):
        return self.arr
    
    def norm(self, N=2):
        return cp.linalg.norm(self.getNdArray().ravel(), ord=N)

    def zero(self):
        self.getNdArray().fill(0)
        return self

    def max(self):
        return self.getNdArray().max()

    def min(self):
        return self.getNdArray().min()

    def set(self, val):
        self.getNdArray().fill(val)
        return self

    def scale(self, sc):
        self.getNdArray()[:] *= sc
        return self

    def addbias(self, bias):
        self.getNdArray()[:] += bias
        return self

    def rand(self, low: float = -1., high: float = 1.):
        self.zero()
        self[:] += cp.random.uniform(low=low, high=high, size=self.shape)
        return self

    def randn(self, mean: float = 0., std: float = 1.):
        self.zero()
        self[:] += cp.random.normal(loc=mean, scale=std, size=self.shape)
        return self

    def clone(self):
        vec_clone = deepcopy(self)  # Deep clone of vector
        # Checking if a vector space was provided
        if vec_clone.getNdArray().size == 0:
            vec_clone.arr = cp.zeros(vec_clone.shape, dtype=self.getNdArray().dtype)
        return vec_clone

    def cloneSpace(self):
        arr = cp.empty(0, dtype=self.getNdArray().dtype)
        vec_space = VectorCupy(arr)
        vec_space.ax_info = self.ax_info
        # Cloning space of input vector
        vec_space.shape = self.shape
        vec_space.ndim = self.ndim
        vec_space.size = self.size
        return vec_space

    def checkSame(self, other):
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
        self.getNdArray()[:] = self.getNdArray() ** power
        return self

    def real(self):
        self.getNdArray()[:] = self.getNdArray().real
        return self

    def imag(self,):
        self.getNdArray()[:] = self.getNdArray().imag
        return self

    def copy(self, other):
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

    def clip(self, low, high):
        if not isinstance(low, VectorCupy):
            raise TypeError("Provided input low vector not a %s!" % self.whoami)
        if not isinstance(high, VectorCupy):
            raise TypeError("Provided input high vector not a %s!" % self.whoami)
        self.getNdArray()[:] = cp.minimum(cp.maximum(low.getNdArray(), self.getNdArray()), high.getNdArray())
        return self
    
    def plot(self):
        return self.getNdArray().get()
