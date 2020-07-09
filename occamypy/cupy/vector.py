from copy import deepcopy
import os
from sys import version_info
import numpy as np
import cupy as cp
from GPUtil import getGPUs
from occamypy import Vector, VectorOC
from occamypy.utils import sep


class VectorCupy(Vector):
    """In-core python vector class based on Cupy"""

    def __init__(self, in_vec):
        """
        VectorIC constructor: arr=cp.ndarray
        This class stores array with C memory order (i.e., row-wise sorting)
        """

        # Verify that input is a numpy array or header file or vectorOC
        if isinstance(in_vec, VectorOC):  # VectorOC passed to constructor
            self.arr, self.ax_info = sep.read_file(in_vec.vecfile)
        elif isinstance(in_vec, str):  # Header file passed to constructor
            self.arr, self.ax_info = sep.read_file(in_vec)
        elif isinstance(in_vec, cp.ndarray):  # Cupy array passed to constructor
            if cp.isfortran(in_vec):
                raise TypeError('Input array not a C contiguous array!')
            self.arr = cp.array(in_vec, copy=False)
            self.ax_info = None
        elif isinstance(in_vec, np.ndarray):  # Numpy array passed to constructor
            if np.isfortran(in_vec):
                raise TypeError('Input array not a C contiguous array!')
            self.arr = cp.asarray(in_vec)
            self.ax_info = None
        elif isinstance(in_vec, tuple):  # Tuple size passed to constructor
            self.arr = cp.zeros(tuple(reversed(in_vec)))
            self.ax_info = None
        else:  # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")

        self.shape = self.arr.shape # Number of elements per axis (tuple)
        self.ndims = len(self.shape)  # Number of axes integer
        self.size = self.arr.size  # Total number of elements

        self.device = self.arr.device
        super(VectorCupy, self).__init__()

    def setDevice(self, devID=0):
        cp.cuda.Device(devID).use()

    def printDevice(self):
        name = getGPUs()[self.arr.device.id].name
        print('GPU selected: %d - %s' % (self.arr.device.id, name))

    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return self.arr

    def norm(self, N=2):
        """Function to compute vector N-norm using Numpy"""
        return cp.linalg.norm(self.getNdArray().flatten(), ord=N)

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
        arr = cp.empty(0,dtype=self.getNdArray().dtype)
        vec_space = VectorCupy(arr)
        # Cloning space of input vector
        vec_space.shape = self.shape
        vec_space.ndims = self.ndims
        vec_space.size = self.size
        return vec_space

    def checkSame(self, other):
        """Function to check dimensionality of vectors"""
        return self.shape == other.shape

    def writeVec(self, filename, mode='w'):
        """Function to write vector to file"""
        # Check writing mode
        if not mode in 'wa':
            raise ValueError("Mode must be appending 'a' or writing 'w' ")
        # Construct ax_info if the object has getHyper
        if hasattr(self, "getHyper"):
            hyper = self.getHyper()
            self.ax_info = []
            for iaxis in range(hyper.getNdim()):
                self.ax_info.append([hyper.getAxis(iaxis + 1).n, hyper.getAxis(iaxis + 1).o, hyper.getAxis(iaxis + 1).d,
                                     hyper.getAxis(iaxis + 1).label])
        # writing header/pointer file if not present and not append mode
        if not (os.path.isfile(filename) and mode in 'a'):
            binfile = sep.datapath + filename.split('/')[-1] + '@'
            with open(filename, mode) as fid:
                # Writing axis info
                if self.ax_info:
                    for ii, ax_info in enumerate(self.ax_info):
                        ax_id = ii + 1
                        fid.write("n%s=%s o%s=%s d%s=%s label%s='%s'\n" % (
                            ax_id, ax_info[0], ax_id, ax_info[1], ax_id, ax_info[2], ax_id, ax_info[3]))
                else:
                    for ii, n_axis in enumerate(tuple(reversed(self.shape))):
                        ax_id = ii + 1
                        fid.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (ax_id, n_axis, ax_id, ax_id))
                # Writing last axis for allowing appending (unless we are dealing with a scalar)
                if self.shape != (1,):
                    ax_id = self.ndims + 1
                    fid.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (ax_id, 1, ax_id, ax_id))
                fid.write("in='%s'\n" % binfile)
                esize = "esize=4\n"
                if self.getNdArray().dtype == cp.complex64 or self.getNdArray().dtype == cp.complex128:
                    esize = "esize=8\n"
                fid.write(esize)
                fid.write("data_format=\"native_float\"\n")
            fid.close()
        else:
            binfile = sep.get_binary(filename)
            if mode in 'a':
                axes = sep.get_axes(filename)
                # Number of vectors already present in the file
                if self.shape == (1,):
                    n_vec = axes[0][0]
                    append_dim = self.ndims
                else:
                    n_vec = axes[self.ndims][0]
                    append_dim = self.ndims + 1
                with open(filename, mode) as fid:
                    fid.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (append_dim, n_vec + 1, append_dim, append_dim))
                fid.close()
        # Writing binary file
        format = '>f'
        if self.getNdArray().dtype == cp.complex64 or self.getNdArray().dtype == cp.complex128:
            format = '>c8'
        with open(binfile, mode + 'b') as fid:
            # Writing big-ending floating point number
            if cp.isfortran(self.getNdArray()):  # Forcing column-wise binary writing
                self.getNdArray().ravel('F').astype(format).tofile(fid)
            else:
                self.getNdArray().astype(format).tofile(fid)
        fid.close()
        return

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
                raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, other.shape))
            self.getNdArray()[:] = cp.maximum(self.getNdArray(), other.getNdArray())
            return self
        else:
            raise TypeError('Provided input has to be either a scalar or a vectorIC')

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
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: vec1 = %s; vec2 = %s" % (self.shape, other.shape))
        # Element-wise copy of the input array
        self.getNdArray()[:] = other.getNdArray()
        return self

    def scaleAdd(self, other, sc1=1.0, sc2=1.0):
        """Function to scale a vector"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: vec1 = %s; vec2 = %s" % (self.shape, other.shape))
        # Performing scaling and addition
        self.getNdArray()[:] = sc1 * self.getNdArray() + sc2 * other.getNdArray()
        return self

    def dot(self, other):
        """Function to compute dot product between two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("Vector size mismatching: vec1 = %d; vec2 = %d" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: vec1 = %s; vec2 = %s" % (self.shape, other.shape))
        return cp.vdot(self.getNdArray().flatten(), other.getNdArray().flatten())

    def multiply(self, other):
        """Function to multiply element-wise two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("Vector size mismatching: vec1 = %d; vec2 = %d" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: vec1 = %s; vec2 = %s" % (self.shape, other.shape))
        # Performing element-wise multiplication
        self.getNdArray()[:] = cp.multiply(self.getNdArray(), other.getNdArray())
        return self

    def isDifferent(self, other):
        """Function to check if two vectors are identical using built-in hash function"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorCupy):
            raise TypeError("Provided input vector not a vectorIC!")
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
            raise TypeError("Provided input low vector not a vectorIC!")
        if not isinstance(high, VectorCupy):
            raise TypeError("Provided input high vector not a vectorIC!")
        self.getNdArray()[:] = cp.minimum(cp.maximum(low.getNdArray(), self.getNdArray()), high.getNdArray())
        return self


if __name__ == '__main__':
    import pyCuOperator

    x = VectorCupy(np.empty((1000, 20000))).set(1.)
    x.printDevice()


    # D = pyCuOperator.FirstDerivative(x)
    # n = x.clone().rand()
    # y = x.clone().set(10) + 0.01 * n
    # S = pyCuOperator.scalingOp(x, 10)
    # xinv = S / y
    # print('Error norm = %.2e' % (xinv.norm() - x.norm()))

    # Test Convolution
    nh = [5, 10]
    hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-0.03 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    kernel = hz[:, np.newaxis] * hx[np.newaxis, :]
    C = pyCuOperator.ConvND(x, kernel)
    C.dotTest(True)
    #
    # x = vectorCupy(cp.arange(9).reshape((3, 3)))
    # pad = ((2, 2), (3, 3))
    # P = pyCuOperator.ZeroPad(x, pad)

    print(0)
