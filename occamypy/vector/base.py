import os

import numpy as np

from occamypy.utils import sep
from occamypy.vector.axis_info import AxInfo


class Vector:
    """
    Abstract base vector class

    Attributes:
        shape: tuple containing the number of samples for each dimension
        size: total amount of samples
        ndim: number of dimensions
        dtype: array data type (e.g., float, int) based on the array backend
        ax_info: list of AxInfo objects describing the array axes

    Methods:
        getNdArray: to access the array
        norm: compute the vector N-norm
        zero: fill the vector with zeros (in-place)
        max: get the max value
        min: get the min value
        set: fill the vector with a certain value (in-place)
        scale: multiply by a scalar (in-place)
        addbias: add a scalar (in-place)
        rand: fill the vector with a uniform random number with a given SNR (in-place)
        randn: fill the vector with a normal random number with a given SNR (in-place)
        clone: deep-copy the input vector
        cloneSpace: copy the attributes, leaving the array empty
        checkSame: check if the input vector lives in the same vector space
        writeVec: write the vector to disk
        abs: compute the absolute value (in-place)
        sign: compute the sign function (in-place)
        reciprocal: compute the reciprocal (in-place)
        maximum: compute the element-wise maximum w.r.t. the input vector (in-place)
        conj: compute the conjugate (in-place)
        transpose: transpose the dimensions (in-place)
        hermitian: compute the conjugate-transpose (in-place)
        pow: compute the element-wise power (in-place)
        real: get the real part (in-place)
        imag: get the imaginary part (in-place)
        copy: copy the input vector array
        scaleAdd: weighted sum with the input vector (in-place)
        dot: compute the dot product with the input vector
        multiply: element-wise multiplication (in-place) with the input vector
        isDifferent: check if two vectors are identical
        clip: clip the array value between a lower and upper bounds (in-place)
        plot: get a plottable numpy.ndarray
    """
    
    def __init__(self, ax_info: list = None):
        """
        Vector constructor

        Args:
            ax_info: list of AxInfo objects about the vector axes
        """
        self.shape = None
        self.size = None
        self.ndim = None
        self.type = type(self)
        if ax_info is None:
            ax_info = []
        elif type(ax_info) not in [tuple, list]:
            raise ValueError("ax_info has to be a list")
        self.ax_info = list(ax_info)
    
    def __repr__(self):
        return self.getNdArray().__repr__()

    def __add__(self, other):  # self + other
        if type(other) in [int, float]:
            self.addbias(other)
            return self
        elif isinstance(other, Vector):
            self.scaleAdd(other)
            return self
        else:
            raise TypeError("Argument has to be either scalar or vector, got %r instead" % other)
    
    def __sub__(self, other):  # self - other
        self.__add__(-other)
        return self
    
    def __neg__(self):  # -self
        self.scale(-1)
        return self
    
    def __mul__(self, other):  # self * other
        if type(other) in [int, float]:
            self.scale(other)
            return self
        elif isinstance(other, Vector):
            self.multiply(other)
            return self
        else:
            raise NotImplementedError
    
    def __rmul__(self, other):
        if type(other) in [int, float]:
            self.scale(other)
            return self
        elif isinstance(other, Vector):
            self.multiply(other)
            return self
        else:
            raise NotImplementedError
    
    def __pow__(self, power, modulo=None):
        if type(power) in [int, float]:
            self.pow(power)
        else:
            raise TypeError('power has to be a scalar')
    
    def __abs__(self):
        self.abs()
    
    def __truediv__(self, other):  # self / other
        if type(other) in [int, float]:
            self.scale(1 / other)
        elif isinstance(other, Vector):
            self.multiply(other.clone().reciprocal())
        else:
            raise TypeError('other has to be either a scalar or a vector')
    
    def __getitem__(self, item):
        return self.getNdArray()[item]
    
    def __setitem__(self, key, value):
        self.getNdArray()[key] = value
    
    def init_ax_info(self):
        """Initialize the AxInfo list for every axis"""
        self.ax_info = [AxInfo(n=s) for s in self.shape]
    
    @property
    def whoami(self):
        return type(self).__name__
    
    @property
    def dtype(self):
        return self.getNdArray().dtype
    
    def getNdArray(self):
        """Get the vector content"""
        raise NotImplementedError("getNdArray must be overwritten")
    
    def norm(self, N=2):
        """Compute vector N-norm

        Args:
            N: vector norm type
        """
        raise NotImplementedError("norm must be overwritten")
    
    def zero(self):
        """Fill the vector with zeros"""
        self.set(0)
        return self
    
    def max(self):
        """Get the vector maximum value"""
        raise NotImplementedError("max must be overwritten")
    
    def min(self):
        """Get the vector minimum value"""
        raise NotImplementedError("min must be overwritten")
    
    def set(self, val):
        """Fill the vector with a value

        Args:
            val: value to fill the vector
        """
        raise NotImplementedError("set must be overwritten")
    
    def scale(self, sc):
        """Scale with a coefficient"""
        self.getNdArray()[:] *= sc
        return self
    
    def addbias(self, bias):
        """Add a bias"""
        self.getNdArray()[:] += bias
        return self
    
    def rand(self, low: float = -1., high: float = 1.):
        """Fill vector with random number ~U[low, high]
        Args:
            low: lower distribution bound
            high: upper distribution bound
        """
        raise NotImplementedError("rand must be overwritten")
    
    def randn(self, mean: float = 0., std: float = 1.):
        """Fill vector with random number ~N[mean, std]
        
        Args:
            mean: distribution mean
            std: distribution standard deviation
        """
        raise NotImplementedError("randn must be overwritten")
    
    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        raise NotImplementedError("clone must be overwritten")
    
    def cloneSpace(self):
        """Function to clone vector space"""
        raise NotImplementedError("cloneSpace must be overwritten")
    
    def checkSame(self, other):
        """Check to make sure the vectors exist in the same vector space"""
        return self.shape == other.shape
    
    def writeVec(self, filename, mode='w'):
        """
        Write vector to file

        Args:
            filename: path/to/file.ext
            mode: 'a' for append, 'w' for overwriting
        """
        # Check writing mode
        if mode not in 'wa':
            raise ValueError("Mode must be appending (a) or writing (w)")
        # Construct ax_info if the object has getHyper
        if hasattr(self, "getHyper"):
            hyper = self.getHyper()
            self.ax_info = []
            for iaxis in range(hyper.getNdim()):
                self.ax_info.append(AxInfo(hyper.getAxis(iaxis + 1).n,
                                           hyper.getAxis(iaxis + 1).o,
                                           hyper.getAxis(iaxis + 1).data,
                                           hyper.getAxis(iaxis + 1).label))
        
        # check output file type
        _, ext = os.path.splitext(filename)  # file extension
        
        # SEP vector with header in filename.H and binary in DATAPATH/filename.H@
        if ext == ".H":
            # writing header/pointer file if not present and not append mode
            if not (os.path.isfile(filename) and mode in 'a'):
                binfile = sep.datapath + filename.split('/')[-1] + '@'
                with open(filename, mode) as f:
                    # Writing axis info
                    if self.ax_info:
                        for ii, ax_info in enumerate(self.ax_info):
                            f.write(ax_info.to_string(ii+1))
                    else:
                        for ii, n_axis in enumerate(tuple(reversed(self.shape))):
                            ax_info = AxInfo(n=n_axis)
                            f.write(ax_info.to_string(ii+1))
                    
                    # Writing last axis for allowing appending (unless we are dealing with a scalar)
                    if self.shape != (1,):
                        ax_info = AxInfo(n=1)
                        f.write(ax_info.to_string(self.ndim + 1))
                    f.write("in='%s'\n" % binfile)
                    esize = "esize=4\n"
                    if self.getNdArray().dtype == np.complex64:
                        esize = "esize=8\n"
                    f.write(esize)
                    f.write("data_format=\"native_float\"\n")
                f.close()
            else:
                binfile = sep.get_binary(filename)
                if mode in 'a':
                    axes = sep.get_axes(filename)
                    # Number of vectors already present in the file
                    if self.shape == (1,):
                        n_vec = axes[0][0]
                        append_dim = self.ndim
                    else:
                        n_vec = axes[self.ndim][0]
                        append_dim = self.ndim + 1
                    with open(filename, mode) as f:
                        ax_info = AxInfo(n_vec + 1)
                        f.write(ax_info.to_string(append_dim))
                    f.close()
            # Writing binary file
            fmt = '>f'
            if self.getNdArray().dtype == np.complex64 or self.getNdArray().dtype == np.complex128:
                fmt = '>c8'
            with open(binfile, mode + 'b') as f:
                # Writing big-ending floating point number
                if np.isfortran(self.getNdArray()):  # Forcing column-wise binary writing
                    # self.getNdArray().flatten('F').astype(fmt,copy=False).tofile(fid)
                    self.getNdArray().flatten('F').tofile(f, format=fmt)
                else:
                    # self.getNdArray().astype(fmt,order='C',subok=False,copy=False).tofile(fid)
                    self.getNdArray().tofile(f, format=fmt)
            f.close()
        
        # numpy dictionary
        elif ext == '.npy':
            if mode not in 'a':
                if self.ax_info:
                    np.save(file=filename,
                            arr=dict(arr=self.getNdArray(),
                                     ax_info=self.ax_info),
                            allow_pickle=True)
                else:
                    np.save(file=filename, arr=self.getNdArray(), allow_pickle=False)
        else:
            raise NotImplementedError("Extension %s not implemented yet" % ext)
        return
    
    def abs(self):
        """Compute in-place absolute value"""
        raise NotImplementedError('abs method must be implemented')
    
    def sign(self):
        """Compute in-place sign function"""
        raise NotImplementedError('sign method have to be implemented')
    
    def reciprocal(self):
        """Compute in-place reciprocal"""
        self.getNdArray()[:] = 1. / self.getNdArray()
        return self
    
    def maximum(self, other):
        """Return a new vector of element-wise maximum of self and other"""
        raise NotImplementedError('maximum method must be implemented')
    
    def conj(self):
        """Compute the complex conjugate of the vector"""
        raise NotImplementedError('conj method must be implemented')
    
    def transpose(self):
        """Compute the in-place transpose of the vector"""
        raise NotImplementedError('transpose method must be implemented')
    
    def hermitian(self):
        """Compute the in-place hermitian, i.e. the conjugate transpose"""
        return self.transpose().conj()
    
    def pow(self, power):
        """Compute the in-place element-wise power of the vector"""
        raise NotImplementedError('pow method must be implemented')
    
    def real(self):
        """Return the in-place real part of the vector"""
        raise NotImplementedError('real method must be implemented')
    
    def imag(self):
        """Return the in-place imaginary part of the vector"""
        raise NotImplementedError('imag method must be implemented')
        
    def copy(self, other):
        """
        Copy input vector

        Args:
            other: vector to be copied
        """
        # Checking whether the input is a vector or not
        if not isinstance(other, self.type):
            raise TypeError("Provided input vector has to be of the same type!")
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError('Dimensionality not equal: self = %s; other = %s' % (self.shape, other.shape))
        # Element-wise copy of the input array
        self.getNdArray()[:] = other.getNdArray()
        return self
    
    def scaleAdd(self, other, sc1=1.0, sc2=1.0):
        """
        Scale two vectors and add them to the first one

        Args:
            other: vector to be added
            sc1: scaling factor of self
            sc2: scaling factor of other
        """
        # Checking whether the input is a vector or not
        if not isinstance(other, self.type):
            raise TypeError("Provided input vector has to be of the same type!")
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError("Dimensionality not equal: self = %s; other = %s" % (self.shape, other.shape))
        # Performing scaling and addition
        self.getNdArray()[:] = sc1 * self.getNdArray() + sc2 * other.getNdArray()
        return self
    
    def dot(self, other):
        """Compute dot product

        Args:
            other: second vector
        """
        raise NotImplementedError("dot must be overwritten")
    
    def multiply(self, other):
        """Element-wise multiplication

        Args:
            other: second vector
        """
        raise NotImplementedError("multiply must be overwritten")
    
    def isDifferent(self, other):
        """Check if two vectors are identical

        Args:
            other: second vector
        """
        raise NotImplementedError("isDifferent must be overwritten")
    
    def clip(self, low, high):
        """
        Bound vector values between two values

        Args:
            low: lower bound value
            high: upper bound value
        """
        raise NotImplementedError("clip must be overwritten")
    
    def plot(self):
        """Get a plottable array"""
        raise NotImplementedError("plot must be overwritten")


class VectorSet:
    """Class to store different vectors that live in the same Space"""
    
    def __init__(self):
        """VectorSet constructor"""
        self.vecSet = []

    def append(self, other, copy=True):
        """Method to add vector to the set
        Args:
            other: vector to be added
            copy: clone or not the vector
        """
        # Checking dimensionality if a vector is present
        if self.vecSet:
            if not self.vecSet[0].checkSame(other):
                raise ValueError("ERROR! Provided vector not in the same Space of the vector set")
        
        self.vecSet.append(other.clone()) if copy else self.vecSet.append(other)
    
    def writeSet(self, filename, mode="a"):
        """
        Write the set to file, all the vectors in set will be appended
        
        Args:
            filename: path/to/file.ext
            mode: 'a' for append, 'w' for overwriting
        """
        if mode not in "aw":
            raise ValueError("ERROR! mode must be either a (append) or w (write)")
        for idx_vec, vec_i in enumerate(self.vecSet):
            if mode == "w" and idx_vec == 0:
                wr_mode = "w"
            else:
                wr_mode = "a"
            vec_i.writeVec(filename, wr_mode)
        self.vecSet = []  # List of vectors of the set


class superVector(Vector):
    """Class to handle a list of vectors as one"""

    def __init__(self, *args):
        """
        superVector constructor

        Args:
            *args: vectors or superVectors or vectors list objects
        """
        self.vecs = []
        for v in args:
            if v is None:
                continue
            elif isinstance(v, Vector):
                self.vecs.append(v)
            elif isinstance(v, list):
                for vv in v:
                    if vv is None:
                        continue
                    elif isinstance(vv, Vector):
                        self.vecs.append(vv)
            else:
                raise TypeError('Argument must be either a vector or a superVector')
        
        self.n = len(self.vecs)
    
    @property
    def ndim(self):
        return [self.vecs[idx].ndim for idx in range(self.n)]
    
    @property
    def shape(self):
        return [self.vecs[idx].shape for idx in range(self.n)]
    
    @property
    def size(self):
        return sum([self.vecs[idx].size for idx in range(self.n)])

    def __getitem__(self, item):
        return self.vecs[item]
    
    def getNdArray(self):
        return [self.vecs[idx].getNdArray() for idx in range(self.n)]
    
    def norm(self, N=2):
        norm = [pow(self.vecs[idx].norm(N), N) for idx in range(self.n)]
        return pow(sum(norm), 1. / N)
    
    def set(self, val):
        for idx in range(self.n):
            self.vecs[idx].set(val)
        return self
    
    def zero(self):
        for idx in range(self.n):
            self.vecs[idx].zero()
        return self
    
    def max(self):
        return max([self.vecs[idx].max() for idx in range(self.n)])
    
    def min(self):
        return min([self.vecs[idx].min() for idx in range(self.n)])
    
    def scale(self, sc):
        if type(sc) is not list:
            sc = [sc] * self.n
        for idx in range(self.n):
            self.vecs[idx].scale(sc[idx])
        return self
    
    def addbias(self, bias):
        if type(bias) is not list:
            bias = [bias] * self.n
        for idx in range(self.n):
            self.vecs[idx].addbias(bias[idx])
        return self
    
    def rand(self, low: float = -1., high: float = 1.):
        for idx in range(self.n):
            self.vecs[idx].rand(low=low, high=high)
        return self
    
    def randn(self, mean: float = 0., std: float = 1.):
        for idx in range(self.n):
            self.vecs[idx].randn(mean=mean, std=std)
        return self
   
    def clone(self):
        vecs = [self.vecs[idx].clone() for idx in range(self.n)]
        return superVector(vecs)
    
    def cloneSpace(self):
        return superVector([self.vecs[idx].cloneSpace() for idx in range(self.n)])
    
    def checkSame(self, other):
        # Checking type
        if not isinstance(other, superVector):
            raise TypeError('Input variable is not a superVector')
        checkspace = np.asarray([self.vecs[idx].checkSame(other.vecs[idx]) for idx in range(self.n)])
        notsame = np.where(checkspace is False)[0]
        for v in notsame:
            raise Warning('Component %d not in the same space!' % v)
        return np.all(checkspace == True)
    
    # Combination of different vectors
    def copy(self, vecs_in):
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        for idx in range(self.n):
            self.vecs[idx].copy(vecs_in.vecs[idx])
        return self
    
    def scaleAdd(self, vecs_in, sc1=1.0, sc2=1.0):
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        for idx in range(self.n):
            self.vecs[idx].scaleAdd(vecs_in.vecs[idx], sc1, sc2)
        return self
    
    def dot(self, vecs_in):
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        return sum([self.vecs[idx].dot(vecs_in.vecs[idx]) for idx in range(self.n)])
    
    def multiply(self, vecs_in):
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        for idx in range(self.n):
            self.vecs[idx].multiply(vecs_in.vecs[idx])
        return self
    
    def isDifferent(self, vecs_in):
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        return any([self.vecs[idx].isDifferent(vecs_in.vecs[idx]) for idx in range(self.n)])
    
    def clip(self, lows, highs):
        for idx in range(self.n):
            self.vecs[idx].clip(lows[idx], highs[idx])
        return self
    
    def abs(self):
        for idx in range(self.n):
            self.vecs[idx].abs()
        return self
    
    def sign(self):
        for idx in range(self.n):
            self.vecs[idx].sign()
        return self
    
    def reciprocal(self):
        for idx in range(self.n):
            self.vecs[idx].reciprocal()
        return self
    
    def maximum(self, other):
        if np.isscalar(other):
            for idx in range(self.n):
                self.vecs[idx].maximum(other)
            return self
        elif type(other) is not superVector:
            raise TypeError("Input variable is not a superVector")
        if other.n != self.n:
            raise ValueError('Input must have the same length of self')
        for idx in range(self.n):
            self.vecs[idx].maximum(other.vecs[idx])
        return self
    
    def conj(self):
        for idx in range(self.n):
            self.vecs[idx].conj()
        return self
    
    def real(self):
        for idx in range(self.n):
            self.vecs[idx].real()
        return self
    
    def imag(self, ):
        for idx in range(self.n):
            self.vecs[idx].imag()
        return self
    
    def pow(self, power):
        for idx in range(self.n):
            self.vecs[idx].pow(power)
        return self
    
    def writeVec(self, filename, mode="w"):
        _, ext = os.path.splitext(filename)
        for ii, vec_cmp in enumerate(self.vecs):
            # Writing components to different files
            filename_cmp = ".".join(filename.split('.')[:-1]) + "_comp%s" % (ii + 1)
            # Writing files (recursively)
            vec_cmp.writeVec(filename_cmp + ext, mode)
        return
