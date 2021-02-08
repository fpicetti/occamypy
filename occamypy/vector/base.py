import os
import numpy as np
import h5py

from occamypy.utils import sep


class Vector:
    """Abstract python vector class"""
    
    def __init__(self):
        """Default constructor"""
        self.shape = None
        self.size = None
        self.ndim = None
        self.type = type(self)
    
    def __repr__(self):
        return self.getNdArray().__repr__()
    
    def __del__(self):
        """Default destructor"""
    
    def __add__(self, other):  # self + other
        if type(other) in [int, float]:
            self.addbias(other)
            return self
        elif isinstance(other, Vector):
            self.scaleAdd(other)
            return self
        else:
            raise TypeError('Argument has to be either scalar or vector, got %r instead' % other)
    
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
    
    @property
    def whoami(self):
        return type(self).__name__
    
    # Class vector operations
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        raise NotImplementedError("getNdArray must be overwritten")
    
    def norm(self, N=2):
        """Function to compute vector N-norm"""
        raise NotImplementedError("norm must be overwritten")
    
    def zero(self):
        """Function to zero out a vector"""
        self.set(0)
        return self
    
    def max(self):
        """Function to obtain maximum value within a vector"""
        raise NotImplementedError("max must be overwritten")
    
    def min(self):
        """Function to obtain minimum value within a vector"""
        raise NotImplementedError("min must be overwritten")
    
    def set(self, val):
        """Function to set all values in the vector"""
        raise NotImplementedError("set must be overwritten")
    
    def scale(self, sc):
        """Function to scale a vector"""
        self.getNdArray()[:] *= sc
        return self
    
    def addbias(self, bias):
        """Function to add bias to a vector"""
        self.getNdArray()[:] += bias
        return self
    
    def rand(self):
        """Function to randomize a vector"""
        raise NotImplementedError("rand must be overwritten")
    
    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        raise NotImplementedError("clone must be overwritten")
    
    def cloneSpace(self):
        """Function to clone vector space"""
        raise NotImplementedError("cloneSpace must be overwritten")
    
    def checkSame(self, other):
        """Function to check to make sure the vectors exist in the same space"""
        return self.shape == other.shape
    
    def writeVec(self, filename, mode='w'):
        """Function to write vector to file"""
        # Check writing mode
        if mode not in 'wa':
            raise ValueError("Mode must be appending 'a' or writing 'w' ")
        # Construct ax_info if the object has getHyper
        if hasattr(self, "getHyper"):
            hyper = self.getHyper()
            self.ax_info = []
            for iaxis in range(hyper.getNdim()):
                self.ax_info.append([hyper.getAxis(iaxis + 1).n,
                                     hyper.getAxis(iaxis + 1).o,
                                     hyper.getAxis(iaxis + 1).d,
                                     hyper.getAxis(iaxis + 1).label])
        
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
                            ax_id = ii + 1
                            f.write("n%s=%s o%s=%s d%s=%s label%s='%s'\n" % (
                                ax_id, ax_info[0], ax_id, ax_info[1], ax_id, ax_info[2], ax_id, ax_info[3]))
                    else:
                        for ii, n_axis in enumerate(tuple(reversed(self.shape))):
                            ax_id = ii + 1
                            f.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (ax_id, n_axis, ax_id, ax_id))
                    # Writing last axis for allowing appending (unless we are dealing with a scalar)
                    if self.shape != (1,):
                        ax_id = self.ndim + 1
                        f.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (ax_id, 1, ax_id, ax_id))
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
                        f.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (append_dim, n_vec + 1, append_dim, append_dim))
                    f.close()
            # Writing binary file
            fmt = '>f'
            if self.getNdArray().dtype == np.complex64 or self.getNdArray().dtype == np.complex128:
                format = '>c8'
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
                if self.ax_info:  # TODO fix saving of ax_info
                    axes = dict()
                    for ii, ax_info in enumerate(self.ax_info):
                        axes['%s' % ii + 1] = dict(n=ax_info[0], o=ax_info[1], d=ax_info[2], label=ax_info[3])
                    
                    np.save(file=filename,
                            arr=dict(arr=self.getNdArray(),
                                     ax_info=axes),
                            allow_pickle=True)
                else:
                    np.save(file=filename, arr=self.getNdArray(), allow_pickle=False)
        
        elif ext == '.h5':  # TODO implement saving to hdf5
            # https://moonbooks.org/Articles/How-to-save-a-large-dataset-in-a-hdf5-file-using-python--Quick-Guide/
            if mode not in 'a':
                with h5py.File(filename, 'wb') as f:
                    dset = f.create_dataset("vec", data=self.getNdArray())
            else:
                raise NotImplementedError
        
        else:
            raise ValueError("ERROR! Output format has to be H, npy, or h5")
        
        return
    
    # TODO implement on seplib
    def abs(self):
        """Return a vector containing the absolute values"""
        raise NotImplementedError('abs method must be implemented')
    
    # TODO implement on seplib
    def sign(self):
        """Return a vector containing the signs"""
        raise NotImplementedError('sign method have to be implemented')
    
    # TODO implement on seplib
    def reciprocal(self):
        """Return a vector containing the reciprocals of self"""
        self.getNdArray()[:] = 1. / self.getNdArray()
        return self
    
    # TODO implement on seplib
    def maximum(self, other):
        """Return a new vector of element-wise maximum of self and other"""
        raise NotImplementedError('maximum method must be implemented')
    
    # TODO implement on seplib
    def conj(self):
        """Compute the complex conjugate of the vector"""
        raise NotImplementedError('conj method must be implemented')
    
    def transpose(self):
        """Compute the transpose of the vector"""
        
    def hermitian(self):
        """Compute the hermitian, i.e. the conjugate transpose"""
        return self.transpose().conj()
    
    # TODO implement on seplib
    def pow(self, power):
        """Compute element-wise power of the vector"""
        raise NotImplementedError('pow method must be implemented')
    
    # TODO implement on seplib
    def real(self):
        """Return the real part of the vector"""
        raise NotImplementedError('real method must be implemented')
    
    # TODO implement on seplib
    def imag(self):
        """Return the imaginary part of the vector"""
        raise NotImplementedError('imag method must be implemented')
    
    # Combination of different vectors
    
    def copy(self, other):
        """Function to copy vector"""
        """Function to copy vector from input vector"""
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
        """Function to scale two vectors and add them to the first one"""
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
        """Function to compute dot product between two vectors"""
        raise NotImplementedError("dot must be overwritten")
    
    def multiply(self, other):
        """Function to multiply element-wise two vectors"""
        raise NotImplementedError("multiply must be overwritten")
    
    def isDifferent(self, other):
        """Function to check if two vectors are identical"""
        
        raise NotImplementedError("isDifferent must be overwritten")
    
    def clipVector(self, low, high):
        """
           Function to bound vector values based on input vectors min and max
        """
        raise NotImplementedError("clipVector must be overwritten")
    
    def plot(self):
        """Method to get a plottable array"""
        return self.getNdArray()


# Set of vectors (useful to store results and same-Space vectors together)
class VectorSet:
    """Class to store different vectors that live in the same Space"""
    
    def __init__(self):
        """Default constructor"""
        self.vecSet = []  # List of vectors of the set
    
    def __del__(self):
        """Default destructor"""
    
    def append(self, other, copy=True):
        """Method to add vector to the set"""
        # Checking dimensionality if a vector is present
        if self.vecSet:
            if not self.vecSet[0].checkSame(other):
                raise ValueError("ERROR! Provided vector not in the same Space of the vector set")
        
        self.vecSet.append(other.clone()) if copy else self.vecSet.append(other)
    
    def writeSet(self, filename, mode="a"):
        """Method to write to SEPlib file (by default it appends vectors to file)"""
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
    
    def __init__(self, *args):
        """
        superVector constructor
        :param args: vectors or superVectors or vectors list objects
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
        # super(superVector, self).__init__()
        # self.shape = None
        # self.size = None
        # self.ndim = None
        
    @property
    def ndim(self):
        return [self.vecs[idx].ndim for idx in range(self.n)]
    
    @property
    def shape(self):
        return [self.vecs[idx].shape for idx in range(self.n)]
    
    @property
    def size(self):
        return sum([self.vecs[idx].size for idx in range(self.n)])
    
    def __del__(self):
        """superVector destructor"""
        del self.vecs, self.n
    
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return [self.vecs[idx].getNdArray() for idx in range(self.n)]
    
    @property
    def shape(self):
        return [self.vecs[idx].shape for idx in range(self.n)]
    
    @property
    def size(self):
        return sum([self.vecs[idx].size for idx in range(self.n)])
    
    def norm(self, N=2):
        """Function to compute vector N-norm"""
        norm = [pow(self.vecs[idx].norm(N), N) for idx in range(self.n)]
        return pow(sum(norm), 1. / N)
    
    def set(self, val):
        """Function to set all values in the vector"""
        for idx in range(self.n):
            self.vecs[idx].set(val)
        return self
    
    def zero(self):
        """Function to zero out a vector"""
        for idx in range(self.n):
            self.vecs[idx].zero()
        return self
    
    def max(self):
        """Function to obtain maximum value within a vector"""
        return max([self.vecs[idx].max() for idx in range(self.n)])
    
    def min(self):
        """Function to obtain minimum value within a vector"""
        return min([self.vecs[idx].min() for idx in range(self.n)])
    
    def scale(self, sc):
        """Function to scale a vector"""
        if type(sc) is not list:
            sc = [sc] * self.n
        for idx in range(self.n):
            self.vecs[idx].scale(sc[idx])
        return self
    
    def addbias(self, bias):
        """Add a constant to the vector"""
        if type(bias) is not list:
            bias = [bias] * self.n
        for idx in range(self.n):
            self.vecs[idx].addbias(bias[idx])
        return self
    
    def rand(self, snr=1.0):
        """Function to randomize a vector"""
        for idx in range(self.n):
            self.vecs[idx].rand()
        return self
    
    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        vecs = [self.vecs[idx].clone() for idx in range(self.n)]
        return superVector(vecs)
    
    def cloneSpace(self):
        """Function to clone vector space"""
        return superVector([self.vecs[idx].cloneSpace() for idx in range(self.n)])
    
    def checkSame(self, other):
        """Function to check to make sure the vectors exist in the same space"""
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
        """Function to copy vector from input vector"""
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
        """Function to scale input vectors and add them to the original ones"""
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
        """Function to compute dot product between two vectors"""
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        return sum([self.vecs[idx].dot(vecs_in.vecs[idx]) for idx in range(self.n)])
    
    def multiply(self, vecs_in):
        """Function to multiply element-wise two vectors"""
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
        """Function to check if two vectors are identical"""
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        return any([self.vecs[idx].isDifferent(vecs_in.vecs[idx]) for idx in range(self.n)])
    
    def clipVector(self, lows, highs):
        for idx in range(self.n):
            self.vecs[idx].clipVector(lows[idx], highs[idx])
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
        """Method to write to vector to file within a Vector set"""
        _, ext = os.path.splitext(filename)
        for ii, vec_cmp in enumerate(self.vecs):
            # Writing components to different files
            filename_cmp = ".".join(filename.split('.')[:-1]) + "_comp%s" % (ii + 1)
            # Writing files (recursively)
            vec_cmp.writeVec(filename_cmp + ext, mode)
        return
