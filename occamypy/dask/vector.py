import numpy as np
import os
import dask.distributed as daskD

from occamypy.utils.os import BUF_SIZE
from occamypy.utils import sep
from occamypy import Vector, VectorNumpy
from .utils import DaskClient

# Verify if SepVector modules are presents
try:
    import SepVector
    
    
    def call_constr_hyper(axes_in):
        """Function to remotely construct an SepVector using the axis object"""
        return SepVector.getSepVector(axes=axes_in)
except ImportError:
    SepVector = None


def _copy_from_NdArray(vecObj, NdArray):
    """Function to set vector values from numpy array"""
    vecObj.getNdArray()[:] = NdArray
    return


# Functions necessary to submit method calls using Dask client
def _call_getNdArray(vecObj):
    """Function to call getNdArray method"""
    res = vecObj.getNdArray()
    return res


def _call_shape(vecObj):
    """Function to return shape attribute"""
    res = vecObj.shape
    return res


def _call_ndim(vecObj):
    """Function to return ndim attribute"""
    res = vecObj.ndim
    return res


def _call_size(vecObj):
    """Function to return size attribute"""
    res = vecObj.size
    return res


def _call_norm(vecObj, N=2):
    """Function to call norm method"""
    res = vecObj.norm(N)
    return res


def _call_zero(vecObj):
    """Function to call zero method"""
    res = vecObj.zero()
    return res


def _call_max(vecObj):
    """Function to call max method"""
    res = vecObj.max()
    return res


def _call_min(vecObj):
    """Function to call min method"""
    res = vecObj.min()
    return res


def _call_set(vecObj, val):
    """Function to call set method"""
    res = vecObj.set(val)
    return res


def _call_scale(vecObj, sc):
    """Function to call scale method"""
    res = vecObj.scale(sc)
    return res


def _call_addbias(vecObj, bias):
    """Function to call addbias method"""
    res = vecObj.addbias(bias)
    return res


def _call_rand(vecObj):
    """Function to call rand method"""
    res = vecObj.rand()
    return res


def _call_clone(vecObj):
    """Function to call clone method"""
    res = vecObj.clone()
    return res


def _call_cloneSpace(vecObj):
    """Function to call cloneSpace method"""
    res = vecObj.cloneSpace()
    return res


def _call_checkSame(vecObj, vec2):
    """Function to call cloneSpace method"""
    res = vecObj.checkSame(vec2)
    return res


def _call_writeVec(vecObj, filename, mode):
    """Function to call cloneSpace method"""
    res = vecObj.writeVec(filename, mode)
    return res


def _call_abs(vecObj):
    """Function to call abs method"""
    res = vecObj.abs()
    return res


def _call_sign(vecObj):
    """Function to call sign method"""
    res = vecObj.sign()
    return res


def _call_reciprocal(vecObj):
    """Function to call reciprocal method"""
    res = vecObj.reciprocal()
    return res


def _call_maximum(vecObj, vec2):
    """Function to call maximum method"""
    res = vecObj.maximum(vec2)
    return res


def _call_conj(vecObj):
    """Function to call conj method"""
    res = vecObj.conj()
    return res


def _call_real(vecObj):
    """Function to call real method"""
    res = vecObj.real()
    return res


def _call_imag(vecObj):
    """Function to call imag method"""
    res = vecObj.imag()
    return res


def _call_pow(vecObj, power):
    """Function to call pow method"""
    res = vecObj.pow(power)
    return res


def _call_copy(vecObj, vec2):
    """Function to call copy method"""
    res = vecObj.copy(vec2)
    return res


def _call_scaleAdd(vecObj, vec2, sc1, sc2):
    """Function to call scaleAdd method"""
    res = vecObj.scaleAdd(vec2, sc1, sc2)
    return res


def _call_dot(vecObj, vec2):
    """Function to call dot method"""
    res = vecObj.dot(vec2)
    return res


def _call_multiply(vecObj, vec2):
    """Function to call multiply method"""
    res = vecObj.multiply(vec2)
    return res


def _call_isDifferent(vecObj, vec2):
    """Function to call isDifferent method"""
    res = vecObj.isDifferent(vec2)
    return res


def _call_clipVector(vecObj, low, high):
    """Function to call multiply method"""
    res = vecObj.clipVector(low, high)
    return res


# Check consistency between vectors
def checkVector(vec1, vec2):
    """Function to check type and chunks of Dask-vector objects"""
    if type(vec1) is not DaskVector:
        raise TypeError("Self vector is not a DaskVector")
    if type(vec2) is not DaskVector:
        raise TypeError("Input vector is not a DaskVector")
    Nvec1 = len(vec1.vecDask)
    Nvec2 = len(vec2.vecDask)
    if Nvec1 != Nvec2:
        raise ValueError("Number of chunks is different! (self chunks %s; vec2 chunks %s)" % (Nvec1, Nvec2))
    return


class DaskVector(Vector):
    """Definition of a vector object whose computations are performed through a Dask Client"""
    
    def __init__(self, dask_client, **kwargs):
        """
        Dask Vector constructor
        dask_client = [no default] - DaskClient; client object to use when submitting tasks (see dask_util module)
        kwargs:
         - vector_template = [no default] - vector class; Vector to use to create chunks of vectors
         - chunks          = [no default] - list; List defininig the size of the multiple instances of the vector template
         or
         - vectors         = [no default] - list; List containing vectors to be spread across Dask workers
         - copy            = [True] - boolean; Whether to copy the content of the vectors or not
         - chunks          = [None] - list; List defininig how the vector list should be spread; if not specified the vectors will be evenly distributed
         or
         - dask_vectors    = [no default] - list; List containing pointers to futures to vector object (useful for clone function)
        """
        # Client to submit tasks
        super(DaskVector).__init__()
        if not isinstance(dask_client, DaskClient):
            raise TypeError("Passed client is not a Dask Client object!")
        self.dask_client = dask_client
        self.client = self.dask_client.getClient()
        # List containing futures to vectors
        self.vecDask = []
        # Getting worker IDs
        wrkIds = self.dask_client.getWorkerIds()
        N_wrk = self.dask_client.getNworkers()
        if "vector_template" in kwargs and "chunks" in kwargs:
            vec_tmplt = kwargs.get("vector_template")
            self.chunks = kwargs.get("chunks")
            # Spreading chunks across available workers
            self. chunks = [np.sum(ix) for ix in np.array_split(self.chunks, N_wrk)]
            # Checking if an SepVector was passed (by getting Hypercube)
            hyper = False
            if SepVector:
                if isinstance(vec_tmplt, SepVector.vector):
                    hyper = True
            # Broadcast vector space
            if hyper:
                vec_space = vec_tmplt.getHyper().axes  # Passing axes since Hypercube cannot be serialized
            else:
                vec_space = vec_tmplt.cloneSpace()
            vec_spaceD = self.client.scatter(vec_space, workers=wrkIds)
            daskD.wait(vec_spaceD)
            # Spreading vectors
            for iwrk, wrkId in enumerate(wrkIds):
                for ivec in range(self.chunks[iwrk]):
                    if hyper:
                        # Instantiating Sep vectors on remote machines
                        self.vecDask.append(
                            self.client.submit(call_constr_hyper, vec_spaceD, workers=[wrkId], pure=False))
                    else:
                        # Scattering vector to different workers
                        self.vecDask.append(
                            self.client.submit(_call_clone, vec_spaceD, workers=[wrkId], pure=False))
        elif "vectors" in kwargs:
            # Vector list to be spread across workers
            vec_list = kwargs.get("vectors")
            copy = kwargs.get("copy", True)
            self.chunks = kwargs.get("chunks", None)
            if self.chunks is None:
                # Spread vectors evenly
                vec_chunks = np.array_split(vec_list, N_wrk)
            else:
                # Spread according to chunk size
                if len(vec_list) != np.sum(self.chunks):
                    raise ValueError("Total number of vectors in chunks not consistent with number of vectors!")
                # Spreading chunks across available workers
                self.chunks = [np.sum(ix) for ix in np.array_split(self.chunks, N_wrk)]
                vec_chunks = np.split(vec_list, np.cumsum(self.chunks))[:-1]
            # Spreading vectors
            for iwrk, wrkId in enumerate(wrkIds):
                for vec in vec_chunks[iwrk]:
                    # Checking if an SepVector was passed
                    IsSepVec = False
                    if SepVector:
                        if isinstance(vec, SepVector.vector):
                            IsSepVec = True
                    if IsSepVec:
                        # Instantiating Sep vectors on remote machines
                        self.vecDask.append(
                            self.client.submit(call_constr_hyper, vec.getHyper().axes, workers=[wrkId], pure=False))
                        # Copying values from NdArray (Cannot scatter SepVector)
                        daskD.wait(self.vecDask[-1])
                        if copy:
                            arrD = self.client.scatter(vec.getNdArray(), workers=[wrkId])
                            daskD.wait(arrD)
                            daskD.wait(
                                self.client.submit(_copy_from_NdArray, self.vecDask[-1], arrD, pure=False))
                    else:
                        if copy:
                            self.vecDask.append(self.client.scatter(vec, workers=[wrkId]))
                        else:
                            self.vecDask.append(
                                self.client.submit(_call_clone, vec.cloneSpace(), workers=[wrkId], pure=False))
        elif "dask_vectors" in kwargs:
            dask_vectors = kwargs.get("dask_vectors")
            for dask_vec in dask_vectors:
                if not issubclass(dask_vec.type, Vector):
                    raise TypeError("One instance in dask_vectors is not a vector-derived object!")
            self.dask_client = dask_client
            self.client = self.dask_client.getClient()
            self.vecDask = dask_vectors
        else:
            raise ValueError("Wrong arguments passed to constructor! Please, read object help!")
        # Waiting vectors to be instantiated
        daskD.wait(self.vecDask)
        return
    
    # Class vector operations
    def getNdArray(self):
        """
        Function to return a list of all the arrays of the vector
        """
        futures = self.client.map(_call_getNdArray, self.vecDask, pure=False)
        arrays = self.client.gather(futures)
        return arrays
    
    @property
    def shape(self):
        futures = self.client.map(_call_shape, self.vecDask, pure=False)
        shapes = self.client.gather(futures)
        return shapes
    
    @property
    def size(self):
        """Attribute of total number of elements in the vector"""
        futures = self.client.map(_call_size, self.vecDask, pure=False)
        sizes = self.client.gather(futures)
        return np.sum(sizes)
    
    @property
    def ndim(self):
        """Attribute of number of dimensions"""
        futures = self.client.map(_call_ndim, self.vecDask, pure=False)
        ndims = self.client.gather(futures)
        return ndims
    
    def norm(self, N=2):
        """Function to compute vector N-norm"""
        norms = self.client.map(_call_norm, self.vecDask, N=N, pure=False)
        norm = 0.0
        for future, result in daskD.as_completed(norms, with_results=True):
            norm += np.power(np.float64(result), N)
        return np.power(norm, 1. / N)
    
    def zero(self):
        """Function to zero out a vector"""
        daskD.wait(self.client.map(_call_zero, self.vecDask, pure=False))
        return self
    
    def max(self):
        """Function to obtain maximum value within a vector"""
        maxs = self.client.map(_call_max, self.vecDask, pure=False)
        max_val = - np.inf
        for future, result in daskD.as_completed(maxs, with_results=True):
            if result > max_val:
                max_val = result
        return max_val
    
    def min(self):
        """Function to obtain minimum value within a vector"""
        mins = self.client.map(_call_min, self.vecDask, pure=False)
        min_val = np.inf
        for future, result in daskD.as_completed(mins, with_results=True):
            if result < min_val:
                min_val = result
        return min_val
    
    def set(self, val):
        """Function to set all values in the vector"""
        daskD.wait(self.client.map(_call_set, self.vecDask, val=val, pure=False))
        return self
    
    def scale(self, sc):
        """Function to scale a vector"""
        daskD.wait(self.client.map(_call_scale, self.vecDask, sc=sc, pure=False))
        return self
    
    def addbias(self, bias):
        """Function to add bias to a vector"""
        daskD.wait(self.client.map(_call_addbias, self.vecDask, bias=bias, pure=False))
        return self
    
    def rand(self):
        """Function to randomize a vector"""
        daskD.wait(self.client.map(_call_rand, self.vecDask, pure=False))
        return self
    
    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        vectors = self.client.map(_call_clone, self.vecDask, pure=False)
        daskD.wait(vectors)
        return DaskVector(self.dask_client, dask_vectors=vectors)
    
    def cloneSpace(self):
        """Function to clone vector space"""
        vectors = self.client.map(_call_cloneSpace, self.vecDask, pure=False)
        daskD.wait(vectors)
        return DaskVector(self.dask_client, dask_vectors=vectors)
    
    def checkSame(self, other):
        """Function to check to make sure the vectors exist in the same space"""
        checkVector(self, other)
        futures = self.client.map(_call_checkSame, self.vecDask, other.vecDask, pure=False)
        results = self.client.gather(futures)
        return all(results)
    
    def writeVec(self, filename, mode='w', multi_file=False):
        """
        Function to write vector to file:

        :param filename     : string - Filename to write the vector to
        :param mode         : string - Writing mode 'w'=overwrite file or 'a'=append to file ['w']
        :param multi_file   : boolean - If True multiple files will be written with suffix _chunk1,2,3,...;
                              otherwise, a single will be written [False]
        """
        # Check writing mode
        if not mode in 'wa':
            raise ValueError("Mode must be appending 'a' or writing 'w' ")
        # Multi-node writing mode
        Nvecs = len(self.vecDask)
        # Creating vector-chunk names
        vec_names = [os.getcwd() + "/" + "".join(filename.split('.')[:-1]) + "_chunk%s.H" % (ii + 1)
                     for ii in range(Nvecs)]
        futures = self.client.map(_call_writeVec, self.vecDask, vec_names, [mode] * Nvecs, pure=False)
        daskD.wait(futures)
        # Single-file writing mode (concatenating all binary files)
        if not multi_file:
            # Getting binary-file locations
            bin_files = [sep.get_binary(vec_name) for vec_name in vec_names]
            # Getting all-axis information
            ax_info = [sep.get_axes(vec_name)[:sep.get_num_axes(vec_name)] for vec_name in vec_names]
            binfile = sep.datapath + filename.split('/')[-1] + '@'
            # Checks for writing header file
            len_ax = [len(ax) for ax in ax_info]
            max_len_idx = np.argmax(len_ax)
            cat_axis_multi = len_ax[max_len_idx]  # Axis on with files are concatenated
            # Getting largest-vector-axis information
            main_axes = ax_info[max_len_idx]
            N_elements_multi = 0  # Number of elements on the concatenation axis of multifiles
            # Getting number of elements if appending mode is requested
            last_axis = [[1, 1.0, 1.0, "Undefined"]]
            if os.path.isfile(filename) and 'a' in mode:
                file_axes = sep.get_axes(filename)
                last_axis[0][0] += file_axes[cat_axis_multi][0]
            # Checking compatibility of vectors
            for axes2check in ax_info:
                # First checking for len of given axis
                Naxes = len(axes2check)
                if Naxes < cat_axis_multi - 1:
                    print("WARNING! Cannot write single file with given vector chunks: "
                          "number of axes not compatible. Wrote chunks!")
                    return
                for idx, ax in enumerate(axes2check):
                    if ax[0] != main_axes[idx][0] and idx != cat_axis_multi - 1:
                        print("WARNING! Cannot write single file with given vector chunks: "
                              "elements on axis number %s not compatible. Wrote chunks!" % (idx + 1))
                        return
                if Naxes == cat_axis_multi:
                    N_elements_multi += axes2check[cat_axis_multi - 1][
                        0]  # Adding number of elements on the given concatenation axis
                else:
                    N_elements_multi += 1  # Only one element present
            # Changing number of elements on last axis
            main_axes[-1][0] = N_elements_multi
            # Adding last appending axes if file existed
            main_axes += last_axis
            # Writing header file
            with open(filename, mode) as fid:
                for ii, ax in enumerate(main_axes):
                    ax_id = ii + 1
                    fid.write("n%s=%s o%s=%s d%s=%s label%s='%s'\n" % (
                        ax_id, ax[0], ax_id, ax[1], ax_id, ax[2], ax_id, ax[3]))
                fid.write("in='%s'\n" % binfile)
                fid.write("esize=4\n")
                fid.write("data_format=\"native_float\"\n")
            # Writing binary file ("reading each binary file by chuncks of BUF_SIZE")
            with open(binfile, mode + 'b') as fid:
                for binfile_ii in bin_files:
                    with open(binfile_ii, 'rb') as fid_toread:
                        while True:
                            data = fid_toread.read(BUF_SIZE)
                            if not data: break
                            fid.write(data)
            # Removing header binary files associated to chunks
            for idx, vec_name in enumerate(vec_names):
                os.remove(vec_name)
                os.remove(bin_files[idx])
        return
    
    def abs(self):
        """Return a vector containing the absolute values"""
        daskD.wait(self.client.map(_call_abs, self.vecDask, pure=False))
        return self
    
    def sign(self):
        """Return a vector containing the signs"""
        daskD.wait(self.client.map(_call_sign, self.vecDask, pure=False))
        return self
    
    def reciprocal(self):
        """Return a vector containing the reciprocals of self"""
        daskD.wait(self.client.map(_call_reciprocal, self.vecDask, pure=False))
        return self
    
    def conj(self):
        """Compute conjugate transpose of the vector"""
        daskD.wait(self.client.map(_call_conj, self.vecDask, pure=False))
        return self
    
    def real(self):
        """Return the real part of the vector"""
        daskD.wait(self.client.map(_call_real, self.vecDask, pure=False))
        return self
    
    def imag(self):
        """Return the imaginary part of the vector"""
        daskD.wait(self.client.map(_call_imag, self.vecDask, pure=False))
        return self
    
    def pow(self, power):
        """Compute element-wise power of the vector"""
        daskD.wait(self.client.map(_call_pow, self.vecDask, power=power, pure=False))
        return self
    
    # Methods combinaning different vectors
    
    def maximum(self, vec2):
        """Return a new vector of element-wise maximum of self and vec2"""
        checkVector(self, vec2)
        daskD.wait(self.client.map(_call_maximum, self.vecDask, vec2.vecDask, pure=False))
        return self
    
    def copy(self, other):
        """Function to copy vector"""
        checkVector(self, other)
        daskD.wait(self.client.map(_call_copy, self.vecDask, other.vecDask, pure=False))
        return self
    
    def scaleAdd(self, other, sc1=1.0, sc2=1.0):
        """Function to scale two vectors and add them to the first one"""
        checkVector(self, other)
        sc1 = [sc1] * len(self.vecDask)
        sc2 = [sc2] * len(self.vecDask)
        futures = self.client.map(_call_scaleAdd, self.vecDask, other.vecDask, sc1, sc2,
                                  pure=False)
        daskD.wait(futures)
        return self
    
    def dot(self, other):
        """Function to compute dot product between two vectors"""
        checkVector(self, other)
        dots = self.client.map(_call_dot, self.vecDask, other.vecDask, pure=False)
        # Adding all the results together
        dot = 0.0
        for future, result in daskD.as_completed(dots, with_results=True):
            dot += np.float64(result)
        return dot
    
    def multiply(self, other):
        """Function to multiply element-wise two vectors"""
        checkVector(self, other)
        futures = self.client.map(_call_multiply, self.vecDask, other.vecDask, pure=False)
        daskD.wait(futures)
        return self
    
    def isDifferent(self, vec2):
        """Function to check if two vectors are identical"""
        checkVector(self, vec2)
        futures = self.client.map(_call_isDifferent, self.vecDask, vec2.vecDask,
                                  pure=False)
        results = self.client.gather(futures)
        return any(results)
    
    def clipVector(self, low, high):
        """Function to bound vector values based on input vectors min and max"""
        checkVector(self, low)  # Checking low-bound vector
        checkVector(self, high)  # Checking high-bound vector
        futures = self.client.map(_call_clipVector, self.vecDask, low.vecDask,
                                  high.vecDask, pure=False)
        daskD.wait(futures)
        return self


# DASK I/O TO READ LARGE-SCALE VECTORS DIRECTLY WITHIN EACH WORKER
def _get_binaries(**kwargs):
    """
    Function to obtain associated binary files to each file name
    :param filenames: list; List/Array containing file names to read
    :return:
    binfiles: list; List containing binary files associated to each file
    Nbytes: list; List containing the number of bytes within binary files
    """
    binfiles = list()
    Nbytes = list()
    filenames = kwargs.get("filenames")
    for filename in filenames:
        _, ext = os.path.splitext(filename)  # Getting file extension
        if ext == ".H":  # SEPlib file
            binfiles.append(sep.get_binary(filename))
            Nbytes.append(os.path.getsize(binfiles[-1]))
        elif ext == ".h5":
            raise NotImplementedError("ERROR! h5 files not supported yet.")
        else:
            raise ValueError("ERROR! Unknown format for file %s" % filename)
    return binfiles, Nbytes


def _set_binfiles(binfiles, Nbytes, **kwargs):
    """
    Function to associate binary file/s for each vector
    :param shapes: list/array; List/Array containing the shape of each chunk to be read
    :param binfiles: list; List containing the number of bytes within binary files
    :param Nbytes: list; List containing the number of bytes within binary files
    :param format: string; Kind of binary format to read ['>f']
    :return:
    bin_chunks: list of lists; binary files associated to each vector
    counts: list of lists; number of bytes to read per binary file
    offsets: list of lists; offset to apply when reading given binary file
    """
    shapes = kwargs.get("shapes")
    fmt = kwargs.get("format", ">f")  # Default floating point number
    esize = np.dtype(fmt).itemsize  # Byte size per vector element
    # Setting bin_chunks, counts, and offsets
    bin_chunks = list()
    counts = list()
    offsets = list()
    # Checking total bytes and number of elements
    totalBytes = np.int(np.sum(Nbytes))
    rqstBytes = 0
    for shp in shapes:
        rqstBytes += np.prod(shp) * esize
    if rqstBytes > totalBytes:
        raise ValueError(
            "ERROR! Total number of bytes needed to be read (%d) greater than bytes in provided file list (%d)"
            % (rqstBytes, totalBytes))
    bytesRd = 0  # Number of bytes read so far from a given file
    for shp in shapes:
        Nelmnt = np.prod(shp)  # Total number of elements within vector
        NelmntBt = Nelmnt * esize  # Number of bytes necessary for given vector
        tmp_bin_files = list()
        tmp_counts = list()
        tmp_offsets = list()
        for _ in range(len(binfiles)):
            tmp_bin_files.append(binfiles[0])
            if NelmntBt >= Nbytes[0]:
                # Entire binary file must be read
                tmp_counts.append(np.int(Nbytes[0] / esize))
                tmp_offsets.append(bytesRd)
                # Updating variables
                bytesRd = 0
                NelmntBt -= Nbytes[0]  # Subtracting bytes already read
                # Removing the current binary file (completely read)
                binfiles.pop(0)
                Nbytes.pop(0)
            else:
                # Only part of the file needs to be read
                tmp_counts.append(np.int(NelmntBt / esize))
                tmp_offsets.append(bytesRd)
                bytesRd += NelmntBt  # Number of bytes read
                Nbytes[0] -= NelmntBt
                NelmntBt = 0  # All necessary bytes are set
            if NelmntBt == 0:
                break  # Read all the elements of a vector
        # For a given vector, placing information on bin files, counts, offset
        bin_chunks.append(tmp_bin_files)
        counts.append(tmp_counts)
        offsets.append(tmp_offsets)
    return bin_chunks, counts, offsets


def _read_vector_dask(shape, binaries, counts, offsets, **kwargs):
    """
    Function to read a vector using a Dask worker
    :param shape : - list/array; Shape of the vector to be instantiated
    :param binaries : - list; Binary files to read to instantiate a vector
    :param counts : - list; Number of elements to read per binary file
    :param offsets : -list; Bytes to be skipped when reading given file
    :return:
    vector - vector class; vector instance creates using
    """
    vector = None
    vtype = kwargs.get("vtype")
    axes = kwargs.get("axes", None)
    fmt = kwargs.get("format", ">f")  # Default floating point number
    # Reading binary data into data array
    data = np.array([])  # Initialize an empty array
    for ii, filename in enumerate(binaries):
        _, ext = os.path.splitext(filename)  # Getting file extension
        if ext == ".H@":
            fid = open(filename, 'r+b')
            # Default formatting big-ending floating point number
            data = np.append(data, np.fromfile(fid, count=counts[ii], offset=offsets[ii], dtype=fmt))
            fid.close()
        else:
            raise ValueError("ERROR! Unknown extension for binary file: %s" % filename)
    # Reshaping array and forcing memory continuity
    data = np.ascontiguousarray(np.reshape(data, shape))
    if vtype == "VectorNumpy":
        vector = VectorNumpy(data)
    elif vtype == "SepVector":
        if SepVector:
            if axes:
                # Instantiate using axis
                vector = SepVector.getSepVector(axes=axes)
            else:
                # Instantiate using shape
                shape.reverse()  # Reversing order of axis
                vector = SepVector.getSepVector(ns=shape)
            vector.getNdArray()[:] = data  # Copying data into SepVector instance
            del data  # Removing data
        else:
            raise ImportError("ERROR! SepVector module not found!")
    else:
        raise ValueError("ERROR! Unknown vtype (%s)" % vtype)
    return vector


def readDaskVector(dask_client, **kwargs):
    """
    Function to read files in parallel and store within Dask vector
    :param dask_client : - DaskClient; client object to use when submitting tasks (see dask_util module)
    :param filenames : - list; List/Array containing file names to read
    :param shapes : - list/array; List/Array containing the shape of each chunk to be read
    :param chunks : - list/array; List/Array defining how each vector should be spread. Note that len(chunks) must be
                                  equal to dask_client.getNworkers(), len(shape) must be equal np.sum(chunks)
    :param vtype : - string; Type of vectors to be instantiated. Supported (VectorNumpy, SepVector)
    :return:
    daskVec - Dask Vector object
    """
    # Getting dask client components
    Nwrks = dask_client.getNworkers()
    client = dask_client.getClient()
    wrkIds = dask_client.getWorkerIds()
    # Args
    shapes = kwargs.get("shapes")
    chunks = kwargs.get("chunks")
    vtype = kwargs.get("vtype")
    axes = kwargs.get("axes", None)  # Necessary for SepVector
    # Q/C steps
    if len(chunks) != Nwrks:
        raise ValueError(
            "ERROR! Number of workers in client (%d) inconsistent with chunks length (%d)" % (Nwrks, len(chunks)))
    if np.sum(chunks) != len(shapes):
        raise ValueError(
            "ERROR! Total number of chunks (%s) inconsistent with shapes length (%d)" % (np.sum(chunks), len(shapes)))
    if vtype not in "VectorIC SepVector":
        raise ValueError("ERROR! Provided vtype (%s) not currently supported" % vtype)
    if axes:
        if len(axes) != len(shapes):
            raise ValueError(
                "ERROR! Length of axes (%s) not consistent with length of shapes (%s)" % (len(axes), len(shapes)))
    # Pre-processing: for each chunk associate necessary files, bytes to read (count), offset (if file goes onto two
    # or more chunks).
    # Get binary files (Necessary to use header-based formats)
    binfiles, Nbytes = _get_binaries(**kwargs)
    # Associate binary files with each vector using shapes and
    # set count and offset for each of them
    bin_shps, counts, offsets = _set_binfiles(binfiles, Nbytes, **kwargs)
    # Split lists/arrays into chunks for parallel processing
    shps2read = list()
    bin2read = list()
    count2read = list()
    offset2read = list()
    # Preprocessing for vector-specific arguments
    # must be done in the next for loop
    axes2read = list()  # List of axes
    for chunk in chunks:
        shps2read.append(shapes[:chunk])
        bin2read.append(bin_shps[:chunk])
        count2read.append(counts[:chunk])
        offset2read.append(offsets[:chunk])
        del shapes[:chunk], bin_shps[:chunk], counts[:chunk], offsets[:chunk]
        # Checking if axis list was provided
        if axes and vtype == "SepVector":
            axes2read.append(axes[:chunk])
            del axes[:chunk]
    # Loop over workers/chunks
    # Read binary files and place within vector objects
    fut_vec = []
    for iWrk, wrkId in enumerate(wrkIds):
        for ivec in range(len(shps2read[iWrk])):
            if len(axes2read) > 0 and vtype == "SepVector":
                kwargs.update({"axes": axes2read[iWrk][ivec]})
            fut_vec.append(client.submit(_read_vector_dask, shps2read[iWrk][ivec], bin2read[iWrk][ivec],
                                         count2read[iWrk][ivec], offset2read[iWrk][ivec], **kwargs,
                                         workers=[wrkId], pure=False))
    # Waiting for all vector chunks to be instantiated
    daskD.wait(fut_vec)
    # Checking for errors
    for idx, vec in enumerate(fut_vec):
        if vec.status == 'error':
            print("Error for when instantiating vector %s" % idx)
            print(vec.result())
    daskVec = DaskVector(dask_client, dask_vectors=fut_vec)
    return daskVec
