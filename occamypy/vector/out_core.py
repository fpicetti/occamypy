import numpy as np
import os
from time import time
from copy import deepcopy
from shutil import copyfile

from .base import Vector
from occamypy.utils import sep
from occamypy.utils.os import RunShellCmd, hashfile, BUF_SIZE

from re import compile
re_dpr = compile("DOT RESULT(.*)")


class VectorOC(Vector):
    """Out-of-core python vector class"""
    
    def __init__(self, in_content):
        """VectorOC constructor: input= numpy array, header file, vectorIC"""
        # Verify that input is a numpy array or header file or vectorOC
        super(VectorOC).__init__()
        if isinstance(in_content, Vector):
            # VectorIC passed to constructor
            # Placing temporary file into datapath folder
            tmp_vec = sep.datapath + "tmp_vectorOC" + str(int(time() * 1000000)) + ".H"
            sep.write_file(tmp_vec, in_content.getNdArray(), in_content.ax_info)
            self.vecfile = tmp_vec  # Assigning internal vector array
            # Removing header file? (Default behavior is to remove temporary file)
            self.remove_file = True
        elif isinstance(in_content, np.ndarray):
            # Numpy array passed to constructor
            tmp_vec = sep.datapath + "tmp_vectorOC" + str(int(time() * 1000000)) + ".H"
            sep.write_file(tmp_vec, in_content)
            self.vecfile = tmp_vec  # Assigning internal vector array
            # Removing header file? (Default behavior is to remove temporary file)
            self.remove_file = True
        elif isinstance(in_content, str):
            # Header file passed to constructor
            self.vecfile = in_content  # Assigning internal vector array
            # Removing header file? (Default behavior is to preserve user file)
            self.remove_file = False
        else:
            # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")
        # Assigning binary file pointer
        self.binfile = sep.get_binary(self.vecfile)
        # Number of axes integer
        self.ndim = sep.get_num_axes(self.vecfile)
        # Number of elements per axis (tuple)
        axes_info = sep.get_axes(self.vecfile)
        axis_elements = tuple([ii[0] for ii in axes_info[:self.ndim]])
        self.shape = tuple(reversed(axis_elements))
        self.size = np.product(self.shape)
        self.ndim = len(self.shape)
        return
    
    def __del__(self):
        """VectorOC destructor"""
        if self.remove_file:
            # Removing both header and binary files (using os.system to make module compatible with python3.5)
            os.system("rm -f %s %s" % (self.vecfile, self.binfile))
        return
    
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        ndarray, _ = sep.read_file(self.vecfile)
        return ndarray
    
    def norm(self, N=2):
        """Function to compute vector N-norm"""
        if N != 2:
            raise NotImplementedError("Norm different than L2 not currently supported")
        # Running Solver_ops to compute norm value
        find = re_dpr.search(RunShellCmd("Solver_ops file1=%s op=dot" % self.vecfile, get_stat=False)[0])
        if find:
            return np.sqrt(float(find.group(1)))
        else:
            raise ValueError("ERROR! Trouble parsing dot product!")
        return
    
    def zero(self):
        """Function to zero out a vector"""
        RunShellCmd("head -c %s </dev/zero > %s" % (self.size * 4, self.binfile),
                    get_stat=False, get_output=False)
        # sys_util.RunShellCmd("Solver_ops file1=%s op=zero"%(self.vecfile),get_stat=False,get_output=False)
        return
    
    def scale(self, sc):
        """Function to scale a vector"""
        RunShellCmd("Solver_ops file1=%s scale1_r=%s op=scale" % (self.vecfile, sc),
                    get_stat=False, get_output=False)
        return
    
    def rand(self, snr=1.0):
        """Fill vector with random number (~U[1,-1]) with a given SNR"""
        # Computing RMS amplitude of the vector
        rms = RunShellCmd("Attr < %s want=rms param=1 maxsize=5000" % (self.vecfile), get_stat=False)[0]
        rms = float(rms.split("=")[1])  # Standard deviation of the signal
        amp_noise = 1.0
        if rms != 0.:
            amp_noise = np.sqrt(3.0 / snr) * rms  # sqrt(3*Power_signal/SNR)
        # Filling file with random number with the proper scale
        RunShellCmd("Noise file=%s rep=1 type=0 var=0.3333333333; Solver_ops file1=%s scale1_r=%s op=scale" % (
            self.vecfile, self.vecfile, amp_noise), get_stat=False, get_output=False)
        return
    
    def clone(self):
        """Function to clone (deep copy) a vector or from a space and creating a copy of the associated header file"""
        # First performing a deep copy of the vector
        vec_clone = deepcopy(self)
        if vec_clone.vecfile is None:
            # Creating header and binary files from vector space
            # Placing temporary file into datapath folder
            tmp_vec = sep.datapath + "clone_tmp_vector" + str(int(time() * 1000000)) + ".H"
            axis_file = ""
            for iaxis, naxis in enumerate(tuple(reversed(vec_clone.shape))):
                axis_file += "n%s=%s " % (iaxis + 1, naxis)
            # Creating temporary vector file
            cmd = "Spike %s | Add scale=0.0 > %s" % (axis_file, tmp_vec)
            RunShellCmd(cmd, get_stat=False, get_output=False)
            vec_clone.vecfile = tmp_vec
            vec_clone.binfile = sep.get_binary(vec_clone.vecfile)
        else:
            # Creating a temporary file with similar name but computer time at the end
            tmp_vec = self.vecfile.split(".H")[0].split("/")[-1]  # Getting filename only
            # Placing temporary file into datapath folder
            tmp_vec = sep.datapath + tmp_vec + "_clone_" + str(int(time() * 1000000)) + ".H"
            tmp_bin = tmp_vec + "@"
            # Copying header and binary files and setting pointers to new file
            copyfile(self.vecfile, tmp_vec)  # Copying header
            copyfile(self.binfile, tmp_bin)  # Copying binary
            vec_clone.vecfile = tmp_vec
            vec_clone.binfile = tmp_bin
            # "Fixing" header file
            with open(vec_clone.vecfile, "a") as fid:
                fid.write("in='%s\n'" % tmp_bin)
        # By default the clone file is going to be removed once the vector is deleted
        vec_clone.remove_file = True
        return vec_clone
    
    def cloneSpace(self):
        """Function to clone vector space only (vector without actual vector binary file by using None values)"""
        vec_space = VectorOC(self.vecfile)
        # Removing header vector file
        vec_space.vecfile = None
        vec_space.binfile = None
        vec_space.remove_file = False
        return vec_space
    
    def checkSame(self, other):
        """Function to check dimensionality of vectors"""
        return self.shape == other.shape
    
    def writeVec(self, filename, mode='w'):
        """Function to write vector to file"""
        # Check writing mode
        if not mode in 'wa':
            raise ValueError("Mode must be appending 'a' or writing 'w' ")
        # writing header/pointer file if not present and not append mode
        if not (os.path.isfile(filename) and mode in 'a'):
            binfile = sep.datapath + filename.split('/')[-1] + '@'
            # Copying SEPlib header file
            copyfile(self.vecfile, filename)
            # Substituting binary file
            with open(filename, 'a') as fid:
                fid.write("\nin='%s'\n" % binfile)
            fid.close()
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
                with open(filename, mode) as fid:
                    fid.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (append_dim, n_vec + 1, append_dim, append_dim))
                fid.close()
        # Writing or Copying binary file
        if not (os.path.isfile(binfile) and mode in 'a'):
            copyfile(self.binfile, binfile)
        else:
            # Writing file if
            with open(binfile, mode + 'b') as fid, open(self.binfile, 'rb') as fid_toread:
                while True:
                    data = fid_toread.read(BUF_SIZE)
                    if not data:
                        break
                    fid.write(data)
            fid.close()
            fid_toread.close()
        return
    
    def copy(self, other):
        """Function to copy vector from input vector"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorOC):
            raise TypeError("ERROR! Provided input vector not a %s!" % self.whoami)
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError(
                "ERROR! Vector dimensionality mismatching: self = %s; other = %s" % (self.shape, other.shape))
        # Copy binary file of input vector
        copyfile(other.binfile, self.binfile)  # Copying binary
        return
    
    def scaleAdd(self, other, sc1=1.0, sc2=1.0):
        """Function to scale a vector"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorOC):
            raise TypeError("ERROR! Provided input vector not a vectorOC!")
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError(
                "ERROR! Vector dimensionality mismatching: self = %s; other = %s" % (self.shape, other.shape))
        # Performing scaling and addition
        cmd = "Solver_ops file1=%s scale1_r=%s file2=%s scale2_r=%s op=scale_addscale" % (
            self.vecfile, sc1, other.vecfile, sc2)
        RunShellCmd(cmd, get_stat=False, get_output=False)
        return
    
    def dot(self, other):
        """Function to compute dot product between two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorOC):
            raise TypeError("ERROR! Provided input vector not a %s!" % self.whoami)
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("ERROR! Vector size mismatching: self = %s; other = %s" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError(
                "ERROR! Vector dimensionality mismatching: self = %s; other = %s" % (self.shape, other.shape))
        # Running Solver_ops to compute norm value
        cmd = "Solver_ops file1=%s file2=%s op=dot" % (self.vecfile, other.vecfile)
        find = re_dpr.search(RunShellCmd(cmd, get_stat=False)[0])
        if find:
            return float(find.group(1))
        else:
            raise ValueError("ERROR! Trouble parsing dot product!")
        return float(out_dot)
    
    def multiply(self, other):
        """Function to multiply element-wise two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorOC):
            raise TypeError("ERROR! Provided input vector not a %s!" % self.whoami)
        # Checking size (must have same number of elements)
        if self.size != other.size:
            raise ValueError("ERROR! Vector size mismatching: self = %s; other = %s" % (self.size, other.size))
        # Checking dimensionality
        if not self.checkSame(other):
            raise ValueError(
                "ERROR! Vector dimensionality mismatching: self = %s; other = %s" % (self.shape, other.shape))
        # Performing scaling and addition
        cmd = "Solver_ops file1=%s file2=%s op=multiply" % (self.vecfile, other.vecfile)
        RunShellCmd(cmd, get_stat=False, get_output=False)
        return
    
    def isDifferent(self, other):
        """Function to check if two vectors are identical using M5 hash scheme"""
        # Checking whether the input is a vector or not
        if not isinstance(other, VectorOC):
            raise TypeError("ERROR! Provided input vector not a %s!" % self.whoami)
        hashmd5_self = hashfile(self.binfile)
        hashmd5_other = hashfile(other.binfile)
        return hashmd5_self != hashmd5_other
