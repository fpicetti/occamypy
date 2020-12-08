# Module containing useful function to interact with sep files
import re
import os
import numpy as np
from occamypy.utils.os import RunShellCmd

# Assigning datapath
HOME = os.environ["HOME"]
datapath = None
# Checking environment definition first
if "DATAPATH" in os.environ:
    datapath = os.environ["DATAPATH"]
# Checking local directory
elif os.path.isfile('.datapath'):
    out = (RunShellCmd("cat .datapath | head -n 1", check_code=False, get_stat=False)[0]).rstrip()
    datapath = out.split("=")[1]
# Checking whether the local host has a datapath
else:
    if os.path.isfile(HOME + "/.datapath"):
        out = RunShellCmd("cat $HOME/.datapath | grep $HOST", check_code=False, get_stat=False)[0]
        if len(out) == 0:
            out = (RunShellCmd("cat $HOME/.datapath | head -n 1", check_code=False, get_stat=False)[0]).rstrip()
        datapath = out.split("=")[1]

# Checking if datapath was found
if datapath is None:
    if os.path.isdir("/tmp/"):
        datapath = "/tmp/"
        print("WARNING! DATAPATH not found. The folder /tmp will be used to write binary files")
    else:
        raise IOError("SEP datapath not found\n Set env variable DATAPATH to a folder to write binary files")


def rm_file(filename):
    """File to remove header and binary files"""
    binfile = get_binary(filename)
    if os.path.isfile(filename):
        os.remove(filename)
    if os.path.isfile(binfile):
        os.remove(binfile)
    return


def get_par(filename, par):
    """ Function to obtain a header parameter within the passed header file"""
    info = None
    # Checking if label is requested
    if "label" in par:
        reg_prog = re.compile("%s=(\'(.*?)\'|\"(.*?)\")" % par)
    else:
        reg_prog = re.compile("%s=([^\s]+)" % par)
    if not os.path.isfile(filename):
        raise OSError("ERROR! No %s file found!" % filename)
    for line in reversed(open(filename).readlines()):
        if info is None:
            find = reg_prog.search(line)
            if find:
                info = find.group(1)
    if info is None:
        raise IOError("%s parameter not found in file %s" % (filename, par))
    # Removing extra characters from found parameter
    if info is not None:
        info = info.replace('"', '')
        info = info.replace('\'', '')
    return info


def get_binary(filename):
    """ Function to obtain binary file associated with a given header file"""
    return get_par(filename, "in")


def get_axes(filename):
    """Function returning all axis information related to a header file"""
    axes = []
    # Currently handling maximum 7 axis
    for iaxis in range(7):
        # Obtaining number of elements within each axis
        try:
            axis_n = int(get_par(filename, par="n%s" % (iaxis + 1)))
        except IOError as exc:
            if iaxis == 0:
                print(exc.args)
                print("ERROR! First axis parameters must be found! Returning None")
                return None
            else:
                # Default value for an unset axis
                axis_n = 1
        # Obtaining origin of each axis
        try:
            axis_o = float(get_par(filename, par="o%s" % (iaxis + 1)))
        except IOError as exc:
            if iaxis == 0:
                print(exc.args)
                print("ERROR! First axis parameters must be found! Returning None")
                return None
            else:
                # Default value for an unset axis
                axis_o = 0.0
        # Obtaining sampling of each axis
        try:
            axis_d = float(get_par(filename, par="d%s" % (iaxis + 1)))
        except IOError as exc:
            if iaxis == 0:
                print(exc.args)
                print("ERROR! First axis parameters must be found! Returning None")
                return None
            else:
                # Default value for an unset axis
                axis_d = 0.0
        # Obtaining label of each axis
        try:
            axis_lab = get_par(filename, par="label%s" % (iaxis + 1))
        except IOError as exc:
            # Default value for an unset axis
            axis_lab = "Undefined"
        axes.append([axis_n, axis_o, axis_d, axis_lab])
    return axes


def get_num_axes(filename):
    """Function to obtain number of axes in a header file"""
    # Obtaining elements in each dimensions
    axis_info = get_axes(filename)
    axis_elements = [ii[0] for ii in axis_info]
    index = [i for i, nelements in enumerate(axis_elements) if int(nelements) > 1]
    if index:
        n_axes = index[-1] + 1
    else:
        n_axes = 1
    return n_axes


def read_file(filename, formatting='>f', mem_order="C"):
    """Function for reading header files"""
    axis_info = get_axes(filename)
    n_axis = get_num_axes(filename)
    shape = [ii[0] for ii in axis_info]
    shape = shape[:n_axis]
    if mem_order == "C":
        shape = tuple(reversed(shape))
    elif mem_order != "F":
        raise ValueError("ERROR! %s not an supported array order" % mem_order)
    fid = open(get_binary(filename), 'r+b')
    # Default formatting big-ending floating point number
    data = np.fromfile(fid, dtype=formatting)
    # Reshaping array and forcing memory continuity
    if mem_order == "C":
        data = np.ascontiguousarray(np.reshape(data, shape, order=mem_order))
    else:
        data = np.asfortranarray(np.reshape(data, shape, order=mem_order))
    fid.close()
    return [data, axis_info]


def write_file(filename, data, axis_info=None, formatting='>f'):
    """Function for writing header files"""
    global datapath
    # write binary file
    binfile = datapath + filename.split('/')[-1] + '@'
    with open(binfile, 'w+b') as fid:
        # Default formatting big-ending floating point number
        if np.isfortran(data):  # Forcing column-wise binary writing
            data.flatten('F').astype(formatting).tofile(fid)
        else:
            data.astype(formatting).tofile(fid)
    fid.close()
    # If axis_info is not provided all the present axis are set to d=1.0 o=0.0 label='Undefined'
    if axis_info is None:
        naxis = data.shape
        if not np.isfortran(data):
            naxis = tuple(reversed(naxis))  # If C last axis is the "fastest"
        axis_info = [[naxis[ii], 0.0, 1.0, 'Undefined'] for ii in range(0, len(naxis))]
    # writing header/pointer file
    with open(filename, 'w') as fid:
        # Writing axis info
        for ii, ax_info in enumerate(axis_info):
            ax_id = ii + 1
            fid.write("n%s=%s o%s=%s d%s=%s label%s='%s'\n"
                      % (ax_id, ax_info[0], ax_id, ax_info[1], ax_id, ax_info[2], ax_id, ax_info[3]))
        fid.write("in='%s'\n" % binfile)
        fid.write("data_format='xdr_float'\n")
        fid.write("esize=4\n")
    fid.close()
    return
