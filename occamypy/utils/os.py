# Module containing useful function to interact with the system
import subprocess
import sys
import os
import hashlib
import random
import string
from importlib.util import find_spec
from .logger import Logger


CUPY_ENABLED = find_spec("cupy") is not None

debug = False  # Debug flag for printing screen output of RunShellCmd as it runs commands
debug_log = None  # File where debug outputs are written if requested (it must be a logger object)
BUF_SIZE = 8388608  # read binary files in 64Mb chunks!
DEVNULL = open(os.devnull, 'wb')


def mkdir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def rand_name(N: int = 6) -> str:
    """function returning random sequence of N letters and numbers"""
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(N))


def hashfile(binfile):
    """Function hashing a binary file. It uses a BUF_SIZE to partially store file in memory
    and do not completely load the file into the RAM"""
    md5 = hashlib.md5()
    with open(binfile, 'rb') as fid:
        while True:
            data = fid.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def RunShellCmd(cmd, print_cmd=False, print_output=False, synch=True, check_code=True, get_stat=True, get_output=True):
    """Function to run a Shell command through python, return code and """
    # Overwrites any previous definition (when used within other programs)
    global debug, debug_log
    # Running command synchronously or asynchronously?
    if synch:
        if debug:
            print_cmd = True
            print_output = True
            # Printing command to be run if requested
        info = "RunShellCmd running: \'%s\'" % cmd
        if isinstance(debug_log, Logger):
            debug_log.addToLog(info)
        if print_cmd:
            print(info)
        # Starting the process (Using PIPE to streaming output)
        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        # Creating Stdout to save command output
        stdout = []
        # Streaming the stdout to screen if requested
        while True:
            line = proc.stdout.readline()
            if line == '' and proc.poll() is not None:
                break
            else:
                stdout.append(line)
                line = line.rstrip()
                if line != '':
                    # Print to debug file?
                    if isinstance(debug_log, Logger):
                        debug_log.addToLog(line)
                    # Print to screen?
                    if print_output:
                        print(line)
                    sys.stdout.flush()
                proc.stdout.flush()
        stdout = ''.join(stdout)
    else:
        # Running process asynchronously (Avoiding PIPE and removing standard output)
        global DEVNULL
        proc = subprocess.Popen([cmd], stdout=DEVNULL, shell=True, universal_newlines=True)
        return proc, "Running command asynchronously, returning process"
    # Command has finished, checking error code and returning requested variables
    err_code = proc.poll()
    return_var = []
    # Returning error code or status
    if get_stat:
        return_var.append(err_code)
    # Returning output
    if get_output:
        return_var.append(stdout)
    # Checking error code
    if check_code and err_code != 0:
        # Writing error code to debug file if any
        info = "ERROR! Command failed: %s; Error code: %s" % (cmd, err_code)
        if isinstance(debug_log, Logger):
            debug_log.addToLog(info)
        raise SystemError("ERROR! Command failed: %s; Error code: %s; Output: %s" % (cmd, err_code, stdout))
    # Returning
    return return_var
