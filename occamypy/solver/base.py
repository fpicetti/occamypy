import atexit
import datetime
import os
import pickle
import re
from copy import deepcopy
from shutil import rmtree

import numpy as np

from occamypy.vector.base import VectorSet
from occamypy.vector.out_core import VectorOC
from occamypy.problem.base import Problem
from occamypy.utils import mkdir, sep
from occamypy.utils.logger import Logger


class Solver:
    """Base solver class"""
    
    def __init__(self, stopper, logger: Logger = None, name: str = "Solver"):
        # Parameter for saving results
        self.save_obj = False
        self.save_res = False
        self.save_grad = False
        self.save_model = False
        self.flush_memory = False
        self.overwrite = True  # Flag to overwrite results if first time writing on disk
        self.prefix = None
        
        self.logger = logger

        self.stopper = stopper
        self.stopper.logger = logger
        
        # Iteration axis-sampling parameters
        self.iter_buffer_size = None
        self.iter_sampling = 1
        
        # Lists of the results (list and vector Sets)
        self.obj = list()
        self.obj_terms = list()
        self.model = list()
        self.res = list()
        self.grad = list()
        self.modelSet = VectorSet()
        self.resSet = VectorSet()
        self.gradSet = VectorSet()
        self.inv_model = None
        self.iter_written = 0
        
        self.name = name
        
        # Set Restart object
        self.restart = Restart()
        create_msg = False
        # Setting defaults for saving results
        self.setDefaults()
    
    def __str__(self):
        return self.name
    
    def setPrefix(self, prefix):
        """Mutator to change prefix and file names for saving inversion results"""
        self.prefix = prefix
        return
    
    def setDefaults(self, save_obj=False, save_res=False, save_grad=False, save_model=False, prefix=None,
                    iter_buffer_size=None, iter_sampling=1, flush_memory=False):
        """
        Function to set parameters for result saving.

        Args:
            save_obj: save objective function values into the list self.obj
            save_res: save residual vectors into the list self.res
            save_grad: save gradient vectors into the list self.grad
            save_model: save domain vectors into the list self.domain.
                It will also say the last inverted domain vector into self.inv_model
            prefix: prefix of the files in which requested results will be saved;
                If prefix is None, then nothing is going to be saved on disk
            iter_buffer_size: number of steps to save before flushing results to disk
                (by default the solver waits until all iterations are done)
            iter_sampling: sampling of the iteration axis
            flush_memory: keep results into the object lists or clean those once inversion is completed or results have been written on disk
        """
        
        # Parameter for saving results
        self.save_obj = save_obj  # Flag to save objective function value
        self.save_res = save_res  # Flag to save residual vector
        self.save_grad = save_grad  # Flag to save gradient vector
        self.save_model = save_model  # Flag to save model vector
        self.flush_memory = flush_memory  # Keep results in RAM or flush memory every time results are written on disk
        
        # Prefix of the saved files (if provided the results will be written on disk)
        self.prefix = prefix  # Prefix for saving inversion results on disk
        
        # Iteration axis-sampling parameters
        self.iter_buffer_size = iter_buffer_size  # Number of steps to save before flushing results to disk
        self.iter_sampling = iter_sampling  # Sampling of the iteration axis
        
        # Lists of the results (list and vector Sets)
        self.obj = np.array([])  # Array for objective function values
        self.obj_terms = np.array([])  # Array for objective function values for each terms
        self.model = list()  # List for model vectors (to save results in-core)
        self.res = list()  # List for residual vectors (to save results in-core)
        self.grad = list()  # List for gradient vectors (to save results in-core)
        self.modelSet = VectorSet()  # Set for model vectors
        self.resSet = VectorSet()  # Set for residual vectors
        self.gradSet = VectorSet()  # Set for gradient vectors
        self.inv_model = None  # Temporary saved inverted model
    
    def flush_results(self):
        """Flushing internal memory of the saved results"""
        # Lists of the results (list and vector Sets)
        self.obj = np.array([])  # Array for objective function values
        self.obj_terms = np.array([])  # Array for objective function values for each terms
        self.model = list()  # List for model vectors (to save results in-core)
        self.res = list()  # List for residual vectors (to save results in-core)
        self.grad = list()  # List for gradient vectors (to save results in-core)
        self.modelSet = VectorSet()  # Set for model vectors
        self.resSet = VectorSet()  # Set for residual vectors
        self.gradSet = VectorSet()  # Set for gradient vectors
        self.inv_model = None  # Temporary saved inverted model
    
    def get_restart(self, log_file):
        """
        Function to retrieve restart folder from log file. It enables the user to use restart flag on self.run().
        
        Args:
             log_file: path to file
        """
        restart_folder = None
        # Obtaining restart folder path
        reg_prog = re.compile("Restart folder: ([^\s]+)")
        if not os.path.isfile(log_file):
            raise OSError("ERROR! No %s file found!" % log_file)
        for line in reversed(open(log_file).readlines()):
            if restart_folder is None:
                find = reg_prog.search(line)
                if find:
                    restart_folder = find.group(1)
        # Setting restart folder if user needs to do so
        if restart_folder is not None:
            self.restart.restart_folder = restart_folder
        else:
            print("WARNING! No restart folder's path was found in %s" % log_file)
        return
    
    def save_results(self, iiter, problem, force_save: bool = False, force_write: bool = False, **kwargs):
        """
        Save results to disk
        
        Args:
            iiter: iteration index
            problem: problem that is being solved
            force_save: ignore iteration sampling
            force_write: force  writing on disk if necessary (e.g., last iteration)
            **kwargs:
                domain: problem solution vector to be saved and/or written
                obj: objective function to be saved
                obj_terms: if problem objective function has more than one term
        """
        if not isinstance(problem, Problem):
            raise TypeError("Input variable is not a Problem object")

        # Getting a model from arguments if provided (necessary to remove preconditioning)
        mod_save = kwargs.get("model", problem.get_model())
        # Obtaining objective function value
        objf_value = kwargs.get("obj", problem.get_obj(problem.get_model()))
        obj_terms = kwargs.get("obj_terms", problem.obj_terms) if "obj_terms" in dir(problem) else None
        
        # Save if it is forced to or if the solver hits a sampled iteration number
        # The objective function is saved every iteration if requested
        if self.save_obj:
            self.obj = np.append(self.obj, deepcopy(float(objf_value)))
            # Checking if the objective function has multiple terms
            if obj_terms is not None:
                if len(self.obj_terms) == 0:
                    # First time obj_terms are saved
                    self.obj_terms = np.expand_dims(np.append(self.obj_terms, [float(_) for _ in obj_terms]), axis=0)
                else:
                    self.obj_terms = np.append(self.obj_terms,
                                               np.expand_dims(np.array([float(_) for _ in obj_terms]), axis=0),
                                               axis=0)
        if iiter % self.iter_sampling == 0 or force_save:
            if self.save_model:
                self.modelSet.append(mod_save)
                # Storing model vector into a temporary vector
                del self.inv_model  # Deallocating previous saved model
                self.inv_model = mod_save.clone()
            if self.save_res:
                res_vec = problem.get_res(problem.get_model())
                self.resSet.append(res_vec)
            if self.save_grad:
                grad = problem.get_grad(problem.get_model())
                self.gradSet.append(grad)
        # Write on disk if necessary or requested
        self._write_steps(force_write)
        return
    
    def _write_steps(self, force_write=False):
        """
        Method to write inversion results on disk if forced to or if buffer is filled

        Args:
            force_write: True - write every step; False - write whe buffer_size has been fulfilled
        """
        # Save results if buffer size is hit
        save = True if force_write or (self.iter_buffer_size is not None and max(len(self.modelSet.vecSet),
                                                                                 len(self.resSet.vecSet),
                                                                                 len(self.gradSet.vecSet))
                                                                             >= self.iter_buffer_size) \
            else False
        
        # Overwriting results if first time to write on disk
        mode = "w" if self.overwrite else "a"
        
        if save:
            self.overwrite = False  # Written at least once; do not overwrite files
            # Getting current saved results into an in-core list
            if not self.flush_memory:
                self.model += self.modelSet.vecSet
                self.res += self.resSet.vecSet
                self.grad += self.gradSet.vecSet
            # Writing objective function value on disk if requested
            if self.save_obj and self.prefix is not None:  # todo change to numpy?
                obj_file = self.prefix + "_obj.H"  # File name in which the objective function is saved
                sep.write_file(obj_file, self.obj)
                # Writing each term of the objective function
                if len(self.obj_terms) != 0:
                    for iterm in range(self.obj_terms.shape[1]):
                        # File name in which the objective function is saved
                        obj_file = self.prefix + "_obj_comp%s.H" % (iterm + 1)
                        sep.write_file(obj_file, self.obj_terms[:, iterm])
            # Writing current inverted model and model vectors on disk if requested
            if self.save_model and self.prefix is not None:
                inv_mod_file = self.prefix + "_inv_mod.H"  # File name in which the current inverted model is saved
                model_file = self.prefix + "_model.H"  # File name in which the model vector is saved
                self.modelSet.writeSet(model_file, mode=mode)
                self.inv_model.writeVec(inv_mod_file, mode="w")  # Writing inverted model file
            # Writing gradient vectors on disk if requested
            if self.save_grad and self.prefix is not None:
                grad_file = self.prefix + "_gradient.H"  # File name in which the gradient vector is saved
                self.gradSet.writeSet(grad_file, mode=mode)
            # Writing residual vectors on disk if requested
            if self.save_res and self.prefix is not None:
                res_file = self.prefix + "_residual.H"  # File name in which the residual vector is saved
                self.resSet.writeSet(res_file, mode=mode)
    
    def run(self, problem, verbose: bool = False, restart: bool = False):
        """
        Solve the given problem
        
        Args:
            problem: problem to be solved
            verbose: verbosity flag
            restart: restart previous inversion from restart folder
        """
        raise NotImplementedError("Implement run Solver in the derived class.")


class Restart:
    """
    Restart a solver.run

    Attributes:
        par_dict: dictionary containing the solver parameters
        vec_dict: dictionary containing all the vectors the solver needs
        restart_folder: path/to/folder where the  previous run has been saved
    """
    
    def __init__(self):
        """Restart constructor"""
        self.par_dict = dict()
        self.vec_dict = dict()
        # Restart folder in case it is necessary to write restart
        now = datetime.datetime.now()
        restart_folder = sep.datapath + "restart_" + now.isoformat() + "/"
        restart_folder = restart_folder.replace(":", "-")
        self.restart_folder = restart_folder
        # Calling write_restart when python session dies
        atexit.register(self.write_restart)
    
    def save_vector(self, vec_name, vector_in):
        """Save vector for restarting"""
        # Deleting the vector if present in the dictionary
        element = self.vec_dict.pop(vec_name, None)
        if element:
            del element
        self.vec_dict.update({vec_name: vector_in.clone()})
    
    def retrieve_vector(self, vec_name):
        """Method to retrieve a vector from restart object"""
        return self.vec_dict[vec_name]
    
    def save_parameter(self, par_name, parameter_in):
        """Method to save parameters for restarting"""
        self.par_dict.update({par_name: parameter_in})
        return
    
    def retrieve_parameter(self, par_name):
        """Method to retrieve a parameter from restart object"""
        return self.par_dict[par_name]
    
    def write_restart(self):
        """Restart destructor: it will write vectors on disk if the solver breaks"""
        if bool(self.par_dict) or bool(self.vec_dict):
            # Creating restarting directory
            mkdir(self.restart_folder)
            with open(self.restart_folder + 'restart_obj.pkl', 'wb') as out_file:
                pickle.dump(self, out_file, pickle.HIGHEST_PROTOCOL)
            # Checking if a vectorOC was in the restart and preventing the removal of the vector file
            for vec_name, vec in self.vec_dict.items():
                if isinstance(vec, VectorOC):
                    vec.remove_file = False
    
    def read_restart(self):
        """Method to read restart object from saved folder"""
        if os.path.isdir(self.restart_folder):
            with open(self.restart_folder + 'restart_obj.pkl', 'rb') as in_file:
                restart = pickle.load(in_file)
            self.par_dict = restart.par_dict
            self.vec_dict = restart.vec_dict
            # Checking if a vectorOC was in the restart and setting the removal of the vector file
            for vec_name, vec in self.vec_dict.items():
                if isinstance(vec, VectorOC):
                    vec.remove_file = True
            # Removing previous restart and deleting read object
            restart.clear_restart()
            del restart
    
    def clear_restart(self):
        """Method to clear the restart"""
        self.par_dict = dict()
        self.vec_dict = dict()
        # Removing restart folder if existing
        if os.path.isdir(self.restart_folder):
            # Removing folder
            rmtree(self.restart_folder)
