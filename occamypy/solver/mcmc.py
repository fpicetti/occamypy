from math import isnan
from copy import deepcopy

import numpy as np

from occamypy.utils import ZERO
from occamypy.solver.base import Solver

__all__ = [
    "MCMC",
]


class MCMC(Solver):
    """Markov chain Monte Carlo sampling algorithm"""
    
    def __init__(self, stopper, prop_distr: str = "u", temperature: float = None, **kwargs):
        """
        MCMC Solver/Sampler constructor

        Args:
            stopper: Stopper object to terminate sampling
            prop_distr: proposal distribution to be employed ["u","n"]
            temperature: coefficient for Temperature Metropolis sampling algorithm
            logger: Logger to write inversion log file
            sigma: standard deviation if prop_distr="n"
            max_step: upper bound if prop_distr="u"
            min_step: lower bound if prop_distr="u"
        """
        
        super(MCMC, self).__init__(stopper=stopper, logger=kwargs.get("logger", None), name="MCMC")
        
        # Proposal distribution parameters
        self.prop_dist = prop_distr.lower()
        # backward compatibility
        if self.prop_dist == "uni":
            self.prop_dist = "u"
        if self.prop_dist == "gauss":
            self.prop_dist = "n"
        
        if self.prop_dist == "u":
            self.max_step = kwargs.get("max_step", 1.)
            self.min_step = kwargs.get("min_step", -self.max_step)
        elif self.prop_dist == "n":
            self.sigma = kwargs.get("sigma", 1.)
        else:
            raise ValueError("Not supported prop_distr")
        
        # print formatting
        self.iter_msg = "sample = %s, log-obj = %.2e, rnorm = %.2e, feval = %s, ar = %2.5f%%, alpha = %1.5f"
        self.ndigits = self.stopper.zfill
        # Temperature Metropolis sampling algorithm (see, Monte Carlo sampling of
        # solutions to inverse problems by Mosegaard and Tarantola, 1995)
        # If not provided the likelihood is assumed to be passed to the run method
        self.T = temperature or kwargs.get("T", None)
        
        # Reject sample outside of bounds or project it onto them
        self.reject_bound = True
    
    def setDefaults(self, save_obj=True, save_res=False, save_grad=False, save_model=True, prefix=None,
                    iter_buffer_size=None, iter_sampling=1, flush_memory=False):
        """
        Function to set parameters for result saving.

        Args:
            save_obj: save objective function values into the list self.obj (True for MCMC)
            save_res: save residual vectors into the list self.res
            save_grad: save gradient vectors into the list self.grad
            save_model: save domain vectors into the list self.domain. (True for MCMC)
                It will also say the last inverted domain vector into self.inv_model
            prefix: prefix of the files in which requested results will be saved;
                If prefix is None, then nothing is going to be saved on disk
            iter_buffer_size: number of steps to save before flushing results to disk
                (by default the solver waits until all iterations are done)
            iter_sampling: sampling of the iteration axis
            flush_memory: keep results into the object lists or clean those once inversion is completed or results have been written on disk
        """
        if not save_obj:
            print("WARNING! MCMC is useful for estimating the PDF. Are you sure you don't want to save obj?")
        if not save_model:
            print("WARNING! MCMC samples different models and compute an obj. Are you sure you don't want to save model?")

        super(MCMC, self).setDefaults(save_obj=save_obj, save_res=save_res, save_grad=save_grad, save_model=save_model,
                                      prefix=prefix, iter_buffer_size=iter_buffer_size,
                                      iter_sampling=iter_sampling, flush_memory=flush_memory)
    
    def run(self, problem, verbose=False, restart=False):
        """
        Run MCMC solver/sampler

        Args:
            problem: problem to be minimized
            verbose: verbosity flag
            restart: restart previously crashed inversion
        """
        create_msg = verbose or self.logger
        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Checking if user is saving the sampled models
        if not self.save_model:
            msg = "WARNING! save_model=False! Running MCMC sampling method will not save accepted model samples!"
            print(msg)
            if self.logger:
                self.logger.addToLog(msg)
        
        if not restart:
            if create_msg:
                msg = 90 * "#" + "\n"
                msg += 12 * " " + "MCMC Solver log file\n"
                msg += 4 * " " + "Restart folder: %s\n" % self.restart.restart_folder
                msg += 4 * " " + "Problem: %s\n" % problem.name
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace("log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
            prblm_mdl = problem.get_model()
            mcmc_mdl_cur = prblm_mdl.clone()
            
            # Other internal variables
            accepted = 1  # number of accepted samples
            iiter = 1  # number of tested point so far
        else:
            msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            
            self.restart.read_restart()
            
            mcmc_mdl_cur = self.restart.retrieve_vector("mcmc_mdl_cur")
            accepted = self.restart.retrieve_parameter("accepted")
            iiter = self.restart.retrieve_parameter("iiter")
            # Setting the last accepted model
            problem.set_model(mcmc_mdl_cur)
            prblm_mdl = problem.get_model()
        
        # Common parameters
        mcmc_mdl_prop = prblm_mdl.clone()
        mcmc_dmodl = prblm_mdl.clone().zero()
        mcmc_mdl_check = prblm_mdl.clone()
        
        # Computing current objective function
        obj_current = problem.get_obj(mcmc_mdl_cur)  # Compute objective function value
        res_norm = problem.get_rnorm(mcmc_mdl_cur)
        # getting each objective function term if present
        obj_terms = problem.obj_terms if "obj_terms" in dir(problem) else None
        
        if not restart:
            if create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.ndigits),
                                       np.log(obj_current),
                                       res_norm,
                                       str(problem.get_fevals()).zfill(self.stopper.zfill + 1),
                                       100. * float(accepted) / iiter,
                                       1.0)
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            
            if isnan(obj_current):
                raise ValueError("objective function value NaN!")
            self.save_results(iiter, problem, force_save=False)
        
        # Sampling loop
        while True:
            # Generate a candidate y from x according to the proposal distribution r(x_cur, x_prop)
            if self.prop_dist == "u":
                mcmc_dmodl.rand(low=self.min_step, high=self.max_step)
            elif self.prop_dist == "n":
                mcmc_dmodl.randn(std=self.sigma)
            
            # Compute a(x_cur, x_prop)
            mcmc_mdl_prop.copy(mcmc_mdl_cur)
            mcmc_mdl_prop.scaleAdd(mcmc_dmodl)
            # Checking if model parameters hit the bounds
            mcmc_mdl_check.copy(mcmc_mdl_prop)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(mcmc_mdl_check)
            
            if mcmc_mdl_prop.isDifferent(mcmc_mdl_check):
                msg = 4 * " " + "Model hit provided bounds. Projecting onto them."
                if self.reject_bound:
                    msg = 4 * " " + "Model hit provided bounds. Resampling proposed point."
                # Model hit bounds
                if self.logger:
                    self.logger.addToLog(msg)
                if self.reject_bound:
                    continue
                else:
                    mcmc_mdl_prop.copy(mcmc_mdl_check)
            
            obj_prop = problem.get_obj(mcmc_mdl_prop) + ZERO
            
            if isnan(obj_prop):
                raise ValueError("objective function of proposed model is NaN!")
            
            if self.T:
                # Using Metropolis method assuming an objective function was passed
                alpha = 1.0
                if obj_prop > obj_current:
                    alpha = np.exp(-(obj_prop - obj_current) / self.T)
            else:
                # computing log acceptance ratio assuming likelihood function
                if obj_current > ZERO and obj_prop > ZERO:
                    alpha = min(1.0, obj_prop / obj_current)
                elif obj_current <= ZERO < obj_prop:
                    # condition to avoid zero/zero
                    alpha = 0.
                else:
                    # condition to avoid division by zero
                    alpha = 1.
            
            # Increase counter of tested samples
            iiter += 1
            
            # Accept the x_prop with probability a
            if np.random.uniform() <= alpha:
                # accepted proposed model
                mcmc_mdl_cur.copy(mcmc_mdl_prop)
                obj_current = deepcopy(obj_prop)
                obj_terms = deepcopy(problem.obj_terms) if "obj_terms" in dir(problem) else None
                res_norm = problem.get_rnorm(mcmc_mdl_cur)
                accepted += 1
            
            # iteration info
            msg = self.iter_msg % (str(iiter).zfill(self.ndigits),
                                   np.log(obj_current),
                                   res_norm,
                                   str(problem.get_fevals()).zfill(self.stopper.zfill + 1),
                                   100. * float(accepted) / iiter,
                                   alpha)
            if verbose:
                print(msg)
            # Writing on log file
            if self.logger:
                self.logger.addToLog("\n" + msg)
            
            # Saving sampled point
            self.save_results(iiter, problem, model=mcmc_mdl_cur, obj=obj_current, obj_terms=obj_terms, force_save=True,
                              force_write=True)
            
            # saving inversion vectors and parameters for restart
            self.restart.save_vector("mcmc_mdl_cur", mcmc_mdl_cur)
            self.restart.save_parameter("accepted", accepted)
            self.restart.save_parameter("iiter", iiter)
            
            # Checking stopping criteria
            if self.stopper.run(problem=problem, iiter=iiter, verbose=verbose):
                break
        
        if create_msg:
            msg = 90 * "#" + "\n"
            msg += 12 * " " + "MCMC Solver log file end\n"
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace("log file ", ""))
            if self.logger:
                self.logger.addToLog(msg)
        self.restart.clear_restart()
