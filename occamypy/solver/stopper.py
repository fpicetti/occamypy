import time
from timeit import default_timer as timer
import numpy as np

from occamypy.problem.base import Problem


__all__ = [
    "Stopper",
    "BasicStopper",
    "SamplingStopper",
]


def seconds_to_hms(seconds: float):
    hours = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return hours, mins, secs


def hms_to_seconds(h: int, m: int, s: float) -> float:
    return h * 3600 + m * 60 + s


class Stopper:
    """
    Base stopper class.
    Used to implement stopping criteria for the solver.

    Methods:
        reset: reset the inner variables
        run(problem): apply its criteria to the problem
        
    Attributes:
        niter: number of iterations to run (must be greater than 0 to be checked)
        maxhours: maximum total running time in hours (must be greater than 0. to be checked)
        logger: logger object (if any)
        zfill: number of zeros to print in a decimal notation the iteration
        start_time: start time
    """

    def __init__(self, niter: int = 0, maxhours: float = 0., logger=None):
        """
        Stopper constructor
        Args:
            niter: number of iterations to run (must be greater than 0 to be checked)
            maxhours: maximum total running time in hours (must be greater than 0. to be checked)
            logger: Logger object
        """
        self.niter = niter
        self.logger = logger
        self.maxhours = maxhours
        # number of digits for printing the iteration number
        self.zfill = int(np.floor(np.log10(self.niter)) + 1)
        # timer
        self.start_time = timer()
    
    def reset(self):
        """Function to reset stopper variables"""
        self.start_time = timer()

    def run(self, problem: Problem, iiter: int, verbose: bool = False, **kwargs):
        """
        Apply the stopping criteria to the problem
        
        Args:
            problem: problem that is being solved
            iiter: current iteration number
            verbose: verbosity flag

        Notes:
            Beware stopper is going to change the gradient/obj/res files
        """
        raise NotImplementedError("Implement run stopper in the derived class.")


class BasicStopper(Stopper):
    """Basic Stopper with different criteria on tolerances and function evaluations"""
    
    def __init__(self, niter=0, maxfevals=0, tolr=1.0e-32, tolg=1.0e-32, tolg_proj=None, tolobj=None, tolobjrel=None,
                 toleta=None, tolobjchng=None, **kwargs):
        """
        BasicStopper constructor
        
        Args:
            maxfevals: maximum number of function evaluations (must be greater than 0 to be checked)
            tolr: tolerance on residual norm
            tolg: tolerance on gradient norm (ignored if symmetric system)
            tolg_proj: tolerance on the projected-gradient infinity norm
            tolobj: absolute tolerance on objective function value (compared to initial one)
            tolobjrel: relative tolerance on objective function value (should range between 0 and 1)
            toleta: tolerance on |Am - d|/|d| (not supported for regularized problems)
            tolobjchng: step-relative tolerance on objective function value (phi(m_i) - phi(m_i-1))/ phi(m_0).
               note: modify the internal variable 'ave_pts' to change the number of points on which to compute the criterion)
        """
        self.maxfevals = maxfevals
        self.tolr = tolr
        self.tolg = tolg
        self.tolg_proj = tolg_proj
        self.tolobj = tolobj
        self.tolobjrel = tolobjrel
        self.toleta = toleta
        # Tolerance on the change of the relative objective function value
        self.tolobjchng = tolobjchng
        # number of points to average plus the oldest one to compute objetive the change
        self.ave_pts = 3
        self.obj_pts = list()
        
        super(BasicStopper, self).__init__(niter=niter,
                                           maxhours=kwargs.get("maxhours", 0.),
                                           logger=kwargs.get("logger", None))
    
    def reset(self):
        self.obj_pts = list()
        super(BasicStopper, self).reset()

    def run(self, problem: Problem, iiter: int, verbose: bool = False, **kwargs):

        stop = False
        
        # Taking time run so far (seconds)
        elapsed_time = timer() - self.start_time
       
        # Printing time stamp to log file if provided
        if self.logger:
            self.logger.addToLog("Elapsed time: %d hours, %d minutes, %d seconds\n" % seconds_to_hms(elapsed_time) +
                                 "Current date & time: %s" % time.strftime("%c"))
        
        res_norm = problem.get_rnorm(problem.model)
        try:
            grad_norm = problem.get_gnorm(problem.model)
        except NotImplementedError:
            grad_norm = None
        obj = problem.get_obj(problem.model)

        # stop by number of iterations
        if 0 < self.niter <= iiter:
            stop = True
            msg = "Terminate: maximum number of iterations reached\n"
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        
        # stop by number of function evaluations
        if 0 < self.maxfevals <= problem.get_fevals():
            stop = True
            msg = "Terminate: maximum number of evaluations\n"
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        
        # stop by computation time
        if 0. < self.maxhours <= elapsed_time / 3600:
            stop = True
            msg = "Terminate: maximum number hours reached %s\n" % (elapsed_time / 3600)
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        
        # residual norm tolerance
        if res_norm < self.tolr:
            stop = True
            msg = "Terminate: minimum residual tolerance reached %s\n" % res_norm
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        
        # gradient norm tolerance
        if grad_norm is not None and grad_norm < self.tolg:
            stop = True
            msg = "Terminate: minimum gradient tolerance reached %s\n" % grad_norm
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        
        # projected gradient tolerance
        if grad_norm is not None and self.tolg_proj is not None:
            # Get the inf-norm of the projected gradient
            # Equation (6.1), Page 17. (L-BFGS-B)
            proj_grad = problem.model.clone()
            proj_grad.scaleAdd(problem.get_grad(problem.model), 1.0, -1.0)
            if "bounds" in dir(problem):
                problem.bounds.apply(proj_grad)
            proj_grad.scaleAdd(problem.model, 1.0, -1.0)
            grad_norm_proj = proj_grad.abs().max()
            del proj_grad
            if grad_norm_proj < self.tolg_proj:
                stop = True
                msg = "Terminate: tolerance of the projected gradient reached %s\n" % grad_norm_proj
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
                return stop
        
        # objective function tolerance
        if self.tolobj is not None and obj < self.tolobj:
            stop = True
            msg = "Terminate: objective function tolerance reached: %s < %s\n"\
                  % (obj, self.tolobj)
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        
        # residual vs data tolerance
        if self.toleta is not None:
            data_norm = problem.data.norm()
            if res_norm < self.toleta * data_norm:
                stop = True
                msg = "Terminate: eta tolerance (i.e., |Am - b|/|b|) reached: %s < %s"\
                      % (res_norm / data_norm, self.toleta)
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
                return stop

        # objective function relative tolerance
        initial_obj_value = kwargs.get("initial_obj_value", None)
        if self.tolobjrel is not None and initial_obj_value is not None and initial_obj_value != 0.:
            if obj / initial_obj_value < self.tolobjrel:
                stop = True
                msg = "Terminate: objective function relative tolerance reached: %s < %s\n"\
                      % (obj / initial_obj_value, self.tolobjrel)
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
                return stop
                
        # objective function update tolerance
        if self.tolobjchng is not None and initial_obj_value is not None and initial_obj_value != 0.:
            # Saving current scaled objective function value
            self.obj_pts.append(obj / initial_obj_value)
            if iiter >= self.ave_pts:
                # Appending initial objective function value
                objtmp = [1.0] + self.obj_pts
                objchng = np.abs(np.sum(np.diff(objtmp[-self.ave_pts - 1:])) / self.ave_pts)
                if objchng < self.tolobjchng:
                    stop = True
                    msg = "Terminate: objective function update tolerance reached: %s < %s (computed using %s points)\n" \
                          % (objchng, self.tolobjchng, self.ave_pts)
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                    return stop
        return stop


class SamplingStopper(Stopper):
    """
    Stopper that check the  number of tested samples.

    Examples:
        used in MCMC solver
    """
    
    def __init__(self, niter: int = 0, **kwargs):
        """SamplingStopper constructor"""
        nsamples = kwargs.get("nsamples", None)
        if nsamples is not None and niter == 0:
            niter = nsamples
        super(SamplingStopper, self).__init__(niter=niter,
                                              maxhours=kwargs.get("maxhours", 0.),
                                              logger=kwargs.get("logger", None))

    def reset(self):
        super(SamplingStopper, self).reset()

    def run(self, problem: Problem, iiter: int, verbose: bool = False, **kwargs):
        stop = False
        
        # Taking time run so far (seconds)
        elapsed_time = timer() - self.start_time
        
        if self.logger:
            self.logger.addToLog("Elapsed time: %d hours, %d minutes, %d seconds\n" % seconds_to_hms(elapsed_time) +
                                 "Current date & time: %s" % time.strftime("%c"))
        
        # Stop by number of iterations
        if 0 < self.niter <= iiter:
            stop = True
            msg = "Terminate: maximum number of requested samples reached\n"
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        
        # stop by computation time
        if 0. < self.maxhours <= elapsed_time / 3600:
            stop = True
            msg = "Terminate: maximum number hours reached %s\n" % (elapsed_time / 3600)
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        
        return stop
