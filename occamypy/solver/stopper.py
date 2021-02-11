import time
from timeit import default_timer as timer
import numpy as np

from occamypy import problem as P


class Stopper:
    """Stopper parent object"""

    # Default class methods/functions
    def __init__(self):
        """Default class constructor for Stopper"""
        return

    def __del__(self):
        """Default destructor"""
        return

    def reset(self):
        """Function to reset stopper variables"""
        raise NotImplementedError("Implement reset stopper in the derived class.")

    def run(self, problem):
        """Dummy stopper running method"""
        raise NotImplementedError("Implement run stopper in the derived class.")


class BasicStopper(Stopper):
    """Basic Stopper with different options"""

    def __init__(self, niter=0, maxfevals=0, maxhours=0.0, tolr=1.0e-32, tolg=1.0e-32, tolg_proj=None, tolobj=None, tolobjrel=None,
                 toleta=None, tolobjchng=None, logger=None):
        """
        Constructor for Basic Stopper:
        niter    	 = [0] - integer; Number of iterations to run (must be greater than 0 to be checked)
        maxfevals    = [0] - integer; Maximum number of function evaluations (must be greater than 0 to be checked)
        maxhours     = [0.0] - float; Maximum total running time in hours (must be greater than 0.0 to be checked)
        tolr     	 = [1.0e-32] - float; Tolerance on residual norm
        tolg     	 = [1.0e-32] - float; Tolerance on gradient norm (Note: ignored for symmetric system)
        tolg_proj    = [None] - float; Tolerance on the projected-gradient infinity norm
        tolobj     	 = [None] - float; Tolerance on objective function value (Not relative value compared to initial one)
        tolobjrel    = [None] - float; Tolerance on relative objective function value (Should range between 0 and 1)
        toleta       = [None] - float; Tolerance on |Am - b|/|b| (Not supported for regularized problems)
        tolobjchng   = [None] - float; Tolerance on the change of the relative objective function value (phi(m_i) - phi(m_i-1)/ phi(m_0)) (Note that, the stopper averages 3 points by default, modify the internal variable 'ave_pts' to change the number of points)
        """
        # Criteria to evaluate whether or not to stop the solver
        super(BasicStopper, self).__init__()
        self.niter = niter
        self.zfill = int(np.floor(np.log10(self.niter)) + 1)  # number of digits for printing the iteration number
        self.maxfevals = maxfevals
        self.maxhours = maxhours
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
        # Logger to write to file stopper information
        self.logger = logger
        # Starting timer
        self.__start = timer()
        return

    def reset(self):
        """Function to reset stopper variables"""
        # Restarting timer
        self.__start = timer()
        self.obj_pts = list()
        return

    # Beware stopper is going to change the gradient/obj/res files
    def run(self, problem, niter, initial_obj_value=None, verbose=False):
        if not isinstance(problem, P.Problem):
            raise TypeError("Input variable is not a Problem object")
        # Variable to impose stopping to solver
        stop = False
        # Taking time run so far (hours)
        elapsed_time = (timer() - self.__start) / 3600.0
        secs = elapsed_time * 3600.0
        # Printing elapsed time in hours, minutes, seconds
        hours = secs // 3600
        mins = (secs % 3600) // 60
        secs = (secs % 60)
        # Printing time stamp to log file if provided
        msg = "Elapsed time: %d hours, %d minutes, %d seconds\n" % (hours, mins, secs) + \
              "Current date & time: %s" % time.strftime("%c")
        res_norm = problem.get_rnorm(problem.model)
        try:
            grad_norm = problem.get_gnorm(problem.model)
        except NotImplementedError:
            grad_norm = None
        obj = problem.get_obj(problem.model)
        if self.logger:
            self.logger.addToLog(msg)
        # Stop by number of iterations
        if 0 < self.niter <= niter:
            stop = True
            msg = "Terminate: maximum number of iterations reached\n"
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        if 0 < self.maxfevals <= problem.get_fevals():
            stop = True
            msg = "Terminate: maximum number of evaluations\n"
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        if 0. < self.maxhours <= elapsed_time:
            stop = True
            msg = "Terminate: maximum number hours reached %s\n" % elapsed_time
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        if res_norm < self.tolr:
            stop = True
            msg = "Terminate: minimum residual tolerance reached %s\n" % res_norm
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        if grad_norm is not None and grad_norm < self.tolg:
            stop = True
            msg = "Terminate: minimum gradient tolerance reached %s\n" % grad_norm
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
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
        if self.tolobj is not None:
            if obj < self.tolobj:
                stop = True
                msg = "Terminate: objective function value tolerance of %s reached, objective function value %s\n"\
                      % (self.tolobj, obj)
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
                return stop
        if self.tolobjrel is not None and initial_obj_value is not None and initial_obj_value != 0.:
            if obj / initial_obj_value < self.tolobjrel:
                stop = True
                msg = "Terminate: relative objective function value tolerance of %s reached, relative objective " \
                      "function value %s\n" % (self.tolobjrel, obj / initial_obj_value)
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
                return stop
        if self.toleta is not None:
            data_norm = problem.data.norm()
            if res_norm < self.toleta * data_norm:
                stop = True
                msg = "Terminate: eta tolerance (i.e., |Am - b|/|b|) of %s reached, eta value %s"\
                      % (self.toleta, res_norm / data_norm)
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
                return stop
        if self.tolobjchng is not None and initial_obj_value is not None and initial_obj_value != 0.0:
            # Saving current scaled objective function value
            self.obj_pts.append(obj / initial_obj_value)
            if niter >= self.ave_pts:
                # Appending initial objective function value
                objtmp = [1.0] + self.obj_pts
                objchng = np.abs(np.sum(np.diff(objtmp[-self.ave_pts - 1:])) / self.ave_pts)
                if objchng < self.tolobjchng:
                    stop = True
                    msg = "Terminate: tolerance on the change of the relative objective function value (i.e., " \
                          "phi(m_i) - phi(m_i-1)/ phi(m_0)) of %s reached (computed using %s points), change value " \
                          "%s" % (self.tolobjchng, self.ave_pts, objchng)
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                    return stop
        return stop


class SamplingStopper(Stopper):
    """Sampling Stopper with different options"""

    def __init__(self, nsamples, maxhours=0.0, logger=None):
        """
        Constructor for Sampling Stopper:
        nsamples - integer; Number of samples to tests
        maxhours - float; Maximum total running time in hours (must be greater than 0.0 to be checked) [0.0]
        """
        # Criteria to evaluate whether or not to stop the solver
        super(SamplingStopper, self).__init__()
        self.nsamples = nsamples
        self.zfill = int(np.floor(np.log10(self.nsamples)) + 1)  # number of digits for printing the iteration number
        self.maxhours = maxhours
        # Logger to write to file stopper information
        self.logger = logger
        # Starting timer
        self.__start = timer()
        return

    def reset(self):
        """Function to reset stopper variables"""
        # Restarting timer
        self.__start = timer()
        return

    # Beware stopper is going to change the gradient/obj/res files
    def run(self, problem, nsamples, verbose=False):
        if not isinstance(problem, P.Problem):
            raise TypeError("Input variable is not a Problem object")
        # Variable to impose stopping to solver
        stop = False
        # Taking time run so far (hours)
        elapsed_time = (timer() - self.__start) / 3600.0
        secs = elapsed_time * 3600.0
        # Printing elapsed time in hours, minutes, seconds
        hours = secs // 3600
        mins = (secs % 3600) // 60
        secs = (secs % 60)
        # Printing time stamp to log file if provided
        msg = "Elapsed time: %d hours, %d minutes, %d seconds\n" % (hours, mins, secs) + \
              "Current date & time: %s" % time.strftime("%c")
        if self.logger:
            self.logger.addToLog(msg)
        # Stop by number of iterations
        if 0 < self.nsamples <= nsamples:
            stop = True
            msg = "Terminate: maximum number of requested samples reached\n"
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        if 0. < self.maxhours <= elapsed_time:
            stop = True
            msg = "Terminate: maximum number hours reached %s\n" % elapsed_time
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            return stop
        return stop
