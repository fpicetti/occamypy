from math import isnan

import numpy as np

from occamypy.vector.base import superVector
from occamypy.operator.base import Vstack
from occamypy.problem.linear import Lasso, GeneralizedLasso, LeastSquares
from occamypy.solver.base import Solver
from occamypy.solver.stopper import BasicStopper
from occamypy.solver.linear import CG, LSQR, SD
from occamypy.utils import Logger, ZERO


def _soft_thresh(x, thresh):
    r"""
    Soft-thresholding function:
    
    .. math::
        \mathbf{y} = \mathrm{sign}(\mathbf{x}) \cdot \mathrm{max}(\vert \mathbf{x} \vert - \tau, \mathbf{0})
    
    Args:
        x: input vector
        thresh: soft threshold value
    
    Returns:
        clipped vector
    """
    return x.clone().sign() * x.clone().abs().addbias(-thresh).maximum(0.)


def _shrinkage(x, thresh, eps=1e-10):
    r"""
    Shrinkage function
    
    .. math::
        \mathbf{y} = \frac{\mathbf{x}}{\vert \mathbf{x} \vert + \varepsilon} \cdot \mathrm{max}(\vert \mathbf{x} \vert - \tau, \mathbf{0})
    """
    y = x.clone()
    y.multiply(x.clone().abs().addbias(eps).reciprocal())
    return y * x.clone().abs().addbias([-t for t in thresh]).maximum(0.)


def _proximal_L2(x, thresh, eps=1e-10):
    r"""
    Proximal operator for L2 distance
    
    .. math::
        \mathbf{y} = \frac{1}{\Vert \mathbf{x} \Vert_2 + \varepsilon} \cdot \mathrm{max}(\Vert \mathbf{x} \Vert_2 - \tau, \mathbf{0})
    
    References:
        https://sporco.readthedocs.io/en/latest/modules/sporco.prox.html#sporco.prox.prox_l2
    """
    x_norm = x.norm()
    m = max(0, x_norm - thresh)
    c = m / (x_norm + eps)
    return x.clone().scale(c)


class ISTA(Solver):
    """Iterative Shrikage-Thresholding Algorithm (ISTA) Solver"""
    
    def __init__(self, stopper, fast=False, **kwargs):
        """
        ISTA constructor
        
        Args:
            stopper: Stopper to terminate the inversion
            fast: use the Fast-ISTA implementation
            logger: Logger to write inversion log file
        """
        super(ISTA, self).__init__(stopper=stopper, logger=kwargs.get("logger", None), name="ISTA")
        
        # Setting the fast flag
        self.fast = fast
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, rnorm = %.2e, gnorm = %.2e, feval = %s"
    
    def run(self, problem, verbose=False, restart=False):
        """Run ISTA solver"""
        
        create_msg = verbose or self.logger
        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Checking if the provided problem is L1-LASSO
        if not isinstance(problem, Lasso):
            raise TypeError("Provided inverse problem not ProblemL1Lasso!")
        # Checking if the regularization weight was set
        if problem.lambda_value is None:
            raise ValueError("Regularization weight (lambda_value) is not set!")
        
        if not restart:
            if create_msg:
                msg = 90 * "#" + "\n"
                msg += 12 * " " + ("FAST-" if self.fast else "")
                msg += "ISTA Solver log file\n"
                msg += 4 * " " + "Restart folder: %s\n" % self.restart.restart_folder
                msg += 4 * " " + "Problem: %s\n" % problem.name
                msg += 4 * " " + "Regularization weight: %.2e\n" % problem.lambda_value
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
            
            # Setting internal vectors (model, search direction, and previous gradient vectors)
            prblm_mdl = problem.get_model()
            ista_mdl = prblm_mdl.clone()
            # Other parameters in case FISTA is requested
            if self.fast:
                t = 1.0
                # fista_mdl = prblm_mdl.clone()
            
            # Other internal variables
            iiter = 0
        else:
            # Retrieving parameters and vectors to restart the solver
            if create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            self.restart.read_restart()
            # Retrieving inversion parameters
            iiter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            ista_mdl = self.restart.retrieve_vector("ista_mdl")
            # Other parameters in case FISTA is requested
            if self.fast:
                t = self.restart.retrieve_parameter("t")
                # fista_mdl = self.restart.retrieve_vector("fista_mdl")
        
        ista_mdl0 = ista_mdl.clone()  # Previous model in case stepping procedure fails
        
        # Inversion loop
        while True:
            obj0 = problem.get_obj(ista_mdl)  # Compute objective function value
            prblm_grad = problem.get_grad(ista_mdl)  # Compute the gradient g = - op' [y - Ax]
            if iiter == 0:
                # Saving initial objective function value
                initial_obj_value = obj0
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if create_msg:
                    msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.get_rnorm(ista_mdl),
                                           problem.get_gnorm(ista_mdl),
                                           problem.get_fevals())
                    # Writing on log file
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj0) or isnan(prblm_grad.norm()):
                    raise ValueError("Either gradient norm or objective function value NaN!")
            if problem.get_gnorm(ista_mdl) == 0.:
                print("Gradient vanishes identically")
                break
            
            # Saving results
            self.save_results(iiter, problem, force_save=False)
            ista_mdl0.copy(ista_mdl)  # Saving model before updating it
            
            # Update model x = x + scale_precond * op' [y - Ax]
            ista_mdl.scaleAdd(prblm_grad, 1.0, -1.0 / problem.op_norm)
            
            # SOFT-THRESHOLDING STEP
            ista_mdl.copy(_soft_thresh(ista_mdl, problem.lambda_value / problem.op_norm))
            
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(ista_mdl)
            
            if self.fast:
                t0 = t
                t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
                # z = x + ((t0 - 1.) / t) * (x - xold)
                scale = (t0 - 1.) / t
                ista_mdl.scaleAdd(ista_mdl0, 1.0 + scale, -scale)
            
            obj1 = problem.get_obj(ista_mdl)
            if obj1 >= obj0:
                if create_msg:
                    msg = "Objective function didn't reduce, will terminate solver:\n\t" \
                          "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                # Copying back to the previous solution
                ista_mdl.copy(ista_mdl0)
                break
            
            # Saving current model in case of restart and other parameters
            self.restart.save_parameter("iter", iiter)
            self.restart.save_vector("ista_mdl", ista_mdl)
            if self.fast:
                self.restart.save_parameter("t", t)
                # self.restart.save_vector("fista_mdl", fista_mdl)
            
            # iteration info
            iiter = iiter + 1
            if create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.get_rnorm(ista_mdl),
                                       problem.get_gnorm(ista_mdl),
                                       str(problem.get_fevals()).zfill(self.stopper.zfill + 1))
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            # Check if either objective function value or gradient norm is NaN
            if isnan(obj1) or isnan(prblm_grad.norm()):
                raise ValueError("Either gradient norm or objective function value NaN!")
            if self.stopper.run(problem=problem, iiter=iiter, verbose=verbose, initial_obj_value=initial_obj_value):
                break
        
        # Writing last inverted model
        self.save_results(iiter, problem, force_save=True, force_write=True)
        if create_msg:
            msg = 90 * "#" + "\n"
            msg += 12 * " " + ("FAST-" if self.fast else "")
            msg += "ISTA Solver log file end\n"
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog(msg)
        # Clear restart object
        self.restart.clear_restart()


class FISTA(ISTA):
    """Fast Iterative Shrikage-Thresholding Algorithm (F-ISTA) Solver"""
    
    def __init__(self, stopper, logger=None):
        """
        F-ISTA constructor

        Args:
            stopper: Stopper to terminate the inversion
            logger: Logger to write inversion log file
        """
        super(FISTA, self).__init__(stopper=stopper, logger=logger, fast=True)
        self.name = "Fast-ISTA"


class ISTC(Solver):
    """Iterative Shrikage-Thresholding Algorithm with Cooling (ISTC) Solver"""
    
    def __init__(self, stopper, inner_it, cooling_start, cooling_end, **kwargs):
        """
        ISTC constructor
        
        Args:
            stopper: Stopper to terminate the inversion
            inner_it: number of inner iterations
            cooling_start: start of cooling continuation as fraction of size of sorted array |A'd|
            cooling_end: end of cooling continuation as fraction of size of sorted array |A'd|
            logger: Logger to write inversion log file
        """
        
        super(ISTC, self).__init__(stopper=stopper, logger=kwargs.get("logger", None), name="Cooled ISTA")

        self.iter_msg = "Inner_iter = %s, obj = %.5e, rnorm = %.2e, gnorm= %.2e, feval = %s"
        
        # ISTC parameters
        if self.stopper.niter <= 0:
            raise ValueError("iiter for stopper object must be positive and greater than 0!")
        self.inner_it = inner_it  # number of inner iterations, the outer iterations are taken care by the stopper
        # cooling_start and cooling_end are numbers between 0 and 1 such that cooling_start <= cooling_end
        if not 0 <= cooling_start <= 1 or not 0 <= cooling_end <= 1 or cooling_end < cooling_start:
            raise ValueError("Cooling_start and end must be within [0,1] interval and cooling_start <= cooling_end")
        self.cooling_start = cooling_start  # start of cooling continuation as fraction of size of sorted array |op'y|
        self.cooling_end = cooling_end  # end of cooling continuation as fraction of size of sorted array |op'y|
    
    def run(self, problem, verbose=False, restart=False):
        """Run ISTC solver"""
        
        create_msg = verbose or self.logger
        
        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Checking if the provided problem is L1-LASSO
        if not isinstance(problem, Lasso):
            raise TypeError("Provided inverse problem not ProblemL1Lasso!")
        # Computing preconditioning
        scale_precond = 0.99 * np.sqrt(2) / problem.op_norm  # scaling factor applied to operator op for preconditioning
        if not restart:
            if create_msg:
                msg = 90 * "#" + "\n"
                msg += 12 * " " + "ISTC Solver log file\n"
                msg += 4 * " " + "Restart folder: %s\n" % self.restart.restart_folder
                msg += 4 * " " + "Problem: %s\n" % problem.name
                msg += 4 * " " + "Regularization weight: %.2e\n" % problem.lambda_value
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
            
            # Setting internal vectors (model, search direction, and previous gradient vectors)
            prblm_mdl = problem.get_model()
            istc_mdl = prblm_mdl.clone()
            
            # Inversion always starts from m = 0
            # (I need to understand if it is possible to start from m different than 0)
            istc_mdl.zero()  # modl = 0
            # Other internal variables
            iiter = 0
            # Computing cooling schedule for lambda values
            # istc_mdl.scale(scale_precond) #Currently unnecessary since starting model is zero
            prblm_grad = problem.get_grad(istc_mdl)
            grad_arr = np.copy(prblm_grad.getNdArray())
            grad_arr = np.abs(grad_arr.flatten())  # |op'y| and removing zero elements
            grad_arr = grad_arr[np.nonzero(grad_arr)]
            if grad_arr.size == 0:
                raise ValueError("-- op'y is returning a null vector (i.e., y in the Null space of op')")
            # Sorting the elements in descending order
            grad_arr.sort()
            grad_arr = np.flip(grad_arr, 0)
            # Setting fraction of points sampled by the outer loop (linear sampling)
            samples = np.array(np.round(np.linspace(self.cooling_start,
                                                    self.cooling_end,
                                                    self.stopper.niter)
                                        * grad_arr.size), dtype=np.uint64, copy=False)
            # Lambda values to be used during inversion for each outer loop iteration
            lambda_values = grad_arr[samples]
            # Scaling by the preconditioning
            lambda_values *= scale_precond
            # Saving the lambda values to avoid recomputation if restart is used
            self.restart.save_parameter("lambda_values", lambda_values)
        else:
            # Retrieving parameters and vectors to restart the solver
            if create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            self.restart.read_restart()
            # Retrieving lambda values and other parameters
            lambda_values = self.restart.retrieve_parameter("lambda_values")
            iiter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            istc_mdl = self.restart.retrieve_vector("istc_mdl")
        
        # Common variables unrelated to restart
        success = True
        istc_mdl0 = istc_mdl.clone()  # Previous model in case stepping procedure fails
        istc_mdl_save = istc_mdl0  # used also to save results
        
        # Outer iteration loop
        while True:
            # Setting lambda value for a given outer loop iteration
            problem.set_lambda(lambda_values[iiter])
            problem.obj_updated = False  # Lambda has been changed so objective function will change as well
            if create_msg:
                msg = "Outer_iter = %s\tlambda_value = %.2e" % (
                    str(iiter).zfill(self.stopper.zfill), lambda_values[iiter])
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            if not restart:
                inner_iter = 0
            else:
                self.restart.retrieve_parameter("inner_iter", inner_iter)
                restart = False
            
            if iiter == 0:
                # Applying preconditioning
                istc_mdl.scale(scale_precond)
                obj = problem.get_obj(istc_mdl)  # Compute objective function value
                # Saving initial objective function value
                initial_obj_value = obj
                self.restart.save_parameter("obj_initial", initial_obj_value)
            while inner_iter < self.inner_it:
                obj0 = problem.get_obj(istc_mdl)  # Compute objective function value
                prblm_grad = problem.get_grad(istc_mdl)  # Compute the gradient g = - op' [y - Ax]
                if inner_iter == 0:
                    if create_msg:
                        msg = self.iter_msg % (str(inner_iter).zfill(self.stopper.zfill),
                                               obj0,
                                               problem.get_rnorm(istc_mdl),
                                               problem.get_gnorm(istc_mdl),
                                               str(problem.get_fevals()).zfill(self.stopper.zfill + 1))
                        # Writing on log file
                        if verbose:
                            print(msg)
                        if self.logger:
                            self.logger.addToLog(msg)
                    # Check if either objective function value or gradient norm is NaN
                    if isnan(obj0) or isnan(prblm_grad.norm()):
                        raise ValueError("Either gradient norm or objective function value NaN!")
                if problem.get_gnorm(istc_mdl) == 0.:
                    print("Gradient vanishes identically")
                    break
                
                # Removing preconditioning scaling factor from inverted model
                istc_mdl_save.copy(istc_mdl)
                istc_mdl_save.scale(scale_precond)
                # Saving results
                self.save_results(iiter, problem, model=istc_mdl_save, force_save=False)
                
                # Stepping for internal iteration model update
                istc_mdl0.copy(istc_mdl)  # Saving model before updating it
                istc_mdl.scaleAdd(prblm_grad, 1.0, -scale_precond)  # Update model x = x + scale_precond * op' [y - Ax]
                #########################################
                # SOFT-THRESHOLDING STEP
                istc_mdl = _soft_thresh(istc_mdl, problem.lambda_value)
                #########################################
                # Projecting model onto the bounds (if any)
                if "bounds" in dir(problem):
                    problem.bounds.apply(istc_mdl)
                
                obj1 = problem.get_obj(istc_mdl)
                problem.get_model().writeVec("problem_model.H")
                istc_mdl.writeVec("solver_model.H")
                if obj1 >= obj0:
                    if create_msg:
                        msg = "Objective function didn't reduce, will terminate solver:\n\t" \
                              "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
                        if verbose:
                            print(msg)
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog(msg)
                    # Copying back to the previous solution
                    istc_mdl.copy(istc_mdl0)
                    break
                
                # Saving current model in case of restart and other parameters
                self.restart.save_parameter("iter", iiter)
                self.restart.save_parameter("inner_iter", inner_iter)
                self.restart.save_vector("istc_mdl", istc_mdl)
                
                # iteration info
                inner_iter += 1
                if create_msg:
                    msg = self.iter_msg % (str(inner_iter).zfill(self.stopper.zfill),
                                           obj1,
                                           problem.get_rnorm(istc_mdl),
                                           problem.get_gnorm(istc_mdl),
                                           (problem.get_fevals()).zfill(self.stopper.zfill + 1))
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj1) or isnan(prblm_grad.norm()):
                    raise ValueError("Either gradient norm or objective function value NaN!")
            iiter = iiter + 1
            if self.stopper.run(problem=problem, iiter=iiter, verbose=verbose, initial_obj_value=initial_obj_value):
                break
        
        # Removing preconditioning scaling factor from inverted model
        istc_mdl_save.copy(istc_mdl)
        istc_mdl_save.scale(scale_precond)
        # Writing last inverted model
        self.save_results(iiter, problem, model=istc_mdl_save, force_save=True, force_write=True)
        if create_msg:
            msg = 90 * "#" + "\n"
            msg += 12 * " " + "ISTC Solver log file end\n"
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog(msg)
        # Clear restart object
        self.restart.clear_restart()


class SplitBregman(Solver):
    """Split-Bregman algorithm for Generalized LASSO problems"""
    
    def __init__(self, stopper, niter_inner=3, niter_solver=5, breg_weight=1., linear_solver='CG',
                 warm_start=False, mod_tol=1e-10, **kwargs):
        """
        SplitBregman constructor
        
        Args:
            stopper: Stopper to terminate the inversion
            logger: Logger to write inversion log file
            niter_inner: number of iterations for the shrinkage loop
            niter_solver: number of iterations for the internal linear solver
            breg_weight: coefficient for the Bregman update b += beta * (R*x - d)
            linear_solver: linear solver to be used [CG, SD, LSQR]
            warm_start: run linear solver from previous solution of inner problem
            mod_tol: stop criterion for relative change of domain norm
        """
        super(SplitBregman, self).__init__(stopper=stopper, logger=kwargs.get("logger", None), name="Split-Bregman")
        
        # Model norm change stop criterion
        self.mod_tol = mod_tol
        
        # Logger for internal linear solver
        self.logger_lin_solv = None
        if self.logger is not None:
            if "/" in self.logger.file.name:
                folder = "/".join(self.logger.file.name.split("/")[:-1]) + "/"
            else:
                folder = ""
            filename = "inner_inv_" + self.logger.file.name.split("/")[-1]
            self.logger_lin_solv = Logger(folder + filename)
        
        self.niter_inner = niter_inner  # number of iterations for the shrinkage
        self.niter_solver = niter_solver  # number of iterations for the internal problem
        self.warm_start = warm_start
        if breg_weight > 1.:
            raise ValueError("ERROR! Bregman update weight has to be <= 1")
        self.breg_weight = float(abs(breg_weight))
        
        if linear_solver == 'CG':
            self.linear_solver = CG(BasicStopper(niter=self.niter_solver), logger=self.logger_lin_solv)
        elif linear_solver == 'SD':
            self.linear_solver = SD(BasicStopper(niter=self.niter_solver), logger=self.logger_lin_solv)
        elif linear_solver == 'LSQR':
            self.linear_solver = LSQR(BasicStopper(niter=self.niter_solver), logger=self.logger_lin_solv)
        else:
            raise ValueError('ERROR! Solver has to be CG, SD or LSQR')
        self.linear_solver.setDefaults(iter_sampling=1, flush_memory=True)
        
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, df_obj = %.2e, reg_obj = %.2e, rnorm = %.2e"
    
    def run(self, problem, verbose=False, inner_verbose=False, restart=False):
        """Run SplitBregman solver"""
        if type(problem) != GeneralizedLasso:
            raise TypeError("Input problem object must be a GeneralizedLasso")
        
        verbose = True if inner_verbose else verbose
        create_msg = verbose or self.logger
        
        # overriding save_grad variable
        self.save_grad = False
        
        # reset stopper before running the inversion
        self.stopper.reset()
        
        # initialize all the vectors and operators for Split-Bregman
        breg_b = problem.reg_op.range.clone().zero()
        breg_d = breg_b.clone()
        RL1x = breg_b.clone()  # store RegL1 * solution
        
        sb_mdl = problem.model.clone()
        if sb_mdl.norm() != 0.:
            self.warm_start = True
        sb_mdl_old = problem.model.clone()
        
        reg_op = np.sqrt(problem.eps) * problem.reg_op  # TODO can we avoid this?
        
        # inner problem
        linear_problem = LeastSquares(model=sb_mdl.clone(),
                                      data=superVector(problem.data, breg_d.clone()),
                                      op=Vstack(problem.op, reg_op),
                                      minBound=problem.minBound,
                                      maxBound=problem.maxBound,
                                      boundProj=problem.boundProj)
        
        if restart:
            self.restart.read_restart()
            outer_iter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            sb_mdl = self.restart.retrieve_vector("sb_mdl")
            if create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
        
        else:
            outer_iter = 0
            if create_msg:
                msg = 90 * '#' + '\n'
                msg += 12 * " " + "SPLIT-BREGMAN Solver log file\n"
                msg += 4 * " " + "Restart folder: %s\n" % self.restart.restart_folder
                msg += 4 * " " + "Inner iterations: %d\n" % self.niter_inner
                msg += 4 * " " + "Solver iterations: %d\n" % self.niter_solver
                msg += 4 * " " + "Problem: %s\n" % problem.name
                msg += 4 * " " + "L1 Regularizer weight: %.2e\n" % problem.eps
                msg += 4 * " " + "Bregman update weight: %.2e\n" % self.breg_weight
                if self.warm_start:
                    msg += 4 * " " + "Using warm start option for inner problem\n"
                msg += 90 * '#' + '\n'
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
                if self.logger_lin_solv:
                    msg = 90 * '#' + '\n'
                    msg += "\t\t\tSPLIT-BREGMAN Solver internal inversions log file\n"
                    msg += 90 * '#' + '\n'
                    self.logger_lin_solv.addToLog(msg)
        
        # Main iteration loop
        while True:
            obj0 = problem.get_obj(sb_mdl)
            # Saving previous model vector
            sb_mdl_old.copy(sb_mdl)
            
            if outer_iter == 0:
                initial_obj_value = obj0
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if create_msg:
                    msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.obj_terms[0],
                                           obj0 - problem.obj_terms[0],
                                           problem.get_rnorm(sb_mdl))
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog("\n" + msg)
            
            if self.logger_lin_solv:
                self.logger_lin_solv.addToLog("\n" + 12 * " " + "Outer iteration: %s"
                                              % (str(outer_iter).zfill(self.stopper.zfill)))
            
            if isnan(obj0):
                raise ValueError("Objective function values NaN!")
            
            if obj0 <= ZERO:
                print("Objective function is numerically zero! Stop the inversion")
                break
            
            self.save_results(outer_iter, problem, force_save=False)
            
            for iter_inner in range(self.niter_inner):
                
                if self.logger_lin_solv:
                    msg = 8 * " " + "starting inner iter %d with d = %.2e, b = %.2e" \
                          % (iter_inner, breg_d.norm(), breg_b.norm())
                    self.logger_lin_solv.addToLog("\n" + msg)
                
                # resetting inversion problem variables
                if not self.warm_start:
                    linear_problem.model.zero()
                # prior = d - b
                linear_problem.data.vecs[-1].copy(breg_b)
                linear_problem.data.vecs[-1].scaleAdd(breg_d, -1., 1.)
                linear_problem.data.vecs[-1].scale(np.sqrt(problem.eps))  # TODO can we avoid this?
                linear_problem.setDefaults()
                
                # solve inner problem
                self.linear_solver.run(linear_problem, verbose=inner_verbose)
                
                # compute RL1*x
                problem.reg_op.forward(False, linear_problem.model, RL1x)
                
                # update breg_d
                breg_d.copy(_soft_thresh(RL1x.clone() + breg_b, thresh=problem.eps))
                
                if self.logger_lin_solv:
                    msg = 8 * " " + "finished inner iter %d with sb_mdl = %.2e, RL1x = %.2e" \
                          % (iter_inner, linear_problem.model.norm(), RL1x.norm())
                    self.logger_lin_solv.addToLog(msg)
            
            # update breg_b
            breg_b.scaleAdd(RL1x, 1.0, self.breg_weight)
            breg_b.scaleAdd(breg_d, 1., -self.breg_weight)
            
            # Update SB model
            sb_mdl.copy(linear_problem.model)
            
            # check objective function
            obj1 = problem.get_obj(sb_mdl)
            sb_mdl_norm = sb_mdl.norm()
            chng_norm = sb_mdl_old.scaleAdd(sb_mdl, 1., -1.).norm()
            if chng_norm <= self.mod_tol * sb_mdl_norm:
                if create_msg:
                    msg = "Relative model change (%.4e) norm smaller than given tolerance (%.4e)" \
                          % (chng_norm, self.mod_tol * sb_mdl_norm)
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                break
            
            # iteration info
            outer_iter += 1
            if create_msg:
                msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.obj_terms[0],
                                       obj1 - problem.obj_terms[0],
                                       problem.get_rnorm(sb_mdl))
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            
            # saving in case of restart
            self.restart.save_parameter("iter", outer_iter)
            self.restart.save_vector("sb_mdl", sb_mdl)
            
            if self.stopper.run(problem=problem, iiter=outer_iter, verbose=verbose, initial_obj_value=initial_obj_value):
                break
        
        # writing last inverted model
        self.save_results(outer_iter, problem, force_save=True, force_write=True)
        
        # ending message and log file
        if create_msg:
            msg = 90 * '#' + '\n'
            msg += 12 * " " + "SPLIT-BREGMAN Solver log file end\n"
            msg += 90 * '#'
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog("\n" + msg)
        
        # Clear restart object
        self.restart.clear_restart()
