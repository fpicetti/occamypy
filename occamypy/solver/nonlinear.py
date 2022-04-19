from collections import deque
from math import isnan

import numpy as np
from occamypy.operator.linear import Scaling
from occamypy.problem.linear import LeastSquaresSymmetric
from occamypy.solver.base import Solver
from occamypy.solver.stopper import BasicStopper
from occamypy.solver.stepper import StrongWolfe, CvSrchStep, ParabolicStep
from occamypy.solver.linear import CGsym
from occamypy.utils import ZERO


# Beta functions
# grad=new gradient, grad0=old, dir=search direction
# From op SURVEY OF NONLINEAR CONJUGATE GRADIENT METHODS

def _betaFR(grad, grad0, dir, logger):
    """Fletcher and Reeves method"""
    # betaFR = sum(dprod(g,g))/sum(dprod(g0,g0))
    dot_grad = grad.dot(grad)
    dot_grad0 = grad0.dot(grad0)
    if dot_grad0 == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of previous gradient is zero!!!")
    else:
        beta = dot_grad / dot_grad0
    return beta


def _betaPRP(grad, grad0, dir, logger):
    """Polak, Ribiere, Polyak method"""
    # betaPRP = sum(dprod(g,g-g0))/sum(dprod(g0,g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = tmp1.dot(grad)
    dot_grad0 = grad0.dot(grad0)
    if dot_grad0 == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of previous gradient is zero!!!")
    else:
        beta = dot_num / dot_grad0
    return beta


def _betaHS(grad, grad0, dir, logger):
    """Hestenes and Stiefel"""
    # betaHS = sum(dprod(g,g-g0))/sum(dprod(d,g-g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = tmp1.dot(grad)
    dot_denom = tmp1.dot(dir)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = dot_num / dot_denom
    return beta


def _betaCD(grad, grad0, dir, logger):
    """Conjugate Descent"""
    # betaCD = -sum(dprod(g,g))/sum(dprod(d,g0))
    dot_num = grad.dot(grad)
    dot_denom = -grad0.dot(dir)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = dot_num / dot_denom
    return beta


def _betaLS(grad, grad0, dir, logger):
    """Liu and Storey"""
    # betaLS = -sum(dprod(g,g-g0))/sum(dprod(d,g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = tmp1.dot(grad)
    dot_denom = -grad0.dot(dir)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = dot_num / dot_denom
    return beta


def _betaDY(grad, grad0, dir, logger):
    """Dai and Yuan"""
    # betaDY = sum(dprod(g,g))/sum(dprod(d,g-g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = grad.dot(grad)
    dot_denom = tmp1.dot(dir)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = dot_num / dot_denom
    return beta


def _betaBAN(grad, grad0, dir, logger):
    """Bamigbola, Ali and Nwaeze"""
    # betaDY = sum(dprod(g,g-g0))/sum(dprod(g0,g-g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    dot_num = tmp1.dot(grad)
    dot_denom = tmp1.dot(grad0)
    if dot_denom == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        beta = -dot_num / dot_denom
    return beta


def _betaHZ(grad, grad0, dir, logger):
    """Hager and Zhang"""
    # betaN = sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))/sum(dprod(d,g-g0))
    tmp1 = grad.clone()
    # g-g0
    tmp1.scaleAdd(grad0, 1.0, -1.0)
    # sum(dprod(g-g0,g-g0))
    dot_diff_g_g0 = tmp1.dot(tmp1)
    # sum(dprod(d,g-g0))
    dot_dir_diff_g_g0 = tmp1.dot(dir)
    if dot_dir_diff_g_g0 == 0.:  # Avoid division by zero
        beta = 0.
        if logger:
            logger.addToLog("Setting beta to zero since norm of denominator is zero!!!")
    else:
        # g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0))
        tmp1.scaleAdd(dir, 1.0, -2.0 * dot_diff_g_g0 / dot_dir_diff_g_g0)
        # sum(dprod(g-g0-2*sum(dprod(g-g0,g-g0))*d/sum(dprod(d,g-g0)),g))
        dot_num = grad.dot(tmp1)
        # dot_num/sum(dprod(d,g-g0))
        beta = dot_num / dot_dir_diff_g_g0
    return beta


def _betaSD(grad, grad0, dir, logger):
    """Steepest descent"""
    beta = 0.
    return beta


class NLCG(Solver):
    """Non-Linear Conjugate Gradient and Steepest-Descent Solver object"""
    
    def __init__(self, stopper, stepper=ParabolicStep(), beta_type="FR", **kwargs):
        """
        NLCG constructor

        Args:
            stopper: Stopper to terminate the inversion
            stepper: Stepper object to perform line-search step
            beta_type: beta function to be used
            logger: Logger to write inversion log file
        """
        
        super(NLCG, self).__init__(stopper=stopper, logger=kwargs.get("logger", None))
        
        # Defining stepper object
        self.stepper = stepper
        
        # print formatting
        self.iter_msg = "iter = %s, obj = %.2e, rnorm = %.2e, gnorm = %.2e, feval = %s, geval = %s"
    
        # Beta function to use during the inversion
        self.beta_type = beta_type.upper()
        if self.beta_type not in ["FR", "PRP", "HS", "CD", "LS", "DY", "BAN", "HZ", "SD"]:
            raise ValueError("Beta function type has to be one of [FR, PRP, HS, CD, LS, DY, BAN, HZ, SD]")
        
        elif beta_type == "PRP":
            self.beta_func = lambda g, g0, d: _betaPRP(g, g0, d, self.logger)
        elif beta_type == "HS":
            self.beta_func = lambda g, g0, d: _betaHS(g, g0, d, self.logger)
        elif beta_type == "CD":
            self.beta_func = lambda g, g0, d: _betaCD(g, g0, d, self.logger)
        elif beta_type == "LS":
            self.beta_func = lambda g, g0, d: _betaLS(g, g0, d, self.logger)
        elif beta_type == "DY":
            self.beta_func = lambda g, g0, d: _betaDY(g, g0, d, self.logger)
        elif beta_type == "BAN":
            self.beta_func = lambda g, g0, d: _betaBAN(g, g0, d, self.logger)
        elif beta_type == "HZ":
            self.beta_func = lambda g, g0, d: _betaHZ(g, g0, d, self.logger)
        elif beta_type == "SD":
            self.beta_func = lambda g, g0, d: _betaSD(g, g0, d, self.logger)
        else:
            self.beta_func = lambda g, g0, d: _betaFR(g, g0, d, self.logger)
    
    def run(self, problem, verbose=False, restart=False):
        """Run NLCG solver"""
        
        create_msg = verbose or self.logger
        
        # Resetting stopper before running the inversion
        self.stopper.reset()
        
        if not restart:
            if create_msg:
                msg = 90 * "#" + "\n"
                msg += 12 * " " + "Nonlinear %s Solver log file\n" % ("SD" if self.beta_type == "SD" else "CG")
                msg += 4 * " " + "Restart folder: %s\n" % self.restart.restart_folder
                if self.beta_type != "SD":
                    msg += 4 * " " + "Conjugate method used: %s\n" % self.beta_type
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace("log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
            
            # Setting internal vectors (model, search direction, and previous gradient vectors)
            prblm_mdl = problem.get_model()
            cg_mdl = prblm_mdl.clone()
            cg_dmodl = prblm_mdl.clone()
            cg_dmodl.zero()
            cg_grad0 = cg_dmodl.clone()
            
            # Other internal variables
            beta = 0.0
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
            iiter = self.restart.retrieve_parameter("iter")
            self.stepper.alpha = self.restart.retrieve_parameter("alpha")
            initial_obj_value = self.restart.retrieve_parameter(
                "obj_initial")  # Retrieving initial objective function value
            cg_mdl = self.restart.retrieve_vector("cg_mdl")
            cg_dmodl = self.restart.retrieve_vector("cg_dmodl")
            cg_grad0 = self.restart.retrieve_vector("cg_grad0")
            # Setting the model and residuals to avoid residual twice computation
            problem.set_model(cg_mdl)
            prblm_mdl = problem.get_model()
            # Setting residual vector to avoid its unnecessary computation
            problem.set_residual(self.restart.retrieve_vector("prblm_res"))
        
        # Common variables unrelated to restart
        prev_mdl = prblm_mdl.clone().zero()
        
        while True:
            # Computing objective function
            obj0 = problem.get_obj(cg_mdl)  # Compute objective function value
            prblm_res = problem.get_res(cg_mdl)  # Compute residuals
            prblm_grad = problem.get_grad(cg_mdl)  # Compute the gradient
            if iiter == 0:
                initial_obj_value = obj0  # For relative objective function value
                # Saving objective function value
                self.restart.save_parameter("obj_initial", initial_obj_value)
                # iteration info
                if create_msg:
                    msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.get_rnorm(cg_mdl),
                                           problem.get_gnorm(cg_mdl),
                                           str(problem.get_fevals()).zfill(self.stopper.zfill + 1),
                                           str(problem.get_gevals()).zfill(self.stopper.zfill))
                    # Writing on log file
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj0) or isnan(prblm_grad.norm()):
                    raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
            if prblm_grad.norm() == 0.:
                print("Gradient vanishes identically")
                break
            
            # Saving results
            self.save_results(iiter, problem, force_save=False)
            # Keeping current inverted model
            prev_mdl.copy(prblm_mdl)
            
            if iiter >= 1:
                beta = self.beta_func(prblm_grad, cg_grad0, cg_dmodl)
                if beta < 0.:
                    if self.logger:
                        self.logger.addToLog("Beta negative setting to zero: beta value=%s!!!" % beta)
                    beta = 0.
            if self.beta_type != "SD":
                if self.logger:
                    self.logger.addToLog("beta coefficient: %s" % beta)
            
            # dmodl = beta*dmodl - grad
            cg_dmodl.scaleAdd(prblm_grad, beta, -1.0)
            # grad0 = grad
            cg_grad0.copy(prblm_grad)
            # Calling line search
            alpha, success = self.stepper.run(problem, cg_mdl, cg_dmodl, self.logger)
            if not success:
                if create_msg:
                    msg = "Stepper couldn't find a proper step size, will terminate solver"
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                break
            
            # Increasing iteration counter
            iiter = iiter + 1
            obj1 = problem.get_obj(cg_mdl)  # Compute objective function value
            # Redundant tests on verifying convergence
            if obj0 <= obj1:
                if create_msg:
                    msg = "Objective function at new point greater or equal than previous one:\n\t" \
                          "obj_new = %.5e\tobj_cur = %.5e\n" \
                          "Potential issue in the stepper or in revaluation of objective function! Solver will stop!" \
                          % (obj1, obj0)
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                break
            
            # Saving current model and previous search direction in case of restart
            self.restart.save_parameter("iter", iiter)
            self.restart.save_parameter("alpha", alpha)
            self.restart.save_vector("cg_mdl", cg_mdl)
            self.restart.save_vector("cg_dmodl", cg_dmodl)
            self.restart.save_vector("cg_grad0", cg_grad0)
            # Saving data space vectors
            self.restart.save_vector("prblm_res", prblm_res)
            # iteration info
            if create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.get_rnorm(cg_mdl),
                                       problem.get_gnorm(cg_mdl),
                                       str(problem.get_fevals()).zfill(self.stopper.zfill + 1),
                                       str(problem.get_gevals()).zfill(self.stopper.zfill))
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            
            # Check if either objective function value or gradient norm is NaN
            if isnan(obj1) or isnan(prblm_grad.norm()):
                raise ValueError("ERROR! Either gradient norm or objective function value NaN!")
            if self.stopper.run(problem=problem, iiter=iiter, verbose=verbose, initial_obj_value=initial_obj_value):
                break
        
        # Writing last inverted model
        self.save_results(iiter, problem, force_save=True, force_write=True)
        if create_msg:
            msg = 90 * "#" + "\n"
            msg += 4 * " " + "Nonlinear %s Solver log file end\n" % ("SD" if self.beta_type == "SD" else "CG")
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog(msg)
        
        # Clear restart object
        self.restart.clear_restart()
        
        return


class TNewton(Solver):
    """Truncated Newton/Gauss-Newton solver object"""
    
    def __init__(self, stopper, niter_max, Hessian, stepper=None, niter_min=None, warm_start=True, Newton_prefix=None, **kwargs):
        """
        Truncated-Newton constructor

        Args:
            stopper: Stopper to terminate the inversion
            niter_max: number of iterations for solving Newton system when starting with a zero initial domain
            Hessian: Hessian of the modeling operator. It must contain a set_background function to set domain vector on which the Hessian is computed.
                Note that the symmetric solver will be used. Hence, this operator must be symmetric.
            stepper: Stepper object to perform line-search step
            niter_min: number of iterations for solving Newton system when linear inversion starts from previous search direction
            warm_start: start the linear Hessian inversion from the previous search direction if aligned with the current gradient
        """
        super(TNewton, self).__init__(stopper=stopper, logger=kwargs.get("logger", None))
        # Defining stopper
        self.stopper = stopper  # Stopper for non-linear problem
        # Setting maximum and minimum number of iterations
        self.niter_max = niter_max
        self.niter_min = niter_max
        # Setting linear inversion iterations
        if niter_min is not None:
            if niter_min <= niter_max:
                raise ValueError(
                    "niter_min of %d must be smaller or equal than niter_max of %d." % (niter_min, niter_max))
            self.niter_min = niter_min
        # Defining stepper object
        self.stepper = stepper if stepper is not None else CvSrchStep()
        # Warm starts requested?
        self.warm_start = warm_start
        # Hessian operator
        if Hessian is not None:
            if "set_background" not in dir(Hessian):
                raise AttributeError("Hessian operator must have a set_background function.")
        # Setting linear solver for solving Newton system and problem class
        StopLin = BasicStopper(niter=self.niter_max)
        self.lin_solver = CGsym(StopLin)
        self.NewtonPrblm = LeastSquaresSymmetric(Hessian.domain.clone(), Hessian.domain.clone(), Hessian)
    
    def run(self, problem, verbose=False, restart=False):
        """Run Truncated Newton solver"""
        return


class LBFGS(Solver):
    """Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) solver"""
    
    def __init__(self, stopper, stepper=None, save_alpha=False, m_steps=None, H0=None, save_est=False, **kwargs):
        """
        LBFGS constructor
        
        Args:
            stopper: Stopper to terminate the inversion
            stepper: Stepper object to perform line-search step
            save_alpha: use previous step-length value as initial guess. Otherwise, the algorithm starts from 1.
            m_steps: maximum number of steps to store to estimate the inverse Hessian (by default it runs BFGS method)
            H0: initial estimated Hessian inverse (by default it assumes an identity operator)
            logger: Logger to write inversion log file
            save_est: save inverse Hessian estimate vectors (self.prefix must not be None)
        """
        
        super(LBFGS, self).__init__(stopper=stopper, logger=kwargs.get("logger", None))
        
        # Defining stepper object
        self.stepper = stepper if stepper is not None else CvSrchStep()

        # LBFGS-specific parameters
        self.save_alpha = save_alpha
        self.H0 = H0
        self.m_steps = m_steps
        self.save_est = save_est
        self.tmp_vector = None  # op copy of the model vector will be create when the function run is invoked
        self.iistep = 0  # necessary to re-used the estimated hessian inverse from previous runs
        # print formatting
        self.iter_msg = "iter = %s, obj = %.2e, rnorm = %.2e, gnorm = %.2e, feval = %s, geval = %s"
    
    def save_hessian_estimate(self, index, iiter):
        """Function to save current vector of estimated Hessian inverse"""
        # index of the step to save
        if self.prefix is not None and self.save_est:
            step_filename = self.prefix + "step_vector_%s.H" % iiter
            grad_diff_filename = self.prefix + "grad_diff_vector_%s.H" % iiter
            self.step_vectors[index].writeVec(step_filename)
            self.grad_diff_vectors[index].writeVec(grad_diff_filename)
    
    def check_rho(self, denom_dot, step_index, iiter):
        """Function to check scaling factor of Hessian inverse estimate"""
        if denom_dot == 0.:
            if self.m_steps is not None:
                self.rho[step_index] = 0.
            else:
                self.rho.append(0.)
            msg = "Skipping update to estimated Hessian; y vector orthogonal to s vector at iteration %s" % iiter
            if self.logger:
                self.logger.addToLog(msg)
        elif denom_dot < 0.:
            if self.m_steps is not None:
                self.rho[step_index] = 0.
            else:
                self.rho.append(0.)
            msg = "Skipping update to estimated Hessian; not positive at iteration %s" % iiter
            if self.logger:
                self.logger.addToLog(msg)
        else:
            if self.m_steps is not None:
                self.rho[step_index] = 1.0 / denom_dot
            else:
                self.rho.append(1.0 / denom_dot)
            # Saving current update for inverse Hessian estimate (i.e., gradient-difference and model-step vectors)
            self.save_hessian_estimate(step_index, iiter)
        return
    
    def BFGSMultiply(self, dmodl, grad, iiter):
        """Function to apply approximated inverse Hessian"""
        # Array containing dot-products
        if self.m_steps is not None:
            alpha = [0.0] * self.m_steps
            # Handling of limited memory
            if iiter <= self.m_steps:
                initial_point = 0
            else:
                initial_point = iiter % self.m_steps
            # Right step list
            rloop = deque(range(0, min(iiter, self.m_steps)))
            rloop.reverse()
            # Rotate the list
            rloop.rotate(initial_point)
            # Left step list
            lloop = deque(range(0, min(iiter, self.m_steps)))
            # Rotate the list
            lloop.rotate(-initial_point)
        else:
            alpha = [0.0] * iiter
            rloop = deque(range(0, iiter))
            rloop.reverse()
            lloop = deque(range(0, iiter))
        # r = -grad
        dmodl.copy(grad)
        dmodl.scale(-1.0)
        # Apply right-hand series of operators
        for ii in rloop:
            # Check positivity, if not true skip the update
            if self.rho[ii] > 0.0:
                # alpha_i=rho_i*s_i'r
                alpha[ii] = self.rho[ii] * self.step_vectors[ii].dot(dmodl)
                # r=r-alpha_i*y_i
                dmodl.scaleAdd(self.grad_diff_vectors[ii], 1.0, -alpha[ii])
        # Comput center (If not provide Identity matrix is assumed)
        # r=H0r
        if self.H0 is not None:
            # Apply a forward of the initial Hessian estimate
            self.H0.forward(False, dmodl, self.tmp_vector)
            dmodl.copy(self.tmp_vector)
        # Apply left-hand series of operators
        for ii in lloop:
            # Check positivity, if not true skip the update
            if self.rho[ii] > 0.0:
                # beta=rhoiyi'r
                beta = self.rho[ii] * self.grad_diff_vectors[ii].dot(dmodl)
                dmodl.scaleAdd(self.step_vectors[ii], 1.0, alpha[ii] - beta)
        return
    
    def run(self, problem, verbose=False, keep_hessian=False, restart=False):
        """
        Run LBFGS solver
        
        Args:
            problem: problem to be minimized
            verbose: verbosity flag
            keep_hessian: use hessian inverse estimate build from previous runs
            restart: restart previously crashed inversion
        """
        create_msg = verbose or self.logger
        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Getting model vector
        prblm_mdl = problem.get_model()
        # Preliminary variables for Hessian inverse estimation
        if not keep_hessian or "rho" not in dir(self):
            if self.m_steps is not None:
                self.step_vectors = [None] * self.m_steps  # s_i vectors
                self.grad_diff_vectors = [None] * self.m_steps  # y_i vectors
                self.rho = [None] * self.m_steps  # Scalar term necessary for Hessian inverse estimation
            else:
                self.step_vectors = []  # s_i vectors
                self.grad_diff_vectors = []  # y_i vectors
                self.rho = []  # Scalar term necessary for Hessian inverse estimation
            self.iistep = 0
        
        if not restart:
            if create_msg:
                msg = 90 * "#" + "\n"
                if self.m_steps is not None:
                    msg += 12 * " " + "L-BFGS Solver log file\n"
                    msg += 4 * " " + "Max steps to invert the Hessian: %d \n" % self.m_steps
                else:
                    msg += 12 * " " + "BFGS Solver log file\n"
                # Printing restart folder
                msg += 4 * " " + "Restart folder: %s\n" % self.restart.restart_folder
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace("log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
            
            # Setting internal vectors (model, search direction, and previous gradient vectors)
            bfgs_mdl = prblm_mdl.clone()
            bfgs_dmodl = prblm_mdl.clone()
            bfgs_dmodl.zero()
            bfgs_grad0 = bfgs_dmodl.clone()
            
            # Other internal variables
            iiter = 0
        else:
            # Retrieving parameters and vectors to restart the solver
            msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            self.restart.read_restart()
            iiter = self.restart.retrieve_parameter("iter")
            self.stepper.alpha = self.restart.retrieve_parameter("alpha")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            bfgs_mdl = self.restart.retrieve_vector("bfgs_mdl")
            bfgs_dmodl = self.restart.retrieve_vector("bfgs_dmodl")
            bfgs_grad0 = self.restart.retrieve_vector("bfgs_grad0")
            # Setting the model and residuals to avoid residual twice computation
            problem.set_model(bfgs_mdl)
            prblm_mdl = problem.get_model()
            # Setting residual vector to avoid its unnecessary computation
            problem.set_residual(self.restart.retrieve_vector("prblm_res"))
            # Retrieving Hessian inverse estimate
            self.rho = self.restart.retrieve_parameter("rho")
            self.iistep = self.restart.retrieve_parameter("iistep")
            for istep in range(min(iiter, self.iistep)):
                if self.m_steps is not None:
                    if istep < self.m_steps:
                        self.grad_diff_files[istep] = self.restart.retrieve_vector("grad_diff_vectors%s" % istep)
                        self.step_files[istep] = self.restart.retrieve_vector("step_vectors%s.H" % istep)
                else:
                    self.grad_diff_files.append(self.restart.retrieve_vector("grad_diff_vectors%s" % istep))
                    self.step_files.append(self.restart.retrieve_vector("step_vectors%s" % istep))
        
        # Common variables unrelated to restart
        self.tmp_vector = bfgs_dmodl.clone()
        self.tmp_vector.zero()
        prev_mdl = prblm_mdl.clone().zero()
        
        # Inversion loop
        while True:
            # Computing objective function
            obj0 = problem.get_obj(bfgs_mdl)  # Compute objective function value
            prblm_res = problem.get_res(bfgs_mdl)  # Compute residuals
            prblm_grad = problem.get_grad(bfgs_mdl)  # Compute the gradient
            if iiter == 0:
                initial_obj_value = obj0  # For relative objective function value
                # Saving objective function value
                self.restart.save_parameter("obj_initial", initial_obj_value)
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       obj0,
                                       problem.get_rnorm(bfgs_mdl),
                                       problem.get_gnorm(bfgs_mdl),
                                       str(problem.get_fevals()).zfill(self.stopper.zfill + 1),
                                       str(problem.get_gevals()).zfill(self.stopper.zfill))
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj0) or isnan(prblm_grad.norm()):
                    raise ValueError("Either gradient norm or objective function value NaN!")
            if prblm_grad.norm() == 0.:
                print("Gradient vanishes identically")
                break
            
            # Saving results
            self.save_results(iiter, problem, force_save=False)
            # Saving current inverted model
            prev_mdl.copy(prblm_mdl)
            
            # Applying approximated Hessian inverse
            msg = "Appplying inverse Hessian estimate"
            if self.m_steps is not None:
                msg += "\nCurrent inverse dot-products of BFGS estimation vectors %s" \
                       % (self.rho[0:min(self.m_steps, self.iistep)])
            else:
                if len(self.rho) > 0:
                    msg += "\nCurrent inverse dot-products of BFGS estimation vectors %s" % self.rho
            if self.logger:
                self.logger.addToLog(msg)
            self.BFGSMultiply(bfgs_dmodl, prblm_grad, self.iistep)
            msg = "Done applying inverse Hessian estimate"
            if self.logger:
                self.logger.addToLog(msg)
            
            # grad0 = grad
            bfgs_grad0.copy(prblm_grad)
            # Calling line search
            alpha, success = self.stepper.run(problem, bfgs_mdl, bfgs_dmodl, self.logger)
            if not success:
                msg = "Stepper couldn't find a proper step size, will terminate solver"
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                break
            
            obj1 = problem.get_obj(bfgs_mdl)  # Compute objective function value
            # Redundant tests on verifying convergence
            if obj0 <= obj1:
                msg = "Objective function at new point greater or equal than previous one: obj_fun_old=%s obj_fun_new=%s\n" \
                      "Potential issue in the stepper or in revaluation of objective function!" % (obj0, obj1)
                if self.logger:
                    self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                raise ValueError(msg)
            
            # Compute new gradient
            prblm_grad = problem.get_grad(bfgs_mdl)
            # Compute updates for estimated Hessian inverse
            if self.m_steps is not None:
                # LBFGS
                step_index = self.iistep % self.m_steps  # Modulo to handle limited memory
                # yn+1=gn+1-gn
                self.grad_diff_vectors[step_index] = bfgs_grad0.clone()
                self.grad_diff_vectors[step_index].scaleAdd(prblm_grad, -1.0, 1.0)
                # sn+1=xn+1-xn = alpha * dmodl
                self.step_vectors[step_index] = bfgs_dmodl.clone()
                self.step_vectors[step_index].scale(alpha)
            else:
                # BFGS
                step_index = self.iistep
                # yn+1=gn+1-gn
                self.grad_diff_vectors.append(bfgs_grad0.clone())
                self.grad_diff_vectors[step_index].scaleAdd(prblm_grad, -1.0, 1.0)
                # sn+1=xn+1-xn = alpha * dmodl
                self.step_vectors.append(bfgs_dmodl.clone())
                self.step_vectors[step_index].scale(alpha)
            # rhon+1=1/yn+1'sn+1
            denom_dot = self.grad_diff_vectors[step_index].dot(self.step_vectors[step_index])
            # Checking rho
            self.check_rho(denom_dot, step_index, self.iistep)
            
            # Making first step-length value Hessian guess if not provided by user
            if iiter == 0 and alpha != 1.0:
                self.restart.save_parameter("fist_alpha", alpha)
                self.H0 = Scaling(bfgs_dmodl, alpha) if self.H0 is None else self.H0 * Scaling(bfgs_dmodl, alpha)
                if self.logger:
                    self.logger.addToLog("First step-length value added to first Hessian inverse estimate!")
                self.stepper.alpha = 1.0
            
            # Increasing iteration counter
            iiter = iiter + 1
            self.iistep += 1
            
            # Using alpha = 1.0 after first iteration
            if iiter != 0 and not self.save_alpha:
                self.stepper.alpha = 1.0
            
            # Saving current model and previous search direction in case of restart
            self.restart.save_parameter("iter", iiter)
            self.restart.save_parameter("alpha", alpha)
            self.restart.save_vector("bfgs_mdl", bfgs_mdl)
            self.restart.save_vector("bfgs_dmodl", bfgs_dmodl)
            self.restart.save_vector("bfgs_grad0", bfgs_grad0)
            # Saving Inverse Hessian estimate for restart
            self.restart.save_parameter("rho", self.rho)
            self.restart.save_parameter("iistep", self.iistep)
            self.restart.save_vector("grad_diff_vectors%s" % step_index, self.grad_diff_vectors[step_index])
            self.restart.save_vector("step_vectors%s" % step_index, self.step_vectors[step_index])
            # Saving data space vectors
            self.restart.save_vector("prblm_res", prblm_res)
            
            # iteration info
            msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                   obj1,
                                   problem.get_rnorm(bfgs_mdl),
                                   problem.get_gnorm(bfgs_mdl),
                                   str(problem.get_fevals()).zfill(self.stopper.zfill + 1),
                                   str(problem.get_gevals()).zfill(self.stopper.zfill))
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
            if self.m_steps is not None:
                msg += 12 * " " + "L-BFGS Solver log file end\n"
            else:
                msg += 12 * " " + "BFGS Solver log file end\n"
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace("log file ", ""))
            if self.logger:
                self.logger.addToLog(msg)
        self.restart.clear_restart()
        # Resetting inverse Hessian matrix
        if not keep_hessian:
            self.H0 = None
        del self.tmp_vector
        self.tmp_vector = None


class LBFGSB(Solver):
    """
    Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds (L-BFGS-B) Solver

    References:
        Implementation inspired by: https://github.com/bgranzow/L-BFGS-B.git
    """
    
    def __init__(self, stopper, stepper=StrongWolfe(), m_steps=np.inf, **kwargs):
        """
        LBFGSB constructor
        
        Args:
            stopper: Stopper to terminate the inversion
            stepper: Stepper object to perform line-search step
            m_steps: maximum number of steps to store to estimate the inverse Hessian (by default it runs BFGS method)
            logger: Logger to write inversion log file
        """
        super(LBFGSB, self).__init__(stopper=stopper, logger=kwargs.get("logger", None))
        
        # Defining stepper object
        self.stepper = stepper
        
        # LBFGSB-specific parameters
        self.m_steps = m_steps
        self.epsmch = None
        # print formatting
        self.iter_msg = "iter = %s, obj = %.2e, rnorm = %.2e, gnorm = %.2e, feval = %s, geval = %s"
    
    def get_breakpoints(self, bfgsb_dmodl, bfgsb_mdl, prblm_grad, minBound, maxBound):
        """
        Compute the breakpoint variables needed for the Cauchy point.
        
        Notes:
            Equations (4.1),(4.2), and F in Algorithm CP: Initialize.
            
        Args:
            bfgsb_dmodl: model perturbation vector (it contains the search direction)
            bfgsb_mdl: model vector
            prblm_grad: problem gradient vector
            minBound: lower bound vector
            maxBound: upper bound vector

        Returns:
            t_vec: breakpoint numpy.ndarray
            F: indices that sort t_vec from low to hig
        """
        n_mod = bfgsb_mdl.size
        t_vec = bfgsb_mdl.clone().zero()
        bfgsb_dmodl.copy(prblm_grad)
        bfgsb_dmodl.scale(-1.0)
        # Getting NdArrays of variables
        g = prblm_grad.getNdArray().ravel()
        x = bfgsb_mdl.getNdArray().ravel()
        t = t_vec.getNdArray().ravel()
        l = minBound.getNdArray().ravel()
        u = maxBound.getNdArray().ravel()
        d = bfgsb_dmodl.getNdArray().ravel()
        # Largest representable number
        realmax = np.finfo(self.epsmch).max
        for ii in range(n_mod):
            if g[ii] < 0.0:
                t[ii] = (x[ii] - u[ii]) / g[ii]
            elif g[ii] > 0.0:
                t[ii] = (x[ii] - l[ii]) / g[ii]
            else:
                t[ii] = realmax
            if t[ii] < self.epsmch:
                d[ii] = 0.0
        F = np.argsort(t)
        return t_vec, F
    
    def get_cauchy_point(self, bfgsb_mdl_cauchy, bfgsb_dmodl, bfgsb_mdl, prblm_grad, minBound, maxBound, W, M, theta):
        """
        Compute the generalized Cauchy point
        
        References:
            Algorithm CP, Pages 8-9
        
        Args:
            bfgsb_mdl_cauchy: model vector of Cauchy point
            bfgsb_dmodl: model perturbation vector (it contains the search direction)
            bfgsb_mdl: model vector
            prblm_grad: problem gradient vector
            minBound: lower bound vector
            maxBound: upper bound vector
            W:
            M:
            theta:

        Returns:
            c: initialization vector for subspace minimization
        """
        t_vec, F = self.get_breakpoints(bfgsb_dmodl, bfgsb_mdl, prblm_grad, minBound, maxBound)
        # xc = x;
        bfgsb_mdl_cauchy.copy(bfgsb_mdl)
        # Getting vector arrays
        d = bfgsb_dmodl.getNdArray().ravel()
        tt = t_vec.getNdArray().ravel()
        xc = bfgsb_mdl_cauchy.getNdArray().ravel()
        x = bfgsb_mdl.getNdArray().ravel()
        l = minBound.getNdArray().ravel()
        u = maxBound.getNdArray().ravel()
        g = prblm_grad.getNdArray().ravel()
        
        # Main algorithm
        p = np.matmul(W.T, d)
        c = np.zeros((W.shape[1]))
        fp = -bfgsb_dmodl.dot(bfgsb_dmodl)
        fpp = -theta * fp - np.sum(p * np.matmul(M, p))
        fpp0 = -theta * fp
        dt_min = -fp / (fpp + self.epsmch)
        t_old = 0
        for jj in range(bfgsb_mdl_cauchy.size):
            ii = jj
            if tt[F[ii]] > 0:
                break
        b = F[ii]
        t = tt[b]
        dt = t - t_old
        
        while dt_min > dt and ii < bfgsb_mdl_cauchy.size:
            if d[b] > 0.0:
                xc[b] = u[b]
            elif d[b] < 0.0:
                xc[b] = l[b]
            zb = xc[b] - x[b]
            c += dt * p
            gb = g[b]
            wbt = W[b, :]
            fp = fp + dt * fpp + gb * gb + theta * gb * zb - gb * np.sum(wbt * np.matmul(M, c))
            fpp = fpp - theta * gb * gb - 2.0 * gb * np.sum(wbt * np.matmul(M, p)) - gb * gb * np.sum(
                wbt * np.matmul(M, wbt))
            fpp = max(self.epsmch * fpp0, fpp)
            p += gb * wbt
            d[b] = 0.0
            dt_min = -fp / fpp
            t_old = t
            ii += 1
            if ii < bfgsb_mdl_cauchy.size:
                b = F[ii]
                t = tt[b]
                dt = t - t_old
        
        # Perform final updates
        dt_min = max(dt_min, 0)
        t_old = t_old + dt_min
        for jj in range(ii, bfgsb_mdl_cauchy.size):
            idx = F[jj]
            xc[idx] = x[idx] + t_old * d[idx]
        c += dt_min * p
        return c
    
    def find_alpha(self, l, u, xc, du, free_vars_idx):
        """
        Equation (5.8), Page 8.
        :return:
        :alpha_star positive scaling parameter.
        """
        alpha_star = 1.0
        n = len(free_vars_idx)
        for ii in range(n):
            idx = free_vars_idx[ii]
            if (du[ii] > 0.0):
                alpha_star = min(alpha_star, (u[idx] - xc[idx]) / du[ii])
            else:
                alpha_star = min(alpha_star, (l[idx] - xc[idx]) / (du[ii] + self.epsmch))
        ######## Check this!
        if alpha_star < 0.0:
            alpha_star = 0.0
        return alpha_star
    
    def subspace_min(self, bfgsb_dmodl, bfgsb_mdl, prblm_grad, minBound, maxBound, bfgsb_mdl_cauchy, c, W, M, theta):
        """
        Subspace minimization for the quadratic domain over free variables.
        
        References:
            Direct Primal Method, Page 12
        
        Args:
            bfgsb_dmodl: model perturbation vector (it contains the search direction)
            bfgsb_mdl: model vector
            prblm_grad: problem gradient vector
            minBound: lower bound vector
            maxBound: upper bound vector
            bfgsb_mdl_cauchy: model vector of Cauchy point
            c:
            W:
            M:
            theta:
            
        Returns:
            free_var: Flag whether there are free variables or not
        """
        # Setting the line search flag to true by default
        free_var = True
        
        # Getting arrays and parameters
        n_mod = bfgsb_mdl.size
        xc = bfgsb_mdl_cauchy.getNdArray().ravel()
        x = bfgsb_mdl.getNdArray().ravel()
        l = minBound.getNdArray().ravel()
        u = maxBound.getNdArray().ravel()
        g = prblm_grad.getNdArray().ravel()
        xbar = bfgsb_dmodl.getNdArray().ravel()
        
        # Finding free variables
        free_vars_idx = []
        Z = []
        for ii in range(n_mod):
            if xc[ii] != u[ii] and xc[ii] != l[ii]:
                free_vars_idx.append(ii)
                unit_tmp = np.zeros((n_mod))
                unit_tmp[ii] = 1.0
                Z.append(unit_tmp)
        Z = np.array(Z).T
        
        num_free_vars = len(free_vars_idx)
        
        if num_free_vars == 0:
            bfgsb_mdl.copy(bfgsb_mdl_cauchy)  # xbar = x = xc
            free_var = False
            return free_var
        
        # Compute W^T Z, the restriction of W to free variables
        WTZ = np.matmul(W.T, Z)
        
        # Compute the reduced gradient of mk restricted to free variables
        rr = g + theta * (xc - x) - np.matmul(W, np.matmul(M, c))
        r = np.zeros((num_free_vars))
        for ii in range(num_free_vars):
            r[ii] = rr[free_vars_idx[ii]]
        
        # Forming intermediate variables
        invtheta = 1.0 / theta
        v = np.matmul(M, np.matmul(WTZ, r))
        N = invtheta * np.matmul(WTZ, WTZ.T)
        N = np.eye(N.shape[0]) - np.matmul(M, N)
        v = np.linalg.solve(N, v)
        du = - invtheta * (r + invtheta * np.matmul(WTZ.T, v))
        
        # find alpha star
        alpha_star = self.find_alpha(l, u, xc, du, free_vars_idx)
        
        # compute the subspace minimization
        bfgsb_dmodl.copy(bfgsb_mdl_cauchy)  # xbar = xc
        for ii in range(num_free_vars):
            idx = free_vars_idx[ii]
            xbar[idx] += alpha_star * du[ii]
        # Returning new search direction
        bfgsb_dmodl.scaleAdd(bfgsb_mdl, 1.0, -1.0)
        
        return free_var
    
    def form_W(self, Y_mat, theta, S_mat):
        """
        :param Y_mat: list containing y_k vectors
        :param theta: theta parameters within the L-BFGS-B algorithm
        :param S_mat: list containing s_k vectors
        :return:
        :W matrix of the L-BFGS Hessian estimate
        :Y matrix containing the y_k vectors
        :S matrix containing the s_k vectors
        """
        Y = np.array([yk.getNdArray().ravel() for yk in Y_mat]).T
        S = np.array([sk.getNdArray().ravel() for sk in S_mat]).T
        W = np.concatenate((Y, theta * S), axis=1)
        return W, Y, S
    
    def max_step(self, bfgsb_mdl, bfgsb_dmodl, minBound, maxBound):
        """
        Method for determine maximum feasible step length
        
        Args:
            bfgsb_mdl: model vector
            bfgsb_dmodl: model perturbation vector (it contains the search direction)
            minBound: lower bound vector
            maxBound: upper bound vector
        
        Returns:
            max_alpha: maximum feasible step length
        """
        n_mod = bfgsb_mdl.size
        x = bfgsb_mdl.getNdArray().ravel()
        l = minBound.getNdArray().ravel()
        u = maxBound.getNdArray().ravel()
        d = bfgsb_dmodl.getNdArray().ravel()
        max_alpha = 1e18
        for ii in range(n_mod):
            d1 = d[ii]
            if d1 < 0.0:
                d2 = l[ii] - x[ii]
                if d2 >= 0.0:
                    max_alpha = 0.0
                elif d1 * max_alpha < d2:
                    max_alpha = d2 / (d1 + self.epsmch)
            elif d1 > 0.0:
                d2 = u[ii] - x[ii]
                if d2 <= 0.0:
                    max_alpha = ZERO
                elif d1 * max_alpha > d2:
                    max_alpha = d2 / (d1 + self.epsmch)
        return max_alpha
    
    def run(self, problem, verbose=False, restart=False):
        """
        Run L-BFGS-B solver
        
        Args:
            problem: problem to be minimized
            verbose: verbosity flag
            restart: restart previously crashed inversion
        """
        create_msg = verbose or self.logger
        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Getting model vector
        prblm_mdl = problem.get_model()
        
        # Obtaining bounds from problem
        minBound = problem.bounds.minBound
        maxBound = problem.bounds.maxBound
        if minBound is None:
            raise ValueError("Minimum bound vector must be provided when instantiating the problem object!")
        if maxBound is None:
            raise ValueError("Maximum bound vector must be provided when instantiating the problem object!")
        
        # Setting matrices (lists) of gradient differences and taken steps
        S_mat = []
        Y_mat = []
        # Numpy Arrays containing the matrices necessary for the L-BFGS-B internal optimization problems
        W = np.zeros((prblm_mdl.size, 1))
        M = np.zeros((1, 1))
        
        if not restart:
            if create_msg:
                msg = 90 * "#" + "\n"
                if self.m_steps is not None:
                    msg += 12 * " " + "L-BFGS-B Solver log file\n"
                    msg += 4 * " " + "Max steps to invert the Hessian: %d \n" % self.m_steps
                else:
                    msg += 12 * " " + "BFGS-B Solver log file\n"
                # Printing restart folder
                msg += 4 * " " + "Restart folder: %s\n" % self.restart.restart_folder
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace("log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
            
            # Setting internal vectors (model, search direction, and previous gradient vectors)
            bfgsb_mdl = prblm_mdl.clone()
            bfgsb_dmodl = prblm_mdl.clone()
            bfgsb_dmodl.zero()
            bfgsb_grad0 = bfgsb_dmodl.clone()
            # Projecting initial guess within the bounded area
            if "bounds" in dir(problem):
                problem.bounds.apply(bfgsb_mdl)
            if prblm_mdl.isDifferent(bfgsb_mdl):
                # Model hit bounds
                msg = "\t!!!Initial guess outside of bounds. Projecting initial guess back into the feasible space.!!!"
                if self.logger:
                    self.logger.addToLog(msg)
                problem.set_model(bfgsb_mdl)
            # Other internal variables
            iiter = 0
            theta = 1.0
        else:
            # Retrieving parameters and vectors to restart the solver
            msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
            if verbose:
                print(msg)
            if self.logger:
                self.logger.addToLog(msg)
            self.restart.read_restart()
            iiter = self.restart.retrieve_parameter("iter")
            k_steps = self.restart.retrieve_parameter("k_steps")
            theta = self.restart.retrieve_vector("theta")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            bfgsb_mdl = self.restart.retrieve_vector("bfgsb_mdl")
            bfgsb_dmodl = self.restart.retrieve_vector("bfgsb_dmodl")
            bfgsb_grad0 = self.restart.retrieve_vector("bfgsb_grad0")
            M = self.restart.retrieve_parameter("M")
            # Setting the model and residuals to avoid residual twice computation
            problem.set_model(bfgsb_mdl)
            prblm_mdl = problem.get_model()
            # Setting residual vector to avoid its unnecessary computation
            problem.set_residual(self.restart.retrieve_vector("prblm_res"))
            # Reading previously obtained S_mat and Y_mat
            k_vecs = min(k_steps, self.m_steps)
            if k_vecs > 0:
                for ii in range(k_vecs):
                    S_mat.append(self.restart.retrieve_vector("s_vec%s" % ii))
                    Y_mat.append(self.restart.retrieve_vector("y_vec%s" % ii))
                    W, _, _ = self.form_W(Y_mat, theta, S_mat)  # [Y theta*S]
        
        # Common variables unrelated to restart
        prev_mdl = prblm_mdl.clone().zero()
        bfgsb_mdl_cauchy = prblm_mdl.clone().zero()  # Generalized Cauchy point vector
        # Precision of numbers from dot-product
        norm_mdl = prblm_mdl.norm()
        if isinstance(norm_mdl, np.float32):
            self.epsmch = np.finfo(np.float32).eps
        else:
            self.epsmch = np.finfo(np.float64).eps
        
        # Inversion loop
        while True:
            # Computing objective function
            obj0 = problem.get_obj(bfgsb_mdl)  # Compute objective function value
            prblm_res = problem.get_res(bfgsb_mdl)  # Compute residuals
            prblm_grad = problem.get_grad(bfgsb_mdl)  # Compute the gradient
            if iiter == 0:
                initial_obj_value = obj0  # For relative objective function value
                # Saving objective function value
                self.restart.save_parameter("obj_initial", initial_obj_value)
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       obj0,
                                       problem.get_rnorm(bfgsb_mdl),
                                       problem.get_gnorm(bfgsb_mdl),
                                       str(problem.get_fevals()).zfill(self.stopper.zfill + 1),
                                       str(problem.get_gevals()).zfill(self.stopper.zfill))
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj0) or isnan(prblm_grad.norm()):
                    raise ValueError("Either gradient norm or objective function value NaN!")
            if prblm_grad.norm() == 0.:
                print("Gradient vanishes identically")
                break
            
            # Saving results
            self.save_results(iiter, problem, force_save=False)
            # Saving current inverted model
            prev_mdl.copy(prblm_mdl)
            # grad0 = grad
            bfgsb_grad0.copy(prblm_grad)
            
            # Compute the new search direction by finding Cauchy point and solving subspace minimization problem
            # [xc, c] = get_cauchy_point(x, g, l, u, theta, W, M);
            c = self.get_cauchy_point(bfgsb_mdl_cauchy, bfgsb_dmodl, bfgsb_mdl, prblm_grad, minBound, maxBound, W, M,
                                      theta)
            self.stepper.alpha = 1.0
            if iiter == 0:
                bfgsb_dmodl.copy(bfgsb_mdl_cauchy)
                bfgsb_dmodl.scaleAdd(bfgsb_mdl, 1.0, -1.0)
                self.stepper.alpha = min(1.0, 1.0 / (bfgsb_dmodl.norm() + self.epsmch))
            else:
                free_var = self.subspace_min(bfgsb_dmodl, bfgsb_mdl, prblm_grad, minBound, maxBound, bfgsb_mdl_cauchy,
                                             c, W, M, theta)
                if not free_var:
                    bfgsb_dmodl.copy(bfgsb_mdl_cauchy)
                    bfgsb_dmodl.scaleAdd(bfgsb_mdl, 1.0, -1.0)
                    self.stepper.alpha = min(1.0, 1.0 / (bfgsb_dmodl.norm() + self.epsmch))
            self.stepper.alpha_max = self.max_step(bfgsb_mdl, bfgsb_dmodl, minBound, maxBound)
            if self.stepper.alpha > self.stepper.alpha_max:
                self.stepper.alpha = self.stepper.alpha_max
            dphi = prblm_grad.dot(bfgsb_dmodl)
            if dphi > 0.0:
                msg = "Current search direction not a descent one (dphi/dalpha|dmod=%.5e)" % dphi
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                break
            else:
                alpha, success = self.stepper.run(problem, bfgsb_mdl, bfgsb_dmodl, self.logger)
            if not success:
                msg = "Stepper couldn't find a proper step size, will terminate solver"
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                break
            
            obj1 = problem.get_obj(bfgsb_mdl)  # Compute objective function value
            # Redundant tests on verifying convergence
            if obj0 <= obj1:
                msg = "Objective function at new point greater or equal than previous one: obj_fun_old=%s obj_fun_new=%s\n" \
                      "Stopping inversion!" % (obj0, obj1)
                if self.logger:
                    self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                break
            
            # Compute new gradient
            prblm_grad = problem.get_grad(bfgsb_mdl)
            
            ###########################################
            # Update the LBFGS-B matrices
            # Gradient difference
            y_tmp = prblm_grad.clone()
            y_tmp.scaleAdd(bfgsb_grad0, 1.0, -1.0)
            # Taken step
            s_tmp = bfgsb_mdl.clone()
            s_tmp.scaleAdd(prev_mdl, 1.0, -1.0)
            
            # Checking curvature condition of equation 3.9 in "op LIMITED MEMORY ALGORITHM FOR BOUND
            # CONSTRAINED OPTIMIZATION" by Byrd et al. (1995)
            if y_tmp.dot(s_tmp) > self.epsmch * y_tmp.dot(y_tmp) and y_tmp.dot(s_tmp) > 0.0:
                msg = "Updating current Hessian estimate"
                if self.logger:
                    self.logger.addToLog(msg)
                if len(Y_mat) < self.m_steps:
                    # Appending new y and s vectors
                    Y_mat.append(y_tmp)
                    S_mat.append(s_tmp)
                else:
                    # Removing first column and updating last one
                    Y_mat[:-1] = Y_mat[1:]
                    Y_mat[-1] = y_tmp
                    S_mat[:-1] = S_mat[1:]
                    S_mat[-1] = s_tmp
                if iiter == 1 and len(Y_mat) == 2:
                    # Removing update from iteration 0
                    Y_mat.pop(0)
                    S_mat.pop(0)
                # Number of steps currently within L-BFGS matrix
                k_steps = len(Y_mat)
                if k_steps > 0:
                    theta = y_tmp.dot(y_tmp) / y_tmp.dot(s_tmp)
                    W, Y, S = self.form_W(Y_mat, theta, S_mat)  # [Y theta*S]
                    A = np.matmul(Y.T, S)
                    StS = np.matmul(S.T, S)
                    L = np.tril(A, -1)
                    D = -np.diag(np.diag(A))
                    M1 = np.concatenate((D, L.T), axis=1)
                    M2 = np.concatenate((L, theta * StS), axis=1)
                    MM = np.concatenate((M1, M2), axis=0)
                    M = np.linalg.inv(MM)
                    del L, D, M1, M2, MM, A
                    # Saving Inverse Hessian estimate for restart
                    self.restart.save_vector("y_vec%s" % (k_steps - 1), y_tmp)
                    self.restart.save_vector("s_vec%s" % (k_steps - 1), s_tmp)
            else:
                msg = "Skipping L-BFGS update; negative curvature detected"
                if self.logger:
                    self.logger.addToLog(msg)
            ###########################################
            # Increasing iteration counter
            iiter = iiter + 1
            
            # Saving current model and previous search direction in case of restart
            self.restart.save_parameter("iter", iiter)
            self.restart.save_parameter("theta", theta)
            self.restart.save_parameter("M", M)
            self.restart.save_parameter("k_steps", k_steps)
            self.restart.save_vector("bfgsb_mdl", bfgsb_mdl)
            self.restart.save_vector("bfgsb_dmodl", bfgsb_dmodl)
            self.restart.save_vector("bfgsb_grad0", bfgsb_grad0)
            # Saving data space vectors
            self.restart.save_vector("prblm_res", prblm_res)
            
            # iteration info
            msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                   obj1,
                                   problem.get_rnorm(bfgsb_mdl),
                                   problem.get_gnorm(bfgsb_mdl),
                                   str(problem.get_fevals()).zfill(self.stopper.zfill + 1),
                                   str(problem.get_gevals()).zfill(self.stopper.zfill))
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
            if self.m_steps is not None:
                msg += 12 * " " + "L-BFGS-B Solver log file end\n"
            else:
                msg += 12 * " " + "BFGS-B Solver log file end\n"
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace("log file ", ""))
            if self.logger:
                self.logger.addToLog(msg)
        self.restart.clear_restart()

