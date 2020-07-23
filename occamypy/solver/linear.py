# Module containing Linear Solver classes
from math import isnan
import numpy as np

from occamypy.solver import Solver
from occamypy import problem as P

zero = 10 ** (np.floor(np.log10(np.abs(float(np.finfo(np.float64).tiny)))) + 2)  # Check for avoid Overflow or Underflow


class CG(Solver):
    """Linear-Conjugate Gradient and Steepest-Descent Solver parent object"""

    # Default class methods/functions
    def __init__(self, stopper, steepest=False, logger=None):
        """
        Constructor for LCG/SD Solver:
        :param stopper: Stopper, object to terminate inversion
        :param steepest: bool, use the steepest-descent instead of conjugate gradient [False]
        :param logger: Logger, object to write inversion log file [None]
        """
        # Calling parent construction
        super(CG, self).__init__()
        # Defining stopper object
        self.stopper = stopper
        # Whether to run steepest descent or not
        self.steepest = steepest
        # Logger object to write on log file
        self.logger = logger
        # Overwriting logger of the Stopper object
        self.stopper.logger = self.logger
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, resnorm = %.2e, gradnorm = %.2e, feval = %d"

    def __del__(self):
        """Default destructor"""
        return

    def run(self, problem, verbose=False, restart=False):
        """Running LCG and steepest-descent solver"""
        self.create_msg = verbose or self.logger

        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Check for preconditioning
        precond = True if "prec" in dir(problem) and problem.prec is not None else False

        if not restart:
            if self.create_msg:
                msg = 90 * "#" + "\n"
                msg += "\t\t\t\tPRECONDITIONED " if precond else "\t\t\t\t"
                msg += "LINEAR %s SOLVER\n" % ("STEEPEST-DESCENT log file" if self.steepest else "CONJUGATE GRADIENT log file")
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)

            # Setting internal vectors (model and search direction vectors)
            prblm_mdl = problem.get_model()
            cg_mdl = prblm_mdl.clone()
            cg_dmodl = prblm_mdl.clone().zero()

            # Other internal variables
            iiter = 0
        else:
            # Retrieving parameters and vectors to restart the solver
            if self.create_msg:
                msg = "Restarting previous solve run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            self.restart.read_restart()
            iiter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            cg_mdl = self.restart.retrieve_vector("cg_mdl")
            cg_dmodl = self.restart.retrieve_vector("cg_dmodl")
            if not precond:
                cg_dres = self.restart.retrieve_vector("cg_dres")
            else:
                dot_grad_prec_grad = self.restart.retrieve_vector("dot_grad_prec_grad")
            # Setting the model and residuals to avoid residual twice computation
            problem.set_model(cg_mdl)
            prblm_mdl = problem.get_model()
            # Setting residual vector to avoid its unnecessary computation
            problem.set_residual(self.restart.retrieve_vector("prblm_res"))

        # Common variables unrelated to restart
        success = True
        # Variables necessary to return inverted model if inversion stops earlier
        prev_mdl = prblm_mdl.clone().zero()
        if precond:
            cg_prec_grad = cg_dmodl.clone().zero()

        # Iteration loop
        while True:
            # Computing objective function
            obj0 = problem.get_obj(cg_mdl)  # Compute objective function value
            prblm_res = problem.get_res(cg_mdl)  # Compute residuals
            prblm_grad = problem.get_grad(cg_mdl)  # Compute the gradient
            if iiter == 0:
                initial_obj_value = obj0  # For relative objective function value
                # Saving initial objective function value
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if self.create_msg:
                    msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.get_rnorm(cg_mdl),
                                           problem.get_gnorm(cg_mdl),
                                           problem.get_fevals())
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj0) or isnan(prblm_grad.norm()):
                    raise ValueError("Either gradient norm or objective function value NaN!")
                # Set internal delta residual vector
                if not precond:
                    cg_dres = prblm_res.clone().zero()
            if prblm_grad.norm() == 0.:
                print("Gradient vanishes identically")
                break

            # Saving results
            self.save_results(iiter, problem, force_save=False)
            prev_mdl.copy(prblm_mdl)  # Keeping the previous model

            # Computing alpha and beta coefficients
            if precond:
                # Applying preconditioning to current gradient
                problem.prec.forward(False, prblm_grad, cg_prec_grad)
                if iiter == 0 or self.steepest:
                    # Steepest descent
                    beta = 0.
                    dot_grad_prec_grad = prblm_grad.dot(cg_prec_grad)
                else:
                    # Conjugate-gradient coefficients for preconditioned CG
                    dot_grad_prec_grad_old = dot_grad_prec_grad
                    if dot_grad_prec_grad_old == 0.:
                        success = False
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog("Gradient orthogonal to preconditioned one, will terminate solver")
                    dot_grad_prec_grad = prblm_grad.dot(cg_prec_grad)
                    beta = dot_grad_prec_grad / dot_grad_prec_grad_old
                # Update search direction
                cg_dmodl.scaleAdd(cg_prec_grad, beta, 1.0)
                cg_dmodld = problem.get_dres(cg_mdl, cg_dmodl)  # Project search direction into the data space
                dot_cg_dmodld = cg_dmodld.dot(cg_dmodld)
                if dot_cg_dmodld == 0.0:
                    success = False
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(
                            "Search direction orthogonal to span of linear operator, will terminate solver")
                else:
                    alpha = - dot_grad_prec_grad / dot_cg_dmodld
                    # Writing on log file
                    if beta == 0.:
                        msg = "Steepest-descent step length: %.2e" % alpha
                    else:
                        msg = "Conjugate alpha, beta: %.2e, %.2e" % (alpha, beta)
                    if self.logger:
                        self.logger.addToLog(msg)
            else:
                prblm_gradd = problem.get_dres(cg_mdl, prblm_grad)  # Project gradient into the data space
                # Computing alpha and beta coefficients
                if iiter == 0 or self.steepest:
                    # Steepest descent
                    beta = 0.0
                    dot_gradd = prblm_gradd.dot(prblm_gradd)
                    if dot_gradd <= zero:
                        success = False
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog(
                                "Gradient orthogonal to span of linear operator, will terminate solver")
                    else:
                        dot_gradd_res = prblm_gradd.dot(prblm_res)
                        alpha = - np.real(dot_gradd_res) / dot_gradd
                        msg = "Steppest-descent step length: " + str(alpha)
                        # Writing on log file
                        if iiter == 0:
                            msg = "First steppest-descent step length: " + str(alpha)
                        if self.logger:
                            self.logger.addToLog(msg)
                else:
                    # Conjugate-gradient coefficients
                    dot_gradd = prblm_gradd.dot(prblm_gradd)
                    dot_dres = cg_dres.dot(cg_dres)
                    dot_gradd_dres = np.real(prblm_gradd.dot(cg_dres))
                    if dot_gradd <= zero or dot_dres <= zero:
                        success = False
                    else:
                        determ = dot_gradd * dot_dres - dot_gradd_dres * dot_gradd_dres
                        # Checking if alpha or beta are becoming infinity
                        if abs(determ) < zero:
                            if self.create_msg:
                                msg = "Plane-search method fails (zero det: %.2e), will terminate solver" % determ
                                if verbose:
                                    print(msg)
                                if self.logger:
                                    self.logger.addToLog(msg)
                            break
                        dot_gradd_res = np.real(prblm_gradd.dot(prblm_res))
                        dot_dres_res = np.real(cg_dres.dot(prblm_res))
                        alpha = -(dot_dres * dot_gradd_res - dot_gradd_dres * dot_dres_res) / determ
                        beta = (dot_gradd_dres * dot_gradd_res - dot_gradd * dot_dres_res) / determ
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog("Conjugate alpha,beta: " + str(alpha) + ", " + str(beta))

            if not success:
                if self.create_msg:
                    msg = "Stepper couldn't find a proper step size, will terminate solver"
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                break

            if precond:
                # modl = modl + alpha * dmodl
                cg_mdl.scaleAdd(cg_dmodl, 1.0, alpha)  # Update model
            else:
                # dmodl = alpha * grad + beta * dmodl
                cg_dmodl.scaleAdd(prblm_grad, beta, alpha)  # update search direction
                # modl = modl + dmodl
                cg_mdl.scaleAdd(cg_dmodl)  # Update model

            # Increasing iteration counter
            iiter += 1
            # Setting the model
            problem.set_model(cg_mdl)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(cg_mdl)

            if prblm_mdl.isDifferent(cg_mdl):
                # Model went out of the bounds
                msg = "Model hit provided bounds. Projecting it onto them."
                if self.logger:
                    self.logger.addToLog(msg)
                # Recomputing m_current = m_new - dmodl
                prblm_mdl.scaleAdd(cg_dmodl, 1.0, -1.0)
                # Finding the projected dmodl = m_new_clipped - m_current
                cg_dmodl.copy(cg_mdl)
                cg_dmodl.scaleAdd(prblm_mdl, 1.0, -1.0)
                problem.set_model(cg_mdl)
                if precond:
                    cg_dmodl.scale(1.0 / alpha)  # Unscaling the search direction
                else:
                    # copying previous residuals dres = res_old
                    cg_dres.copy(prblm_res)
                    # Computing actual change in the residual vector dres = res_new - res_old
                    prblm_res = problem.get_res(cg_mdl)  # New residual vector
                    cg_dres.scaleAdd(prblm_res, -1.0, 1.0)
            else:
                # Setting residual vector to avoid its unnecessary computation (if model was not clipped)
                if precond:
                    # res = res + alpha * dres
                    prblm_res.scaleAdd(cg_dmodld, 1.0, alpha)  # Update residuals
                else:
                    # dres  = alpha * gradd + beta * dres
                    cg_dres.scaleAdd(prblm_gradd, beta, alpha)  # Update residual step
                    # res = res + dres
                    prblm_res.scaleAdd(cg_dres)  # Update residuals
                problem.set_residual(prblm_res)

            # Computing new objective function value
            obj1 = problem.get_obj(cg_mdl)
            if obj1 >= obj0:
                if self.create_msg:
                    msg = "Objective function didn't reduce, will terminate solver:\n\t" \
                          "obj_new = %.5e\tobj_cur = %.5e" % (obj1, obj0)
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                problem.set_model(prev_mdl)
                break

            # Saving current model and previous search direction in case of restart
            self.restart.save_parameter("iter", iiter)
            self.restart.save_vector("cg_mdl", cg_mdl)
            self.restart.save_vector("cg_dmodl", cg_dmodl)
            # Saving data space vectors or scaling if preconditioned
            if not precond:
                self.restart.save_vector("cg_dres", cg_dres)
            else:
                self.restart.save_parameter("dot_grad_prec_grad", dot_grad_prec_grad)
            self.restart.save_vector("prblm_res", prblm_res)

            # iteration info
            if self.create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.get_rnorm(cg_mdl),
                                       problem.get_gnorm(cg_mdl),
                                       problem.get_fevals())
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            # Check if either objective function value or gradient norm is NaN
            if isnan(obj1) or isnan(prblm_grad.norm()):
                raise ValueError("Either gradient norm or objective function value NaN!")
            if self.stopper.run(problem, iiter, initial_obj_value, verbose):
                break

        # Writing last inverted model
        self.save_results(iiter, problem, force_save=True, force_write=True)
        if self.create_msg:
            msg = 90 * "#" + "\n"
            msg += "\t\t\t\tPRECONDITIONED " if precond else "\t\t\t\t"
            msg += "LINEAR %s SOLVER log file end\n" % ("STEEPEST-DESCENT" if self.steepest else "CONJUGATE GRADIENT")
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog(msg)
        # Clear restart object
        self.restart.clear_restart()

        return


class SD(CG):
    def __init__(self, stopper, logger=None):
        super(SD, self).__init__(stopper, steepest=True, logger=logger)
    
    
def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.
    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).
    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0.:
        return np.sign(a), 0, np.abs(a)
    elif a == 0.:
        return 0., np.sign(b), np.abs(b)
    elif np.abs(b) > np.abs(a):
        tau = a / b
        s = np.sign(b) / np.sqrt(1. + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / np.sqrt(1. + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


class LSQR(Solver):
    """
       LSQR Solver parent object following algorithm in Paige and Saunders (1982)
       Find the least-squares solution to a large, sparse, linear system
       of equations.
       The function solves Ax = b or min ||b - Ax||^2
       If A is symmetric, LSQR should not be used! Use SymLCGsolver
       If initial model different than zero the solver will perform the following:
         1. Compute initial residual vector ``r0 = b - A*x0``.
         2. Use LSQR to solve the system  ``A*dx = r0``.
         3. Add the correction dx to obtain a final solution ``x = x0 + dx``.
    """

    def __init__(self, stopper, estimate_cond=False, estimate_var=False, logger=None):
        """
        Constructor for LSQR Solver:
        :param stopper          : Stopper, object to terminate inversion
        :param estimate_cond    : Boolean, whether the condition number of A is estimated
                                  access self.acond after solver run [False]
        :param estimate_var     : Boolean, whether the diagonal of A'A^-1 is estimated or not;
                                  access self.var after solver run [False]
        :param logger           : Logger, object to write inversion log file [None]
        """
        # Calling parent construction
        super(LSQR, self).__init__()
        # Defining stopper object
        self.stopper = stopper
        # Logger object to write on log file
        self.logger = logger
        # Create var variable if estimate var is requested
        self.est_cond = True if estimate_cond or estimate_var else False
        self.var = estimate_var
        # Overwriting logger of the Stopper object
        self.stopper.logger = self.logger
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, resnorm = %.2e, gradnorm = %.2e, feval = %d"

    def __del__(self):
        """Default destructor"""
        return

    def run(self, problem, verbose=False, restart=False):
        """Running LSQR solver"""
        self.create_msg = verbose or self.logger

        # Resetting stopper before running the inversion
        self.stopper.reset()

        # Setting internal vectors and initial variables
        prblm_mdl = problem.get_model()
        initial_mdl = prblm_mdl.clone()

        if not restart:
            if self.create_msg:
                msg = 90 * "#" + "\n"
                msg += "\t\t\t\tLSQR SOLVER log file\n"
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                msg += 90 * "#" + "\n"
                if verbose:
                    print(msg.replace("log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)

            # If initial model different than zero the solver will perform the following:
            # 1. Compute a residual vector ``r0 = b - A*x0``.
            # 2. Use LSQR to solve the system  ``A*dx = r0``.
            # 3. Add the correction dx to obtain a final solution ``x = x0 + dx``.
            prblm_res = problem.get_res(initial_mdl)  # Initial data residuals
            obj0 = initial_obj_value = problem.get_obj(initial_mdl)  # For relative objective function value
            # Saving initial objective function value
            u = prblm_res.clone().scale(-1.0)
            x = prblm_mdl.clone().zero()  # Solution vector
            w = x.clone()
            v = x.clone()
            if self.est_cond:
                dk = x.clone()
                ddnorm = 0.
            # Estimating variance or diagonal elements of the inverse
            if self.var:
                self.var = x.clone()

            # Other internal variables
            iiter = 0

            # Initial inversion parameters
            alpha = 0.
            beta = u.norm()

            if beta > 0.:
                u.scale(1. / beta)
                # A.H * u => gradient with scaled residual vector
                problem.set_model(x)  # x = 0
                problem.set_residual(u)  # res = u
                prblm_grad = problem.get_grad(x)  # g = A.H * u
                v.copy(prblm_grad)  # v = g
                alpha = v.norm()
            else:
                prblm_grad = problem.get_grad(initial_mdl)
            if alpha > 0.:
                v.scale(1. / alpha)
                w.copy(v)

            rhobar = alpha
            phibar = beta
            anorm = 0.

            # First inversion logging
            self.restart.save_parameter("obj_initial", initial_obj_value)
            # Estimating residual and gradient norms
            prblm_res.scale(phibar)
            prblm_grad.scale(alpha*beta)
            if self.create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       initial_obj_value,
                                       problem.get_rnorm(prblm_mdl),
                                       problem.get_gnorm(prblm_mdl),
                                       problem.get_fevals())
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog(msg)
            # Check if either objective function value or gradient norm is NaN
            if isnan(initial_obj_value) or isnan(problem.get_gnorm(x)):
                raise ValueError("Either gradient norm or objective function value NaN!")

        else:
            # Retrieving parameters and vectors to restart the solver
            if self.create_msg:
                msg = "Restarting previous solve run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            self.restart.read_restart()

            # Retrieving iteration number
            iiter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            obj0 = self.restart.retrieve_parameter("obj0")
            # Retrieving state vectors
            u = self.restart.retrieve_vector("u")
            x = self.restart.retrieve_vector("x")
            w = self.restart.retrieve_vector("w")
            v = self.restart.retrieve_vector("v")
            problem.set_model(x)
            problem.set_residual(u)

            prblm_res = problem.get_res(x)
            if self.est_cond:
                dk = self.restart.retrieve_vector("dk")
                ddnorm = self.restart.retrieve_parameter("ddnorm")
            if self.var:
                self.var = self.restart.retrieve_vector("var")
            # Retrieving inversion parameters
            alpha = self.restart.retrieve_parameter("alpha")
            rhobar = self.restart.retrieve_parameter("rhobar")
            phibar = self.restart.retrieve_parameter("phibar")
            anorm = self.restart.retrieve_parameter("anorm")

        # Common variables unrelated to restart
        prblm_mdl = problem.get_model()
        inv_model = prblm_mdl.clone()  # Inverted model to be saved during the inversion
        # Variables necessary to return inverted model if inversion stops earlier
        prev_x = x.clone().zero()
        early_stop =False

        # Iteration loop
        while True:
            if problem.get_gnorm(prblm_mdl) == 0.:
                print("Gradient vanishes identically")
                break

            # Saving results
            inv_model.copy(initial_mdl)
            inv_model.scaleAdd(x)  # x = x0 + dx; Updating inverted model
            self.save_results(iiter, problem, model=inv_model, force_save=False)
            # Necessary to save previous iteration
            prev_x.copy(x)

            """
                %     Perform the next step of the bidiagonalization to obtain the
                %     next  beta, u, alpha, v.  These satisfy the relations
                %                beta*u  =  A*v   -  alpha*u,
                %                alpha*v  =  A'*u  -  beta*v.
            """


            # A.matvec(v) (i.e., projection of v onto the data space)
            v_prblm = problem.get_dres(x, v)
            # u = A.matvec(v) - alpha * u
            u.scaleAdd(v_prblm, -alpha, 1.0)
            beta = u.norm()

            if beta > 0.:
                u.scale(1. / beta)
                anorm = np.sqrt(anorm ** 2 + alpha ** 2 + beta ** 2)
                problem.set_model(x)
                problem.set_residual(u)  # res = u
                prblm_grad = problem.get_grad(x)  # g = A.H * u
                # v = A.rmatvec(u) - beta * v
                v.scaleAdd(prblm_grad, -beta, 1.)
                alpha = v.norm()
                if alpha > 0.:
                    v.scale(1. / alpha)
            else:
                problem.set_model(x)
                problem.set_residual(u)  # res = u

            # Use a plane rotation to eliminate the subdiagonal element (beta)
            # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
            cs, sn, rho = _sym_ortho(rhobar, beta)

            theta = sn * alpha
            rhobar = -cs * alpha
            phi = cs * phibar
            phibar *= sn

            # Estimating residual and gradient norms
            prblm_res.scale(phibar)
            if prblm_grad.norm() > zero:
                prblm_grad.scale(alpha * abs(sn * phi) / prblm_grad.norm())
            else:
                prblm_grad.zero()
            # New objective function value
            obj1 = problem.get_obj(prblm_mdl)


            # Update x and w.
            # x = x + t1 * w
            x.scaleAdd(w, 1.0, phi / rho)
            # w = v + t2 * w
            w.scaleAdd(v, -theta / rho, 1.0)

            # Increasing iteration counter
            iiter += 1

            # Checking new objective function value
            if obj1 >= obj0:
                if self.create_msg:
                    msg = "Objective function didn't reduce, will terminate solver:\n\t" \
                          "obj_new = %.5e\tobj_cur = %.5e" % (obj1, obj0)
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                early_stop = True
                break

            if self.est_cond:
                dk.copy(w).scale(1. / rho)
                ddnorm += dk.norm() ** 2
                # Estimate the condition of the matrix  Abar,
                self.acond = anorm * np.sqrt(ddnorm)
                self.restart.save_vector("dk", dk)
                self.restart.save_parameter("ddnorm", ddnorm)
            if self.var:
                # var = var + dk ** 2
                self.var.scaleAdd(dk.clone().multiply(dk))
                self.restart.save_vector("var", self.var)

            # Saving previous objective function value
            obj0 = obj1

            # Saving state variables and vectors for restart
            self.restart.save_parameter("iter", iiter)
            self.restart.save_parameter("obj0", obj0)
            self.restart.save_parameter("alpha", alpha)
            self.restart.save_parameter("rhobar", rhobar)
            self.restart.save_parameter("phibar", phibar)
            self.restart.save_parameter("anorm", anorm)
            self.restart.save_vector("u", u)
            self.restart.save_vector("x", x)
            self.restart.save_vector("w", w)
            self.restart.save_vector("v", v)

            # iteration info
            if self.create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       problem.get_obj(prblm_mdl),
                                       problem.get_rnorm(prblm_mdl),
                                       problem.get_gnorm(prblm_mdl),
                                       problem.get_fevals())
                if self.est_cond:
                    msg += ", condition_num =  %.2e, matrix_norm = %.2e" % (self.acond, anorm)
                if verbose:
                    print(msg)
                # Writing on log file
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            # Check if either objective function value or gradient norm is NaN
            if isnan(problem.get_obj(prblm_mdl)) or isnan(problem.get_gnorm(prblm_mdl)):
                raise ValueError("Either gradient norm or objective function value NaN!")
            if self.stopper.run(problem, iiter, initial_obj_value, verbose):
                break

        # Writing last inverted model
        inv_model.copy(initial_mdl)
        if early_stop:
            x.copy(prev_x)
        inv_model.scaleAdd(x)  # x = x0 + dx; Updating inverted model
        self.save_results(iiter, problem, model=inv_model, force_save=True, force_write=True)
        prblm_mdl.copy(inv_model) # Setting inverted model to final one
        if self.create_msg:
            msg = 90 * "#" + "\n"
            msg += "\t\t\t\tLSQR SOLVER log file end\n"
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace("log file ", ""))
            if self.logger:
                self.logger.addToLog(msg)
        # Clear restart object
        self.restart.clear_restart()
        return


class CGsym(Solver):
    """Linear-Conjugate Gradient Solver (for symmetric systems) parent object"""

    # Default class methods/functions
    def __init__(self, stopper, steepest=False, logger=None):
        """Constructor for LCG Solver for symmetric systems"""
        # Calling parent construction
        super(CGsym, self).__init__()
        # Defining stopper object
        self.stopper = stopper
        # Whether to run steepest descent or not
        self.steepest = steepest
        # Logger object to write on log file
        self.logger = logger
        # Overwriting logger of the Stopper object
        self.stopper.logger = self.logger
        # Setting defaults for saving results
        self.setDefaults()
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, resnorm = %.2e, feval = %d"
        return

    def __del__(self):
        """Default destructor"""
        return

    def run(self, problem, verbose=False, restart=False):
        """Running LCG solver for symmetric systems"""
        self.create_msg = verbose or self.logger

        # Resetting stopper before running the inversion
        self.stopper.reset()
        # Checking if we are solving a linear square problem
        if not isinstance(problem, P.LeastSquaresSymmetric):
            raise TypeError("ERROR! Provided problem object not a linear symmetric problem")
        # Check for preconditioning
        precond = False
        if "prec" in dir(problem):
            if problem.prec is not None:
                precond = True
        if not restart:
            if self.create_msg:
                msg = 90 * "#" + "\n"
                msg += "PRECONDITIONED " if precond else ""
                msg += "LINEAR %s SOLVER FOR SYMMETRIC MATRIX log file" % (
                    "STEEPEST-DESCENT" if self.steepest else "CONJUGATE GRADIENT")
                msg += "Restart folder: %s\n" % self.restart.restart_folder
                msg = 90 * "#" + "\n"
                if verbose:
                    print(msg.replace("log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)

            # Setting internal vectors (model and search direction vectors)
            prblm_mdl = problem.get_model()
            cg_mdl = prblm_mdl.clone()
            cg_dmodl = prblm_mdl.clone().zero()
            if precond:
                cg_prec_res = cg_dmodl.clone()

            # Other internal variables
            iiter = 0
            beta = 0.
        else:
            # Retrieving parameters and vectors to restart the solver
            if self.create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
            self.restart.read_restart()
            iiter = self.restart.retrieve_parameter("iter")
            beta = self.restart.retrieve_parameter("beta")
            obj_old = self.restart.retrieve_parameter("obj_old")
            cg_mdl = self.restart.retrieve_vector("cg_mdl")
            cg_dmodl = self.restart.retrieve_vector("cg_dmodl")
            if precond: cg_prec_res = self.restart.retrieve_vector("cg_prec_res")
            # Setting the model and residuals to avoid residual double computation
            problem.set_model(cg_mdl)
            prblm_mdl = problem.get_model()
            # Setting residual vector to avoid its unnecessary computation
            problem.set_residual(self.restart.retrieve_vector("prblm_res"))

        # Common variables unrelated to restart
        success = True
        data_norm = problem.data.norm()
        prev_mdl = prblm_mdl.clone().zero()

        # Iteration loop
        while True:
            # Computing objective function
            prblm_res = problem.get_res(cg_mdl)  # Compute residuals
            obj0 = problem.get_obj(cg_mdl)  # Compute objective function value
            if iiter == 0:
                if self.create_msg:
                    msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.get_rnorm(cg_mdl),
                                           problem.get_fevals())
                    if verbose:
                        print(msg)
                    msg += "\nrelative data matching (i.e., 1-|Am-b|/|b|): %.1f" \
                           % ((1.0 - problem.get_rnorm(cg_mdl) / data_norm) * 100.0) + "%"
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                # Check if either objective function value or gradient norm is NaN
                if isnan(obj0):
                    raise ValueError("Error! Objective function value NaN!")

            # Saving results
            self.save_results(iiter, problem, force_save=False)
            # Copying current model in case of early stop
            prev_mdl.copy(cg_mdl)

            # Applying preconditioning to gradient (first time)
            if iiter == 0 and precond:
                problem.prec.forward(False, prblm_res, cg_prec_res)
            # dmodl = beta * dmodl - res
            cg_dmodl.scaleAdd(cg_prec_res if precond else prblm_res, beta, -1.0)  # Update search direction
            prblm_ddmodl = problem.get_dres(cg_mdl, cg_dmodl)  # Project search direction in the data space

            dot_dmodl_ddmodl = cg_dmodl.dot(prblm_ddmodl)
            if precond:
                dot_res = prblm_res.dot(cg_prec_res)  # Dot product of residual and preconditioned one
            else:
                dot_res = prblm_res.dot(prblm_res)
            if dot_res == 0.:
                if self.create_msg:
                    msg = "Residual/Gradient vector vanishes identically"
                    if verbose:
                        print(msg)
                    # Writing on log file
                    if self.logger:
                        self.logger.addToLog(msg)
                break
            elif dot_dmodl_ddmodl == 0.:
                success = False
                msg = "Residual/Gradient vector orthogonal to span of linear operator, will terminate solver"
                # Writing on log file
                if self.logger:
                    self.logger.addToLog(msg)

            if not success:
                if self.create_msg:
                    msg = "Stepper couldn't find a proper step size, will terminate solver"
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog(msg)
                break

            alpha = dot_res / dot_dmodl_ddmodl
            if self.logger:
                self.logger.addToLog("Alpha step length: %.2e" % alpha)

            # modl = modl + alpha * dmodl
            cg_mdl.scaleAdd(cg_dmodl, sc2=alpha)  # Update model

            # Increasing iteration counter
            iiter = iiter + 1
            # Setting the model and residuals to avoid residual twice computation
            problem.set_model(cg_mdl)

            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(cg_mdl)

            if prblm_mdl.isDifferent(cg_mdl):
                # Model went out of the bounds
                msg = "Model hit provided bounds. Projecting it onto them."
                if self.logger:
                    self.logger.addToLog(msg)
                # Recomputing m_current = m_new - alpha * dmodl
                prblm_mdl.scaleAdd(cg_dmodl, 1.0, -alpha)
                # Finding the projected dmodl = m_new_clipped - m_current
                cg_dmodl.copy(cg_mdl)
                cg_dmodl.scaleAdd(prblm_mdl, 1.0, -1.0)
                # dmodl is scaled by the inverse of the step length
                cg_dmodl.scale(1.0 / alpha)
                # copying previos residuals dres = res_old
                prblm_ddmodl.copy(prblm_res)
                problem.set_model(cg_mdl)
                # Computing actual change in the residual vector dres = res_new - res_old
                prblm_res = problem.get_res(cg_mdl)  # New residual vector
                prblm_ddmodl.scaleAdd(prblm_res, -1.0, 1.0)
                # dres is scaled by the inverse of the step length
                prblm_ddmodl.scale(1.0 / alpha)
            else:
                # Setting residual vector to avoid its unnecessary computation (if model was not clipped)
                # res  = res + alpha * dres =  res + alpha * A * dmodl
                prblm_res.scaleAdd(prblm_ddmodl, sc2=alpha)  # update residuals
                problem.set_residual(prblm_res)
                if iiter == 1:
                    problem.fevals += 1  # To correct objective function evaluation number since residuals are set

            # Computing new objective function value
            obj1 = problem.get_obj(cg_mdl)
            if precond:
                # Applying preconditioning to gradient
                problem.prec.forward(False, prblm_res, cg_prec_res)
                dot_res_new = prblm_res.dot(cg_prec_res)
            else:
                # New residual norm
                dot_res_new = prblm_res.dot(prblm_res)
            if not self.steepest:
                beta = dot_res_new / dot_res
            # Checking monotonic behavior of objective function
            if iiter == 1:
                obj_old = obj0  # Saving objective function at iter-1
            else:
                # If not monotonically changing stop the inversion
                if not ((obj_old < obj0 < obj1) or (obj_old > obj0 > obj1)):
                    if self.create_msg:
                        msg = "Objective function variation not monotonic, will terminate solver:" \
                              "obj_old=%.5e obj_cur=%.5e obj_new=%.5e" % (obj_old, obj0, obj1)
                        if verbose:
                            print(msg)
                        # Writing on log file
                        if self.logger:
                            self.logger.addToLog(msg)
                    problem.set_model(prev_mdl)
                    break
                obj_old = obj0  # Saving objective function at iter-1

            # Saving current model and previous search direction in case of restart
            self.restart.save_parameter("iter", iiter)
            self.restart.save_parameter("beta", beta)
            self.restart.save_parameter("obj_old", obj_old)
            self.restart.save_vector("cg_mdl", cg_mdl)
            self.restart.save_vector("cg_dmodl", cg_dmodl)
            if precond:
                self.restart.save_vector("cg_prec_res", cg_prec_res)
            # Saving data space vectors
            self.restart.save_vector("prblm_res", prblm_res)

            # iteration info
            if self.create_msg:
                msg = self.iter_msg % (str(iiter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.get_rnorm(cg_mdl),
                                       problem.get_fevals())
                if verbose:
                    print(msg)
                msg += "\nrelative data matching (i.e., 1-|Am-b|/|b|): %.1f" \
                       % ((1.0 - problem.get_rnorm(cg_mdl) / data_norm) * 100.0) + "%"
                # Writing on log file
                if self.logger:
                    self.logger.addToLog(msg)
            # Check if either objective function value or gradient norm is NaN
            if isnan(obj1):
                raise ValueError("Error! Objective function value NaN!")
            if self.stopper.run(problem, iiter, verbose=verbose):
                break

        # Writing last inverted model
        self.save_results(iiter, problem, force_save=True, force_write=True)
        if self.create_msg:
            msg = 90 * "#" + "\n"
            msg += "PRECONDITIONED " if precond else ""
            msg += "LINEAR %s SOLVER FOR SYMMETRIC MATRIX log file end\n" \
                   % ("STEEPEST-DESCENT" if self.steepest else "CONJUGATE GRADIENT")
            msg += 90 * "#" + "\n"
            if verbose:
                print(msg.replace("log file ", ""))
            if self.logger:
                self.logger.addToLog(msg)
        # Clear restart object
        self.restart.clear_restart()
