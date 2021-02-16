import numpy as np
from math import isnan
from copy import deepcopy


class Stepper:
    """Stepper parent object"""

    # Default class methods/functions
    def __init__(self):
        """Default class constructor for Stepper"""
        return

    def __del__(self):
        """Default destructor"""
        return

    def run(self, model, search_dir):
        """Dummy stepper running method"""
        raise NotImplementedError("Implement run stepper in the derived class.")

    def estimate_initial_guess(self, problem, modl, dmodl, logger):
        """Function to estimate initial step length value"""
        try:
            # Projecting search direction in the data space
            dres = problem.get_dres(modl, dmodl)
        except NotImplementedError:
            if logger:
                logger.addToLog(
                    "\t!!!dresf not implemented; stepper will use inverse of search direction norm as initial step length value!!!")
            # Initial step length value of 1.0 / |dm|_2
            alpha_guess = 1.0 / dmodl.norm()
            return alpha_guess
        res = problem.get_res(modl)
        dres_res = res.dot(dres)
        dres_dres = dres.dot(dres)
        if dres_dres == 0.:
            if logger:
                logger.addToLog(
                    "\t!!!Gradient in the null space of linear forward operator; using inverse of search direction norm as step length value!!!")
            # Initial step length value of 1.0 / |dm|_2
            alpha_guess = 1.0 / dmodl.norm()
        else:
            # alpha = -phi'(0)/phi''(0)
            alpha_guess = -dres_res / dres_dres
        return alpha_guess


class CvSrchStep(Stepper):
    """
    Originally published by More and Thuente (1994) "Line Search Algorithms with Guaranteed Sufficient Decrease"
    CvSrch stepper (from Dianne O'Leary's code):

    THE PURPOSE OF CVSRCH IS TO FIND A STEP WHICH SATISFIES
    A SUFFICIENT DECREASE CONDITION AND A CURVATURE CONDITION.

    AT EACH STAGE THE SUBROUTINE UPDATES AN INTERVAL OF
    UNCERTAINTY WITH ENDPOINTS STX AND STY. THE INTERVAL OF
    UNCERTAINTY IS INITIALLY CHOSEN SO THAT IT CONTAINS A
    MINIMIZER OF THE MODIFIED FUNCTION

        F(X+STP*S) - F(X) - FTOL*STP*(GRADF(X)'S).

    IF A STEP IS OBTAINED FOR WHICH THE MODIFIED FUNCTION
    HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE DERIVATIVE,
    THEN THE INTERVAL OF UNCERTAINTY IS CHOSEN SO THAT IT
    CONTAINS A MINIMIZER OF F(X+STP*S).

    THE ALGORITHM IS DESIGNED TO FIND A STEP WHICH SATISFIES
    THE SUFFICIENT DECREASE CONDITION

         F(X+STP*S) .LE. F(X) + FTOL*STP*(GRADF(X)'S),

    AND THE CURVATURE CONDITION

         ABS(GRADF(X+STP*S)'S)) .LE. GTOL*ABS(GRADF(X)'S).

    IF FTOL IS LESS THAN GTOL AND IF, FOR EXAMPLE, THE FUNCTION
    IS BOUNDED BELOW, THEN THERE IS ALWAYS A STEP WHICH SATISFIES
    BOTH CONDITIONS. IF NO STEP CAN BE FOUND WHICH SATISFIES BOTH
    CONDITIONS, THEN THE ALGORITHM USUALLY STOPS WHEN ROUNDING
    ERRORS PREVENT FURTHER PROGRESS. IN THIS CASE STP ONLY
    SATISFIES THE SUFFICIENT DECREASE CONDITION.
    """

    def __init__(self, alpha=0.0, xtol=1.0e-16, ftol=1.0e-4, gtol=0.95, alpha_min=1.0e-20, alpha_max=1.e20, maxfev=20,
                 xtrapf=4., delta=0.66):
        """
           CvSrch constructor:
           alpha 		 = [0.] - float; Initial step-length guess
           xtol  	 	 = [1e-16] - float; Relative width tolerance: convergence is reached if width falls below xtol * maximum step size.
           ftol  	 	 = [1e-16] - float; c1 value to tests first Wolfe condition (should be between 0 and 1)
           gtol  	 	 = [0.95] - float; c2 value to tests second Wolfe condition (should be between c1 or ftol and 1). For Quasi-Newton (e.g., L-BFGS) choose default. Otherwise, for other methods (e.g., NLCG) choose 0.1
           alpha_min  	 = [1e-20] - float; Minimum step length value of the step length interval
           alpha_max  	 = [1e20] - float; Maximum step length value of the step length interval
           maxfev  	     = [20] - int; Maximum number of function evaluation to step length
           xtrapf  	     = [4.0] - float; Scaling factor to find right limit of uncertainty interval
           delta  	     = [0.66] - float; Value to force sufficient decrease of interval size on successive iterations. Should be a positive value less than 1.
        """
        self.alpha = alpha  # Initial step length guess
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.maxfev = maxfev
        self.xtrapf = xtrapf
        self.delta = delta
        self.zero = 10 ** (np.floor(
            np.log10(np.abs(float(np.finfo(np.float64).tiny)))) + 2)  # Check for avoid Overflow or Underflow
        # Checking stepper parameters
        if xtol < 0.:
            raise ValueError("ERROR! xtol must be greater than 0.0, current value %.2e" % xtol)
        if ftol < 0. or ftol >= 1.:
            raise ValueError("ERROR! ftol must be greater than 0.0 and smaller than 1.0, current value %.2e" % ftol)
        if gtol < 0. or ftol >= 1.:
            raise ValueError("ERROR! gtol must be greater than 0.0 and smaller than 1.0, current value %.2e" % gtol)
        if alpha_min < 0.:
            raise ValueError("ERROR! alpha_min must be greater than 0.0, current value %.2e" % alpha_min)
        if xtrapf < 0.:
            raise ValueError("ERROR! xtrapf must be greater than 0.0, current value %.2e" % xtrapf)
        if delta < 0. or delta >= 1.:
            raise ValueError("ERROR! delta must be greater than 0.0 and smaller than 1.0, current value %.2e" % delta)
        if maxfev < 0:
            raise ValueError("ERROR! maxfev must be greater than 0, current value %d" % maxfev)
        if alpha_max < alpha_min:
            raise ValueError(
                "ERROR! alpha_max must be greater than alpha_min, current values: alpha_min=%.2e; alpha_max=%.2e"
                % (alpha_min, alpha_max))

    def cstep(self, stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax, logger):
        """
            Modified Cstep function (from the code by Dianne O'Leary July 1991):
            The purpose of cstep is to compute a safeguarded step for
            a linesearch and to update an interval of uncertainty for
            a minimizer of the function.

            The parameter stx contains the step with the least function
            value. The parameter stp contains the current step. It is
            assumed that the derivative at stx is negative in the
            direction of the step. If brackt is set true then a
            minimizer has been bracketed in an interval of uncertainty
            with endpoints stx and sty.
            The subroutine statement is

            subroutine cstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,stpmin,stpmax,info)

            where

            stx, fx, and dx are variables which specify the step,
            the function, and the derivative at the best step obtained
            so far. The derivative must be negative in the direction
            of the step, that is, dx and stp-stx must have opposite
            signs. On output these parameters are updated appropriately.

            sty, fy, and dy are variables which specify the step,
            the function, and the derivative at the other endpoint of
            the interval of uncertainty. On output these parameters are
            updated appropriately.

            stp, fp, and dp are variables which specify the step,
            the function, and the derivative at the current step.
            If brackt is set true then on input stp must be
            between stx and sty. On output stp is set to the new step.

            brackt is a logical variable which specifies if a minimizer
            has been bracketed. If the minimizer has not been bracketed
            then on input brackt must be set false. If the minimizer
            is bracketed then on output brackt is set true.

            stpmin and stpmax are input variables which specify lower
            and upper bounds for the step.

            info is an integer output variable set as follows:
            If info = True, then the step has been computed
            according to one of the five cases below. Otherwise
            info = False, and this indicates improper input parameters.
        """

        success = False  # which is info

        # Check the input parameters for errors.
        if ((brackt and (stp <= np.minimum(stx, sty) or
                         stp >= np.maximum(stx, sty))) or
                dx * (stp - stx) >= 0.0 or
                stpmax < stpmin):
            if logger:
                logger.addToLog("\tFunction cstep could find step and update interval of uncertainty!")
            return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, success

        # Determine if the derivatives have opposite sign.
        sgnd = dp * (dx / np.abs(dx))

        # First case. A higher function value.
        # The minimum is bracketed. If the cubic step is closer
        # to stx than the quadratic step, the cubic step is taken,
        # else the average of the cubic and quadratic steps is taken.

        if fp > fx:
            success = True
            bound = True
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
            s = np.linalg.norm([theta, dx, dp], np.inf)
            gamma = s * np.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if stp < stx:
                gamma = -gamma
            p = (gamma - dx) + theta
            q = ((gamma - dx) + gamma) + dp
            r = p / q
            stpc = stx + r * (stp - stx)
            stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
            if np.abs(stpc - stx) < np.abs(stpq - stx):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2.0
            brackt = True

        # Second case. A lower function value and derivatives of
        # opposite sign. The minimum is bracketed. If the cubic
        # step is closer to stx than the quadratic (secant) step,
        # the cubic step is taken, else the quadratic step is taken.

        elif sgnd < 0.0:
            success = True
            bound = False
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
            s = np.linalg.norm([theta, dx, dp], np.inf)
            gamma = s * np.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s))
            if stp > stx:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dx
            r = p / q
            stpc = stp + r * (stx - stp)
            stpq = stp + (dp / (dp - dx)) * (stx - stp)
            if np.abs(stpc - stp) > np.abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            brackt = True

        # Third case. A lower function value, derivatives of the
        # same sign, and the magnitude of the derivative decreases.
        # The cubic step is only used if the cubic tends to infinity
        # in the direction of the step or if the minimum of the cubic
        # is beyond stp. Otherwise the cubic step is defined to be
        # either stpmin or stpmax. The quadratic (secant) step is also
        # computed and if the minimum is bracketed then the the step
        # closest to stx is taken, else the step farthest away is taken.

        elif np.abs(dp) < np.abs(dx):
            success = True
            bound = True
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
            s = np.linalg.norm([theta, dx, dp], np.inf)

            # The case gamma = 0 only arises if the cubic does not tend
            # to infinity in the direction of the step.

            gamma = s * np.sqrt(np.maximum(0., (theta / s) * (theta / s) - (dx / s) * (dp / s)))
            if stp > stx:
                gamma = -gamma

            p = (gamma - dp) + theta
            q = (gamma + (dx - dp)) + gamma
            r = p / q
            if r < 0.0 and gamma != 0.0:
                stpc = stp + r * (stx - stp)
            elif stp > stx:
                stpc = stpmax
            else:
                stpc = stpmin
            stpq = stp + (dp / (dp - dx)) * (stx - stp)
            if brackt:
                if np.abs(stp - stpc) < np.abs(stp - stpq):
                    stpf = stpc
                else:
                    stpf = stpq
            else:
                if np.abs(stp - stpc) > np.abs(stp - stpq):
                    stpf = stpc
                else:
                    stpf = stpq

        # Fourth case. A lower function value, derivatives of the
        # same sign, and the magnitude of the derivative does
        # not decrease. If the minimum is not bracketed, the step
        # is either stpmin or stpmax, else the cubic step is taken.

        else:
            success = True
            bound = False
            if brackt:
                theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
                s = np.linalg.norm([theta, dx, dp], np.inf)
                gamma = s * np.sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s))
                if stp > sty:
                    gamma = -gamma
                p = (gamma - dp) + theta
                q = ((gamma - dp) + gamma) + dy
                r = p / q
                stpc = stp + r * (sty - stp)
                stpf = stpc
            elif stp > stx:
                stpf = stpmax
            else:
                stpf = stpmin

        # Update the interval of uncertainty. This update does not
        # depend on the new step or the case analysis above.

        if fp > fx:
            sty = stp;
            fy = fp;
            dy = dp;
        else:
            if sgnd < 0.0:
                sty = stx
                fy = fx
                dy = dx
            stx = stp
            fx = fp
            dx = dp

        # Compute the new step and safeguard it.
        stpf = np.minimum(stpmax, stpf);
        stpf = np.maximum(stpmin, stpf);
        stp = stpf
        if brackt & bound:
            if sty > stx:
                stp = np.minimum(stx + self.delta * (sty - stx), stp)
            else:
                stp = np.maximum(stx + self.delta * (sty - stx), stp)

        return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, success

    def run(self, problem, modl, dmodl, logger=None):
        """Method to apply CvSrch stepper"""
        # Writing to log file if any
        if logger:
            logger.addToLog("CVSRCH STEPPER BY STEP-LENGTH BRACKETING")
            logger.addToLog("xtol=%.2e ftol=%.2e gtol=%.2e alpha_min=%.2e alpha_max=%.2e maxfev=%d xtrapf=%.2e"
                            % (
                            self.xtol, self.ftol, self.gtol, self.alpha_min, self.alpha_max, self.maxfev, self.xtrapf))
        success = False
        # Obtain objective function for provided model
        phi_init = problem.get_obj(modl)
        # Getting pointer to problem's gradient vector
        prblm_grad = problem.get_grad(modl)
        dphi_init = prblm_grad.dot(dmodl)
        if dphi_init > 0.0:
            if logger:
                logger.addToLog("\tWarning! Current search direction is not a descent one!")
            return self.alpha, success
        # Model temporary vector
        model_step = modl.clone()
        # Getting pointer to problem's model vector
        prblm_mdl = problem.get_model()
        # Initial step length value
        alpha = deepcopy(self.alpha)
        # Estimating initial step length value
        if alpha < self.zero:
            alpha = self.estimate_initial_guess(problem, modl, dmodl, logger)
        if logger:
            logger.addToLog("\tinitial-steplength=%.2e" % alpha)

        # Initializing parameters
        p5 = 0.5
        cstep_success = True
        fev = 0
        width = self.alpha_max - self.alpha_min
        width1 = 2 * width
        brackt = False
        stage1 = True
        dphi_test = self.ftol * dphi_init

        # The variables alphax, phix, dphix contain the values of the step, function, and directional derivative at the best step.
        # The variables alphay, phiy, dphiy contain the value of the step, function, and derivative at the other endpoint of the interval of uncertainty.
        # The variables alpha, phi_c, dphi_c contain the values of the step, function, and derivative at the current step.
        alphax = 0.0
        phix = phi_init
        dphix = dphi_init
        alphay = 0.0
        phiy = phi_init
        dphiy = dphi_init

        # Start testing iteration
        while True:
            # Set the minimum and maximum steps to correspond to the present interval of uncertainty.
            if brackt:
                alpha_int_min = np.minimum(alphax, alphay)
                alpha_int_max = np.maximum(alphax, alphay)
            else:
                alpha_int_min = alphax
                alpha_int_max = alpha + self.xtrapf * (alpha - alphax)

            # Force the step to be within the bounds alpha_max and alpha_min.
            alpha = np.maximum(alpha, self.alpha_min)
            alpha = np.minimum(alpha, self.alpha_max)

            # If an unusual termination is to occur then choose alpha be the lowest point obtained so far.
            if ((brackt and (alpha <= alpha_int_min or alpha >= alpha_int_max)) or fev >= self.maxfev - 1 or (
                    not cstep_success) or (brackt and alpha_int_max - alpha_int_min <= self.xtol * alpha_int_max)):
                if logger:
                    logger.addToLog("\tUnusual termination is to occur. Setting alpha to be the lowest point obtained so far.")
                alpha = alphax

            # Evaluate the function and gradient at alpha and compute the directional derivative.
            if logger:
                logger.addToLog("\tCurrent testing point (alpha=%.2e): m_current+alpha*dm" % alpha)
            model_step.copy(modl)
            model_step.scaleAdd(dmodl, sc2=alpha)
            # Checking if model parameters hit the bounds
            problem.set_model(model_step)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(model_step)
            if prblm_mdl.isDifferent(model_step):
                # Model hit bounds
                if logger:
                    logger.addToLog("\tModel hit provided bounds. Projecting it onto them.")
            phi_alpha = problem.get_obj(model_step)
            fev += 1
            # Checking if a NaN is encountered
            if isnan(phi_alpha):
                if logger:
                    logger.addToLog("\t!!!Objective function is NaN! Stepper unsuccessful!!!")
                problem.set_model(modl)
                break
            if logger:
                logger.addToLog("\tObjective function value of %.5e (feval = %d)" % (phi_alpha, problem.get_fevals()))
            prblm_grad = problem.get_grad(model_step)
            dphi_alpha = prblm_grad.dot(dmodl)
            phi_test1 = phi_init + alpha * dphi_test

            # Test for convergence
            if (brackt and (alpha <= alpha_int_min or alpha >= alpha_int_max)) or (not cstep_success):
                if logger:
                    logger.addToLog("\tRounding errors prevent further progress. There may not be a step which satisfies "
                                    "the sufficient decrease and curvature conditions. Tolerances may be too small.")
                break
            if alpha == self.alpha_max and phi_alpha <= phi_test1 and dphi_alpha <= dphi_test:
                if logger:
                    logger.addToLog("\tThe step-length value is at the upper bound (alpha_max) of %.2e" % self.alpha_max)
                break
            if alpha == self.alpha_min and (phi_alpha > phi_test1 or dphi_alpha >= dphi_test):
                if logger:
                    logger.addToLog("\tThe step-length value is at the lower bound (alpha_min) of %.2e" % self.alpha_min)
                break
            if fev >= self.maxfev:
                if logger:
                    logger.addToLog("\tNumber of objective function evaluation reached maxfev of %d" % self.maxfev)
                break
            if brackt and alpha_int_max - alpha_int_min <= self.xtol * alpha_int_max:
                if logger:
                    logger.addToLog("\tRelative width of the interval of uncertainty is at most xtol of %.2e" % self.xtol)
                break
            if phi_alpha <= phi_test1 and abs(dphi_test) <= self.gtol * (-dphi_init) and phi_alpha < phi_init:
                success = True
                if logger:
                    logger.addToLog("\tThe sufficient decrease condition and the directional derivative condition hold "
                                    "(i.e., Strong Wolfe conditions met).\n	Stepper successuful for step length value"
                                    "of %.2e and objective function of %.2e (feval = %d)"
                                    % (alpha, phi_alpha, problem.get_fevals()))
                break

            # In the first stage we seek a step for which the modified function has a nonpositive value and
            # nonnegative derivative.
            if stage1 and (phi_alpha <= phi_test1) and (dphi_alpha >= np.minimum(self.ftol, self.gtol) * dphi_init):
                stage1 = False

            # A modified function is used to predict the step only if
            # we have not obtained a step for which the modified
            # function has a nonpositive function value and nonnegative
            # derivative, and if a lower function value has been
            # obtained but the decrease is not sufficient.

            if stage1 and (phi_alpha <= phix) and (phi_alpha > phi_test1):
                # Define the modified function and derivative values.
                phim = phi_alpha - alpha * dphi_test
                phixm = phix - alphax * dphi_test
                phiym = phiy - alphay * dphi_test
                dphim = dphi_alpha - dphi_test
                dphixm = dphix - dphi_test
                dphiym = dphiy - dphi_test

                # Call cstep to update the interval of uncertainty and to compute the new step.
                [alphax, phixm, dphixm, alphay, phiym, dphiym, alpha, phim, dphim, brackt, cstep_success] = self.cstep(
                    alphax, phixm, dphixm, alphay, phiym, dphiym, alpha, phim, dphim, brackt, alpha_int_min,
                    alpha_int_max, logger)

                # Reset the function and gradient values for phi.
                phix = phixm + alphax * dphi_test
                phiy = phiym + alphay * dphi_test
                dphix = dphixm + dphi_test
                dphiy = dphiym + dphi_test

            else:
                # Call cstep to update the interval of uncertainty and to compute the new step.
                [alphax, phix, dphix, alphay, phiy, dphiy, alpha, phi_alpha, dphi_alpha, brackt,
                 cstep_success] = self.cstep(alphax, phix, dphix, alphay, phiy, dphiy, alpha, phi_alpha, dphi_alpha,
                                             brackt, alpha_int_min, alpha_int_max, logger)

            # Force a sufficient decrease in the size of the interval of uncertainty.
            if brackt:
                if abs(alphay - alphax) >= self.delta * width1:
                    alpha = alphax + p5 * (alphay - alphax)
                width1 = width
                width = abs(alphay - alphax)

            # End of iteration.

        if success:
            # Line search has finished, update model
            self.alpha = deepcopy(alpha)
            modl.copy(model_step)

        # Delete temporary vectors
        del model_step
        return alpha, success


class ParabolicStep(Stepper):
    """Parabolic Stepper class with three-point interpolation"""

    def __init__(self, c1=1.0, c2=2.0, ntry=10, alpha=0., alpha_scale_min=1.0e-10, alpha_scale_max=2000.00, shrink=0.25,
                 eval_parab=True):
        """
           Constructor for parabolic stepper with three-point interpolation:
           c1  		   	   = [1.0] - float; Scaling factor of first search point (i.e., m1 = c1*alpha*dm + m_current)
           c2  		   	   = [2.0] - float; Scaling factor of first search point (i.e., m2 = c2*alpha*dm + m_current)
           ntry  	   	   = [10] - integer; Number of trials for finding the step length
           alpha 		   = [0.] - float; Initial step-length guess
           alpha_scale_min = [1.0e-10] - float; Minimum scaling factor (c_optimal) for step-length allowed
           alpha_scale_max = [1000.00] - float; Maximum scaling factor (c_optimal) for step-length allowed
           shrink 		   = [0.25] - float; Shrinking factor if step length is not found at a given trial
           eval_parab 	   = [True] - boolean; Force parabola minimum to be computed. If False, the best point will be chosen from c1 or c2 and the parabola minimum is computed if necessary
        """
        self.c1 = c1  # Scaling for first tested point
        self.c2 = c2  # Scaling for second tested point
        self.ntry = ntry  # Number of total trials before re-estimating initial alpha value
        self.alpha = alpha  # Initial step length guess
        self.alpha_scale_min = alpha_scale_min  # Maximum scaling value for the step length
        self.alpha_scale_max = alpha_scale_max  # Minimum scaling value for the step length
        self.shrink = shrink  # Shrinking scaling factor if trial is unsuccessful
        self.zero = 10 ** (np.floor(
            np.log10(np.abs(float(np.finfo(np.float64).tiny)))) + 2)  # Check for avoid Overflow or Underflow
        self.eval_parab = eval_parab
        return

    def run(self, problem, modl, dmodl, logger=None):
        """Method to apply parabolic stepper"""
        # Writing to log file if any
        global obj1
        if logger:
            logger.addToLog("PARABOLIC STEPPER USING THREE-POINT INTERPOLATION")
            logger.addToLog("c1=%.2e c2=%.2e ntry=%d steplength-scaling-min=%.2e steplength-scaling-max=%.2e shrinking-factor=%.2e"
                            % (self.c1, self.c2, self.ntry, self.alpha_scale_min, self.alpha_scale_max, self.shrink))
        success = False
        # Obtain objective function for provided model
        obj0 = problem.get_obj(modl)
        # Model temporary vector
        model_step = modl.clone()
        # Getting pointer to problem's model vector
        prblm_mdl = problem.get_model()
        # Initial step length value
        alpha = deepcopy(self.alpha)
        # Checking if current search direction is a descending one
        prblm_grad = problem.get_grad(prblm_mdl)
        dphi = prblm_grad.dot(dmodl)
        if dphi > 0.0:
            if logger:
                logger.addToLog("\tWarning! Current search direction is not a descent one!")
            return alpha, success
        itry = 1
        total_trials = deepcopy(self.ntry)
        if alpha != 0.:
            # If initial step length is different than zero, we tests twice in case we need to re-estimate initial alpha
            total_trials *= 2
        while itry <= total_trials:
            # Writing info to log file
            if logger:
                logger.addToLog("\ttrial number: %d" % itry)
                logger.addToLog("\tinitial-steplength=%.2e" % alpha)
            # Find the first guess as if the problem was linear (Tangent method)
            if (itry == self.ntry) or (alpha < self.zero):
                alpha = self.estimate_initial_guess(problem, modl, dmodl, logger)
                if logger:
                    logger.addToLog("\tGuessing step length of: %.2e" % alpha)
            # Test values of objective function for two scaled versions of the step length
            # Testing c1 scale
            if logger:
                logger.addToLog("\tTesting point (c1=%.2e): m_current+c1*alpha*dm" % self.c1)
            model_step.copy(modl)
            model_step.scaleAdd(dmodl, sc2=self.c1 * alpha)
            # Checking if model parameters hit the bounds
            problem.set_model(model_step)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(model_step)
            if prblm_mdl.isDifferent(model_step):
                # Model hit bounds
                msg = "\tModel hit provided bounds. Projecting it onto them."
                if logger:
                    logger.addToLog(msg)
            obj1 = problem.get_obj(model_step)
            # Copying residuals for point c1
            res_prblm = problem.get_res(model_step)
            res1 = res_prblm.clone()
            if logger:
                logger.addToLog("\tObjective function value of %.5e" % obj1)
            # Checking if a NaN is encountered in any of the two tested points
            if isnan(obj1):
                if logger:
                    logger.addToLog("\t!!!Problem with step length and objective function!!!")
                if itry >= self.ntry:
                    if logger:
                        logger.addToLog("\t!!!Check problem definition or change solver!!!")
                    # Setting model to current one and resetting initial step length value
                    alpha = 0.0
                    self.alpha = 0.0
                    problem.set_model(modl)
                    break
                else:
                    if logger:
                        logger.addToLog("\t!!!Guessing linear step length to try to solve problem!!!")
                    itry = self.ntry  # To not repeat computation of linear guess
                    continue
            # Testing c2 scale
            msg = "\tTesting point (c2=%.2e): m_current+c2*alpha*dm" % self.c2
            if logger:
                logger.addToLog(msg)
            model_step.copy(modl)
            model_step.scaleAdd(dmodl, sc2=self.c2 * alpha)
            # Checking if model parameters hit the bounds
            problem.set_model(model_step)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(model_step)
            if prblm_mdl.isDifferent(model_step):
                # Model hit bounds
                msg = "\tModel hit provided bounds. Projecting it onto them."
                if logger:
                    logger.addToLog(msg)
            obj2 = problem.get_obj(model_step)
            # Copying residuals for point c1
            res_prblm = problem.get_res(model_step)
            res2 = res_prblm.clone()
            if logger:
                logger.addToLog("\tObjective function value of %.5e" % obj2)
            # Checking for NaN
            if isnan(obj2):
                if logger:
                    logger.addToLog("\t!!!Problem with step length and objective function!!!")
                if itry >= self.ntry:
                    if logger:
                        logger.addToLog("\t!!!Check problem definition or change solver!!!")
                    # Setting model to current one and resetting initial step length value
                    alpha = 0.0
                    self.alpha = 0.0
                    problem.set_model(modl)
                    break
                else:
                    if logger:
                        logger.addToLog("\t!!!Guessing linear step length to try to solve problem!!!")
                    itry = self.ntry  # To not repeat computation of linear guess
                    continue
            # Checking if parabolic point is necessary or not
            if not self.eval_parab:
                # Setting third point to infinity
                obj3 = np.inf
                # Check which one is the best step length
                msg = "\n\tAs requested, parabola minimum was not evaluated! Unless necessary!"
                if obj1 < obj0 and obj1 < obj2 and obj1 < obj3:
                    success = True
                    alpha *= self.c1
                    if logger:
                        logger.addToLog("\tc1 best step-length value of: %.2e (feval = %d)" % (alpha, problem.get_fevals() - 1) + msg)
                    break
                elif obj2 < obj0 and obj2 < obj1 and obj2 < obj3:
                    success = True
                    alpha *= self.c2
                    if logger:
                        logger.addToLog("\tc2 best step-length value of: %.2e (feval = %d)" % (alpha, problem.get_fevals()) + msg)
                    break
            # If points lay on a horizontal line pick minimum alpha set by user
            if obj0 == obj1 == obj2 or (self.c2 * (obj1 - obj0) + self.c1 * (obj0 - obj2)) == 0.:
                step_scale = self.alpha_scale_min
                if logger:
                    logger.addToLog("\tTwo testing points on a line: cannot fit a parabola, using minimum step-length of %.2e"
                                    % (step_scale * alpha))
            else:
                # Otherwise, find the optimal parabolic step length
                step_scale = 0.5 * (self.c2 * self.c2 * (obj1 - obj0) + self.c1 * self.c1 * (obj0 - obj2)) / (
                        self.c2 * (obj1 - obj0) + self.c1 * (obj0 - obj2))
                if logger:
                    logger.addToLog("\tTesting point (c_opt=%.2e): m_current+c_opt*alpha*dm (parabola minimum)" % step_scale)
            # If step length negative, re-evaluate points
            if step_scale < 0.:
                if logger:
                    logger.addToLog("\tEncountered a negative step-length value: %.2e; Setting parabola-minimum objective function to infinity."
                                    % (step_scale * alpha))
                # Skipping parabola minimum and setting obj3 to infinity
                obj3 = np.inf
            else:
                # Clipping the step-length scale
                if step_scale < self.alpha_scale_min:
                    if logger:
                        logger.addToLog("\t!!! step-length scale of %.2e smaller than provided lower bound."
                                        "Clipping its value to bound value of %.2e !!!" % (step_scale, self.alpha_scale_min))
                    step_scale = self.alpha_scale_min
                elif step_scale > self.alpha_scale_max:
                    if logger:
                        logger.addToLog("\t!!! step-length scale of %.2e greater than provided upper bound."
                                        "Clipping its value to bound value of %.2e !!!" % (step_scale, self.alpha_scale_max))
                    step_scale = self.alpha_scale_max

                # Testing parabolic scale
                # Compute new objective function at the minimum of the parabolic approximation
                model_step.copy(modl)
                model_step.scaleAdd(dmodl, sc2=step_scale * alpha)
                # Checking if model parameters hit the bounds
                problem.set_model(model_step)
                # Projecting model onto the bounds (if any)
                if "bounds" in dir(problem):
                    problem.bounds.apply(model_step)
                if prblm_mdl.isDifferent(model_step):
                    # Model hit bounds
                    msg = "\tModel hit provided bounds. Projecting it onto them."
                    if logger:
                        logger.addToLog(msg)
                obj3 = problem.get_obj(model_step)
                if logger:
                    logger.addToLog("\tObjective function value of %.5e" % obj3)

            # Writing info to log file
            if logger:
                logger.addToLog("\tInitial objective function value: %.5e,"
                                "Objective function at c1*alpha*dm: %.5e,"
                                "Objective function at c2*alpha*dm: %.5e,"
                                "Objective function at parabola minimum: %.5e"
                                % (obj0, obj1, obj2, obj3))
            itry += 1

            # Check which one is the best step length
            if obj1 < obj0 and obj1 < obj2 and obj1 < obj3:
                success = True
                alpha *= self.c1
                if logger:
                    logger.addToLog("\tc1 best step-length value of: %.2e (feval = %d)" % (alpha, problem.get_fevals() - 2))
                break
            elif obj2 < obj0 and obj2 < obj1 and obj2 < obj3:
                success = True
                alpha *= self.c2
                if logger:
                    logger.addToLog("\tc2 best step-length value of: %.2e (feval = %d)" % (alpha, problem.get_fevals() - 1))
                break
            elif obj3 < obj0 and obj3 <= obj1 and obj3 <= obj2:
                success = True
                alpha *= step_scale
                if logger:
                    logger.addToLog("\tparabola minimum best step-length value of: %.2e (feval = %d)" % (alpha, problem.get_fevals()))
                break
            else:
                # Shrink line search
                alpha *= self.shrink
                if logger:
                    logger.addToLog("\tShrinking search direction")

        if success:
            # Line search has finished, update model
            self.alpha = deepcopy(alpha)
            model_step.copy(modl)  # model_step = m_current
            model_step.scaleAdd(dmodl, sc2=self.alpha)
            # Checking if model parameters hit the bounds
            modl.copy(model_step)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(model_step)
            if modl.isDifferent(model_step):
                # Computing true scaled search direction dm = m_new_clipped - m_current
                dmodl.copy(model_step)
                dmodl.scaleAdd(modl, 1.0, -1.0)
                # Scaled by the inverse of the step length
                dmodl.scale(1.0 / self.alpha)
            # Setting model and residual vectors to c1 or c2 point if parabola minimum is not picked
            problem.set_model(model_step)
            if obj1 < obj0 and obj1 < obj2 and obj1 < obj3:
                problem.set_residual(res1)
            elif obj2 < obj0 and obj2 < obj1 and obj2 < obj3:
                problem.set_residual(res2)
            modl.copy(model_step)
        # Delete temporary vectors
        del model_step, res1, res2
        return alpha, success


class ParabolicStepConst(Stepper):
    """Parabolic Stepper class assuming constant local curvature"""

    def __init__(self, c1=1.0, ntry=10, alpha=0., alpha_scale_min=1.0e-10, alpha_scale_max=2000.00, shrink=0.25):
        """
           Constructor for parabolic stepper assuming constant local curvature:
           c1  		   	   = [1.0] - float; Scaling factor of the search point (i.e., m1 = c1*alpha*dm + m_current)
           ntry  	   	   = [10] - integer; Number of trials for finding the step length
           alpha 		   = [0.] - float; Initial step-length guess
           alpha_scale_min = [1.0e-10] - float; Minimum scaling factor (c_optimal) for step-length allowed
           alpha_scale_max = [1000.00] - float; Maximum scaling factor (c_optimal) for step-length allowed
           shrink 		   = [0.25] - float; Shrinking factor if step length is not found at a given trial
        """
        self.c1 = c1  # Scaling for first tested point
        self.ntry = ntry  # Number of total trials before re-estimating initial alpha value
        self.alpha = alpha  # Initial step length guess
        self.alpha_scale_min = alpha_scale_min  # Minimum scaling value for the step length
        self.alpha_scale_max = alpha_scale_max  # Maximum scaling value for the step length
        self.shrink = shrink  # Shrinking scaling factor if trial is unsuccessful
        self.zero = 10 ** (np.floor(
            np.log10(np.abs(float(np.finfo(np.float64).tiny)))) + 2)  # Check for avoid Overflow or Underflow
        return

    def run(self, problem, modl, dmodl, logger=None):
        """Method to apply parabolic stepper"""
        # Writing to log file if any
        if logger:
            logger.addToLog("PARABOLIC STEPPER ASSUMING CONSTANT LOCAL CURVATURE")
            logger.addToLog("c1=%.2e ntry=%d steplength-scaling-min=%.2e steplength-scaling-max=%.2e shrinking-factor=%.2e"
                            % (self.c1, self.ntry, self.alpha_scale_min, self.alpha_scale_max, self.shrink))
        success = False
        # Obtain objective function for provided model
        obj0 = problem.get_obj(modl)
        # Model temporary vector
        model_step = modl.clone()
        # Getting pointer to problem's model vector
        prblm_mdl = problem.get_model()
        # Initial step length value
        alpha = deepcopy(self.alpha)
        # Getting pointer to problem's gradient vector
        prblm_grad = problem.get_grad(prblm_mdl)
        dphi = prblm_grad.dot(dmodl)
        if dphi > 0.0:
            if logger:
                logger.addToLog("\tWarning! Current search direction is not a descent one!")
            return alpha, success
        itry = 1
        total_trials = deepcopy(self.ntry)
        if alpha != 0.:
            # If initial step length is different than zero, we tests twice in case we need to re-estimate initial alpha
            total_trials *= 2
        while itry <= total_trials:
            # Writing info to log file
            if logger:
                logger.addToLog("\ttrial number: %d" % itry)
                logger.addToLog("\tinitial-steplength=%.2e" % alpha)
            # Find the first guess as if the problem was linear (Tangent method)
            if (itry == self.ntry) or (alpha < self.zero):
                alpha = self.estimate_initial_guess(problem, modl, dmodl, logger)
                if logger:
                    logger.addToLog("\tGuessing step length of: %.2e" % alpha)
            # Test values of objective function for two scaled versions of the step length
            # Testing c1 scale
            if logger:
                logger.addToLog("\tTesting point (c1=%.2e): m_current+c1*alpha*dm" % self.c1)
            model_step.copy(modl)
            model_step.scaleAdd(dmodl, sc2=self.c1 * alpha)
            # Checking if model parameters hit the bounds
            problem.set_model(model_step)
            # Projecting model onto the bounds (if any) and rotate search direction
            if "bounds" in dir(problem):
                problem.bounds.apply(model_step)
            if prblm_mdl.isDifferent(model_step):
                # Model hit bounds
                msg = "\tModel hit provided bounds. Projecting it onto them."
                if logger:
                    logger.addToLog(msg)
                # Computing true scaled search direction dm = m_new_clipped - m_current
                dmodl.copy(model_step)
                dmodl.scaleAdd(modl, 1.0, -1.0)
                # Scaled by the inverse of the step length
                dmodl.scale(1.0 / (self.c1 * alpha))
            obj1 = problem.get_obj(model_step)
            # Copying residuals for point c1
            res_prblm = problem.get_res(model_step)
            res1 = res_prblm.clone()
            if logger:
                logger.addToLog("\tObjective function value of %.5e" % obj1)
            # Checking if a NaN is encountered in any of the two tested points
            if isnan(obj1):
                if logger:
                    logger.addToLog("\t!!!Problem with step length and objective function!!!")
                if itry >= self.ntry:
                    if logger:
                        logger.addToLog("\t!!!Check problem definition or change solver!!!")
                    # Setting model to current one and resetting initial step length value
                    alpha = 0.0
                    self.alpha = 0.0
                    problem.set_model(modl)
                    break
                else:
                    if logger:
                        logger.addToLog("\t!!!Guessing linear step length to try to solve problem!!!")
                    itry = self.ntry  # To not repeat computation of linear guess
                    continue
            # Computing local constant curvature
            phi_der = prblm_grad.dot(dmodl)  # First derivative of the objective function with respect to alpha
            c = 2.0 * ((obj1 - obj0) / (self.c1 * alpha * self.c1 * alpha) - phi_der / (self.c1 * alpha))
            # Checking the curvature value
            if c <= 0.:
                # Shrink line search
                alpha *= self.shrink
                if logger:
                    logger.addToLog("\tEstimated a negative curvature of %.2e. Shrinking search direction" % c)
                itry += 1
                continue
            # Computing objective function at local parabola minimum
            alpha_parab = - phi_der / c
            step_scale = alpha_parab / alpha
            if logger:
                logger.addToLog("\tTesting point (c_opt=%.2e): m_current+c_opt*alpha*dm (parabola minimum)" % step_scale)
            # If step length negative, re-evaluate points
            if alpha_parab < 0.:
                if logger:
                    logger.addToLog("\tEncountered a negative step-length value: %.2e; Shrinking step-length value." % alpha_parab)
                # Shrink line search
                alpha *= self.shrink
                itry += 1
                continue
            # Clipping the step-length scale
            if step_scale < self.alpha_scale_min:
                if logger:
                    logger.addToLog("\t!!! step-length scale of %.2e smaller than provided lower bound."
                                    "Clipping its value to bound value of %.2e !!!" % (step_scale, self.alpha_scale_min))
                step_scale = self.alpha_scale_min
            elif step_scale > self.alpha_scale_max:
                if logger:
                    logger.addToLog("\t!!! step-length scale of %.2e greater than provided upper bound."
                                    "Clipping its value to bound value of %.2e !!!" % (step_scale, self.alpha_scale_max))
                step_scale = self.alpha_scale_max

            # Testing parabolic scale
            # Compute new objective function at the minimum of the parabolic approximation
            model_step.copy(modl)
            model_step.scaleAdd(dmodl, sc2=alpha * step_scale)
            # Checking if model parameters hit the bounds
            problem.set_model(model_step)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(model_step)
            if prblm_mdl.isDifferent(model_step):
                # Model hit bounds
                if logger:
                    logger.addToLog("\tModel hit provided bounds. Projecting it onto them.")
            obj2 = problem.get_obj(model_step)
            if logger:
                logger.addToLog("\tObjective function value of %.5e" % obj2)

            # Writing info to log file
            if logger:
                logger.addToLog("\tInitial objective function value: %2e,"
                                "Objective function at c1*alpha*dm: %.2e,"
                                "Objective function at parabola minimum: %.2e"
                                % (obj0, obj1, obj2))
            itry += 1

            # Check which one is the best step length
            if obj1 < obj0 and obj1 < obj2:
                success = True
                alpha *= self.c1
                if logger:
                    logger.addToLog("\tc1 best step-length value of: %.2e (feval = %d)" % (alpha, problem.get_fevals() - 1))
                break
            elif obj2 < obj0 and obj2 <= obj1:
                success = True
                alpha *= step_scale
                if logger:
                    logger.addToLog("\tparabola minimum best step-length value of: %.2e (feval = %d)" % (alpha, problem.get_fevals()))
                break
            else:
                # Shrink line search
                alpha *= self.shrink
                if logger:
                    logger.addToLog("\tShrinking search direction")

        if success:
            # Line search has finished, update model
            self.alpha = deepcopy(alpha)
            model_step.copy(modl)  # model_step = m_current
            model_step.scaleAdd(dmodl, sc2=self.alpha)
            # Checking if model parameters hit the bounds
            modl.copy(model_step)
            # Projecting model onto the bounds (if any)
            if "bounds" in dir(problem):
                problem.bounds.apply(model_step)
            if modl.isDifferent(model_step):
                # Computing true scaled search direction dm = m_new_clipped - m_current
                dmodl.copy(model_step)
                dmodl.scaleAdd(modl, 1.0, -1.0)
                # Scaled by the inverse of the step length
                dmodl.scale(1.0 / self.alpha)
            # Setting model and residual vectors to c1 or c2 point if parabola minimum is not picked
            problem.set_model(model_step)
            if obj1 < obj0 and obj1 < obj2:
                problem.set_residual(res1)
            modl.copy(model_step)
        # Delete temporary vectors
        del model_step, res1
        return alpha, success

class StrongWolfe(Stepper):
    """Compute a line search to satisfy the strong Wolfe conditions.
       Algorithm 3.5. Page 60. "Numerical Optimization". Nocedal & Wright.
       Implementation based on the ones in the GitHub repo: https://github.com/bgranzow/L-BFGS-B.git
       """

    def __init__(self, c1=1.e-4, c2=0.9 , ntry=20, alpha=1., alpha_scale=0.8, alpha_max=2.5, keepAlpha=False):
        """
           Constructor for parabolic stepper assuming constant local curvature:
           c1  		   	   = [1.e-4] - float; c1 value to tests first Wolfe condition (should be between 0 and 1)
           c2  		   	   = [0.9] - float; c2 value to tests second Wolfe condition (should be between c1 and 1). For Quasi-Newton (e.g., L-BFGS) choose default. Otherwise, for other methods (e.g., NLCG) choose 0.1
           ntry  	   	   = [20] - integer; Number of trials for finding the step length
           alpha 		   = [1.] - float; Initial step-length guess
           alpha_scale 	   = [0.8] - float; step-length update factor used for updating step length guess
           alpha_max       = [2.5] - float; Maximum step-length value allowed
           keepAlpha       = [False] - boolean; Whether to keep or forget previously found step-length value
        """
        self.c1 = c1
        self.c2 = c2
        self.ntry = ntry  # Number of total trials before re-estimating initial alpha value
        self.alpha = alpha  # Initial step length guess
        self.alpha_max = alpha_max  # Maximum step-length value
        self.alpha_scale = alpha_scale
        self.zero = 10 ** (np.floor(
            np.log10(np.abs(float(np.finfo(np.float64).tiny)))) + 2)  # Check for avoid Overflow or Underflow
        self.keepAlpha = keepAlpha
        return

    def alpha_zoom(self, problem, mdl0, mdl, obj0, dphi0, dmodl, alpha_lo, alpha_hi, logger=None):
        """Algorithm 3.6, Page 61. "Numerical Optimization". Nocedal & Wright."""
        itry = 0
        alpha = 0.0
        while itry < self.ntry:
            if logger:
                logger.addToLog("\t\ttrial number [alpha_zoom]: %d" % (itry+1))
            alpha_i = 0.5 * (alpha_lo + alpha_hi)
            alpha = alpha_i
            # x = x0 + alpha_i * p
            mdl.copy(mdl0)
            mdl.scaleAdd(dmodl, sc2=alpha_i)
            # Evaluating objective and gradient function
            obj_i = problem.get_obj(mdl)
            if logger:
                logger.addToLog("\t\tObjective function value of %.5e at m_i with alpha=%.5e [alpha_zoom]" %(obj_i, alpha_i))
            if isnan(obj_i):
                if logger:
                    logger.addToLog("\t\t!!!Problem with step length and objective function; Setting alpha = 0.0 [alpha_zoom]!!!")
                alpha = 0.0
                break
            grad_i = problem.get_grad(mdl)
            # x_lo = x0 + alpha_lo * p;
            mdl.copy(mdl0)
            mdl.scaleAdd(dmodl, sc2=alpha_lo)  # x = x0 + alpha_i * p;
            obj_lo = problem.get_obj(mdl)
            if logger:
                logger.addToLog("\t\tObjective function value of %.5e at m_lo with alpha_lo=%.5e [alpha_zoom]" %(obj_lo, alpha_lo))
            if isnan(obj_lo):
                if logger:
                    logger.addToLog("\t\t!!!Problem with step length and objective function; Setting alpha = 0.0 [alpha_zoom]!!!")
                alpha = 0.0
                break
            if (obj_i > obj0 + self.c1 * alpha_i * dphi0) or (obj_i >= obj_lo):
                alpha_hi = alpha_i
            else:
                dphi = grad_i.dot(dmodl)
                if np.abs(dphi) <= -self.c2 * dphi0:
                    alpha = alpha_i
                    break
                if dphi * (alpha_hi - alpha_lo) >= 0.0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_i
            # Increasing trial counter of alpha zoom
            itry += 1
            if itry > self.ntry:
                alpha = alpha_i
                break
        return alpha

    def run(self, problem, modl, dmodl, logger=None):
        """Method to apply line search that satisfies strong Wolfe conditions"""
        # Writing to log file if any
        if logger:
            logger.addToLog("STRONG-WOLFE STEPPER")
            logger.addToLog(
                "c1=%.2e c2=%.2e ntry=%d alpha-max=%.2e keepAlpha=%s"
                % (self.c1, self.c2, self.ntry, self.alpha_max, self.keepAlpha))
        success = False
        # Obtain objective function for provided model
        obj0 = problem.get_obj(modl)
        obj_im1 = deepcopy(obj0)
        # Model temporary vector
        model_step = modl.clone()
        # Initial step length value
        alpha_i = deepcopy(self.alpha)
        alpha_im1 = 0.0
        alpha = 0.0
        # Getting pointer to problem's gradient vector
        prblm_grad = problem.get_grad(modl).clone()
        dphi0 = prblm_grad.dot(dmodl)
        itry = 0
        while itry < self.ntry:
            # Writing info to log file
            if logger:
                logger.addToLog("\ttrial number: %d" % (itry+1))
                logger.addToLog("\tinitial-steplength=%.2e" % alpha_i)
            # Find the first guess as if the problem was linear (Tangent method)
            if alpha_i <= self.zero:
                alpha_i = self.estimate_initial_guess(problem, modl, dmodl, logger)
                self.alpha_max *= alpha_i
                if logger:
                    logger.addToLog("\tGuessing step length of: %.2e" % alpha_i)

            # Updating model point
            model_step.copy(modl) # x = x0
            model_step.scaleAdd(dmodl, sc2=alpha_i) # x = x0 + alpha_i * p;
            # Evaluating objective and gradient function
            obj_i = problem.get_obj(model_step)
            if logger:
                logger.addToLog("\tObjective function value of %.5e" % obj_i)
            if isnan(obj_i):
                if logger:
                    logger.addToLog("\t!!!Problem with step length and objective function; Stopping stepper!!!")
                break
            grad_i = problem.get_grad(model_step)
            if (obj_i > obj0 + self.c1 * dphi0) or ((itry > 1) and (obj_i >= obj_im1)):
                alpha = self.alpha_zoom(problem, modl, model_step, obj0, dphi0, dmodl, alpha_im1, alpha_i, logger)
                if logger:
                    logger.addToLog("\tCondition 1 matched; step-length value of: %.2e" % alpha)
                success = True
                break

            # dphi = transpose(g_i) * p;
            dphi = grad_i.dot(dmodl)
            if np.abs(dphi) <= -self.c2 * dphi0:
                alpha = alpha_i
                if logger:
                    logger.addToLog("\tCondition 2 matched; step-length value of: %.2e" % alpha)
                success = True
                break
            if dphi >= self.zero:
                alpha = self.alpha_zoom(problem, modl, model_step, obj0, dphi0, dmodl, alpha_i, alpha_im1, logger)
                if logger:
                    logger.addToLog("\tCondition 3 matched; step-length value of: %.2e" % alpha)
                success = True
                break

            # Update step-length value
            alpha_im1 = alpha_i
            obj_im1 = obj_i
            alpha_i = alpha_i + self.alpha_scale * (self.alpha_max - alpha_i)

            if itry > self.ntry:
                alpha = alpha_i
                if logger:
                    logger.addToLog("\tMaximum number of trials reached (ntry = %d); step-length value of: %.2e" %(self.ntry,alpha))
                success = True
                break

            # Update trial number
            itry += 1
        if success:
            # Line search has finished, update model
            modl.scaleAdd(dmodl, sc2=alpha)
            if self.keepAlpha:
                self.alpha = alpha
        # Delete temporary vectors
        del model_step

        return alpha, success