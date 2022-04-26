from math import isnan

from occamypy.vector.base import superVector
from occamypy.operator.linear import Identity
from occamypy.operator.nonlinear import NonlinearOperator, NonlinearVstack, VarProOperator
from occamypy.problem.base import Problem
from occamypy.problem.linear import LeastSquares, LeastSquaresRegularized


class NonlinearLeastSquares(Problem):
    r"""
    Nonlinear inverse problem of the form

    .. math::
        \frac{1}{2} \Vert f(\mathbf{m}) - \mathbf{d}\Vert_2^2
    """
    
    def __init__(self, model, data, op, grad_mask=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        NonlinearLeastSquares constructor
        
        Args:
            model: initial domain vector
            data: data vector
            op: nonlinear operator
            grad_mask: mask to be applied on the gradient during the inversion
            minBound: lower bound vector
            maxBound: upper bound vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(NonlinearLeastSquares, self).__init__(model=model, data=data, minBound=minBound, maxBound=maxBound, boundProj=boundProj,
                                                    name="Nonlinear Least Squares")
        
        # Gradient vector
        self.grad = self.pert_model.clone()
        
        # Setting non-linear and linearized operators
        if isinstance(op, NonlinearOperator):
            self.op = op
        else:
            raise TypeError("Not provided a non-linear operator!")
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = False
    
    def res_func(self, model):
        r"""
        Method to return residual vector

        .. math::
            \mathbf{r} = f(\mathbf{m}) - \mathbf{d}
        """
        self.op.nl_op.forward(False, model, self.res)
        # Computing f(m) - d
        self.res.scaleAdd(self.data, 1., -1.)
        return self.res
    
    def grad_func(self, model, residual):
        r"""
        Method to return gradient vector

        .. math::
            \mathbf{g} = \mathbf{F}'(\mathbf{m}) \mathbf{r} = \mathbf{F}'(\mathbf{m}) [f(\mathbf{m}) - \mathbf{d}]
        """
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # Computing F'r = g
        self.op.lin_op.adjoint(False, self.grad, residual)
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad
    
    def pert_res_func(self, model, pert_model):
        r"""
        Method to return residual vector

        .. math::
            \mathbf{d}_r = \mathbf{G} \mathbf{d}_m
        """
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # Computing Fdm = pert_res
        self.op.lin_op.forward(False, pert_model, self.pert_res)
        return self.pert_res
    
    def obj_func(self, residual):
        r"""
        Method to return objective function value

        .. math::
            \frac{1}{2} \Vert f(\mathbf{m})-\mathbf{d}\Vert_2^2
        """
        val = residual.norm()
        obj = 0.5 * val * val
        return obj


class NonlinearLeastSquaresRegularized(Problem):
    r"""
    Nonlinear inverse problem with a linear regularization

    .. math::
        \frac{1}{2} \Vert f(\mathbf{m}) - \mathbf{d} \Vert_2^2 +
        \frac{\varepsilon^2}{2} \Vert \mathbf{R} \mathbf{m} - \mathbf{m}_p\Vert_2^2

    or nonlinear regularization

    .. math::
        \frac{1}{2} \Vert f(\mathbf{m}) - \mathbf{d}\Vert_2^2 +
        \frac{\varepsilon^2}{2} \Vert g(\mathbf{m}) - \mathbf{m}_p|_2^2
    """
    
    def __init__(self, model, data, op, epsilon, grad_mask=None, reg_op=None, prior_model=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        NonlinearLeastSquaresRegularized constructor
        
        Args:
            model: initial domain vector
            data: data vector
            op: nonlinear operator
            epsilon: regularization weight
            grad_mask: mask to be applied on the gradient during the inversion
            reg_op: linear or nonlinear regularizer operator
            prior_model: prior vector for the regularization term
            minBound: lower bound vector
            maxBound: upper bound vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(NonlinearLeastSquaresRegularized, self).__init__(model=model, data=data, minBound=minBound, maxBound=maxBound, boundProj=boundProj,
                                                               name="Regularized Nonlinear Least Squares")
        
        # Gradient vector
        self.grad = self.pert_model.clone()
       
        # Setting a prior model (if any)
        self.prior_model = prior_model
        # Setting linear operators
        # Assuming identity operator if regularization operator was not provided
        if reg_op is None:
            Id_op = Identity(self.model)
            reg_op = NonlinearOperator(Id_op, Id_op)
        # Checking if space of the prior model is constistent with range of regularization operator
        if self.prior_model is not None:
            if not self.prior_model.checkSame(reg_op.range):
                raise ValueError("Prior model space no constistent with range of regularization operator")
        # Setting non-linear and linearized operators
        if not isinstance(op, NonlinearOperator):
            raise TypeError("Not provided a non-linear operator!")
        # Setting non-linear stack of operators
        self.op = NonlinearVstack(op, reg_op)
        self.epsilon = epsilon  # Regularization weight
        # Residual vector (data and model residual vectors)
        self.res = self.op.nl_op.range.clone()
        self.res.zero()
        # Dresidual vector
        self.pert_res = self.res.clone()
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = False
        # Objective function terms (useful to analyze each term)
        self.obj_terms = [None, None]
    
    def estimate_epsilon(self, verbose=False, logger=None):
        """
        Estimate the epsilon that balances the two terms of the objective function
        
        Args:
           verbose: whether to print messages or not
           logger: occamypy.Logger instance to log the estimate
        
        Returns:
           estimated epsilon
        """
        msg = "Epsilon Scale evaluation"
        if verbose:
            print(msg)
        if logger:
            logger.addToLog("REGULARIZED PROBLEM log file\n" + msg)
        # Keeping the initial model vector
        prblm_mdl = self.get_model()
        # Keeping user-predefined epsilon if any
        epsilon = self.epsilon
        # Setting epsilon to one to evaluate the scale
        self.epsilon = 1.0
        prblm_res = self.get_res(prblm_mdl)  # Compute residual arising from the gradient
        # Balancing the two terms of the objective function
        res_data_norm = prblm_res.vecs[0].norm()
        res_model_norm = prblm_res.vecs[1].norm()
        if isnan(res_model_norm) or isnan(res_data_norm):
            raise ValueError("Obtained NaN: Residual-data-side-norm = %s, Residual-model-side-norm = %s"
                             % (res_data_norm, res_model_norm))
        if res_model_norm == 0.:
            msg = "Trying to perform a linearized step"
            if verbose:
                print(msg)
            prblm_grad = self.get_grad(prblm_mdl)  # Compute first gradient
            # Gradient in the data space
            prblm_dgrad = self.get_pert_res(prblm_mdl, prblm_grad)
            # Computing linear step length
            dgrad0_res = prblm_res.vecs[0].dot(prblm_dgrad.vecs[0])
            dgrad0_dgrad0 = prblm_dgrad.vecs[0].dot(prblm_dgrad.vecs[0])
            if isnan(dgrad0_res) or isnan(dgrad0_dgrad0):
                raise ValueError("Obtained NaN: gradient-dataspace-norm = %s, gradient-dataspace-dot-residuals = %s"
                                 % (dgrad0_dgrad0, dgrad0_res))
            if dgrad0_dgrad0 != 0.:
                alpha = -dgrad0_res / dgrad0_dgrad0
            else:
                msg = "Cannot compute linearized alpha for the given problem! Provide a different initial model"
                if logger:
                    logger.addToLog(msg)
                raise ValueError(msg)
            # model=model+alpha*grad
            prblm_mdl.scaleAdd(prblm_grad, 1.0, alpha)
            prblm_res = self.res_func(prblm_mdl)
            # Recompute the new objective function terms
            res_data_norm = prblm_res.vecs[0].norm()
            res_model_norm = prblm_res.vecs[1].norm()
            # If regularization term is still zero, stop the solver
            if res_model_norm == 0.:
                msg = "Model residual component norm is zero, cannot find epsilon scale! Provide a different initial model"
                if logger:
                    logger.addToLog(msg)
                raise ValueError(msg)
        # Resetting user-predefined epsilon if any
        self.epsilon = epsilon
        epsilon_balance = res_data_norm / res_model_norm
        # Setting default variables
        self.setDefaults()
        self.linear = False
        msg = "	Epsilon balancing the the two objective function terms is: %.2e" % epsilon_balance
        if verbose:
            print(msg)
        if logger:
            logger.addToLog(msg + "\nREGULARIZED PROBLEM end log file")
        return epsilon_balance
    
    def res_func(self, model):
        r"""
        Method to return residual vector
        
        .. math::
            \begin{bmatrix}
                \mathbf{r}_{d}  \\
                \mathbf{r}_{m}  \\
            \end{bmatrix} =
            \begin{bmatrix}
                f(\mathbf{m}) - \mathbf{d}  \\
                g(\mathbf{m}) - \mathbf{m}_p  \\
            \end{bmatrix}
        """
        self.op.nl_op.forward(False, model, self.res)
        # Computing r_d = f(m) - d
        self.res.vecs[0].scaleAdd(self.data, 1., -1.)
        # Computing r_m = Am - m_prior
        if self.prior_model is not None:
            self.res.vecs[1].scaleAdd(self.prior_model, 1., -1.)
        # Scaling by epsilon epsilon*r_m
        self.res.vecs[1].scale(self.epsilon)
        return self.res
    
    def grad_func(self, model, residual):
        r"""
        Method to return gradient vector
        
        .. math::
            \mathbf{g} = \mathbf{F}' \mathbf{r}_d + \varepsilon \mathbf{G}' \mathbf{r}_m
        """
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # g = epsilon*op'r_m
        self.op.lin_op.ops[1].adjoint(False, self.grad, residual.vecs[1])
        self.grad.scale(self.epsilon)
        # g = F'r_d + op'(epsilon*r_m)
        self.op.lin_op.ops[0].adjoint(True, self.grad, residual.vecs[0])
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad
    
    def pert_res_func(self, model, pert_model):
        r"""
        Method to return residual vector
        
        .. math::
            \mathbf{d}_r = [\mathbf{F} + \varepsilon \mathbf{G}] \mathbf{d}_m
        """
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # Computing Ldm = dres_d
        self.op.lin_op.forward(False, pert_model, self.pert_res)
        # Scaling by epsilon
        self.pert_res.vecs[1].scale(self.epsilon)
        return self.pert_res
    
    def obj_func(self, residual):
        r"""
        Method to return objective function value
        
        .. math::
            \frac{1}{2} \Vert \mathbf{r}_d \Vert_2^2 +
            \frac{\varepsilon^2}{2} \Vert \mathbf{r}_{m} \Vert_2^2
        """
        # data term
        val = residual.vecs[0].norm()
        self.obj_terms[0] = 0.5 * val * val
        # model term
        val = residual.vecs[1].norm()
        self.obj_terms[1] = 0.5 * val * val
        obj = self.obj_terms[0] + self.obj_terms[1]
        return obj


class VarProRegularized(Problem):
    r"""
    Non-linear inverse problem of the form
    
    .. math::
        \phi(\mathbf{m}) = \frac{1}{2} \Vert g(\mathbf{m}_nl) + h(\mathbf{m}_nl)\mathbf{m}_lin - \mathbf{d}\Vert_2^2
        + \frac{\varepsilon^2}{2} \Vert g'(\mathbf{m}_nl) + h'(\mathbf{m}_nl)\mathbf{m}_lin - \mathbf{d}'\Vert_2^2

    in which part of the domain parameters define a quadratic function;
    the non-linear component is solved using the variable-projection method (Golub and Pereyra, 1973)

    Notes:
        To save the results of the linear inversion, use the `lin_solver.setDefaults` method.
        The results can only be saved on files. To the prefix specified within the lin_solver f_eval_# will be added.
    """
    
    def __init__(self, model_nl, lin_model, h_op, data, lin_solver, g_op=None, g_op_reg=None, h_op_reg=None,
                 data_reg=None, epsilon=None, minBound=None, maxBound=None, boundProj=None, prec=None,
                 warm_start=False):
        """
        VarProRegularized constructor
        
        Args:
            model_nl: initial domain vector for the nonlinear component
            lin_model: initial domain vector for the linear component
            h_op: VarPro operator
            data: data vector
            lin_solver: Solver instance for the linear part
            g_op: nonlinear operator
            g_op_reg: nonlinear regularizer operator
            h_op_reg: VarPro regularizer operator
            data_reg: data vector for regularization term
            epsilon: regularization weight
            minBound: lower bound vector
            maxBound: upper bound vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
            prec: preconditioner linear operator
            warm_start: start VP problem from previous linearly inverted domain
        """
        if not isinstance(h_op, VarProOperator):
            raise TypeError("ERROR! Not provided an operator class for the variable projection problem")
        # Setting the bounds (if any)
        super(VarProRegularized, self).__init__(model=model_nl, data=data, minBound=minBound, maxBound=maxBound, boundProj=boundProj,
                                                name="Regularized Variable Projection")
        
        # Linear component of the inverted model
        self.lin_model = lin_model
        self.lin_model.zero()
        
        # Setting non-linear/linear operator
        if not isinstance(h_op, VarProOperator):
            raise TypeError("ERROR! Provide a VpOperator operator class for h_op")
        self.h_op = h_op
        # Setting non-linear operator (if any)
        self.g_op = g_op
        # Verifying if a regularization is requested
        self.epsilon = epsilon
        # Setting non-linear regularization operator
        self.g_op_reg = g_op_reg
        # Setting non-linear/linear operator
        self.h_op_reg = h_op_reg
        # Setting data term in regularization
        self.data_reg = data_reg
        if self.h_op_reg is not None and self.epsilon is None:
            raise ValueError("ERROR! Epsilon value must be provided if a regularization term is requested.")
        # Residual vector
        if self.epsilon is not None:
            # Creating regularization residual vector
            res_reg = None
            if self.g_op_reg is not None:
                res_reg = self.g_op_reg.nl_op.range.clone()
            elif self.h_op_reg is not None:
                if not isinstance(h_op_reg, VarProOperator):
                    raise TypeError("ERROR! Provide a VpOperator operator class for h_op_reg")
                res_reg = self.h_op_reg.h_lin.range.clone()
            elif self.data_reg is not None:
                res_reg = self.data_reg.clone()
            # Checking if a residual vector for the regularization term was created
            if res_reg is None:
                raise ValueError("ERROR! If epsilon is provided, then a regularization term must be provided")
            self.res = superVector(data.clone(), res_reg)
            # Objective function terms (useful to analyze each term)
            self.obj_terms = [None, None]
        else:
            self.res = data.clone()
        # Instantiating linear inversion problem
        if self.h_op_reg is not None:
            self.vp_linear_prob = LeastSquaresRegularized(self.lin_model, self.data, self.h_op.h_lin, self.epsilon,
                                                          reg_op=self.h_op_reg.h_lin, prior_model=self.data_reg,
                                                          prec=prec)
        else:
            self.vp_linear_prob = LeastSquares(self.lin_model, self.data, self.h_op.h_lin, prec=prec)
        
        # Zeroing out the residual vector
        self.res.zero()
        
        # Dresidual vector
        self.pert_res = self.res.clone()
        # Gradient vector
        self.grad = self.pert_model.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = False
        # Linear solver for inverting quadratic component
        self.lin_solver = lin_solver
        self.lin_solver.flush_memory = True
        self.lin_solver_prefix = self.lin_solver.prefix
        self.vp_linear_prob.linear = True
        self.warm_start = warm_start
    
    def estimate_epsilon(self, verbose=False, logger=None):
        """
        Estimate the epsilon that balances the two terms of the objective function

        Args:
           verbose: whether to print messages or not
           logger: occamypy.Logger instance to log the estimate

        Returns:
           estimated epsilon
        """
        if self.epsilon is None:
            raise ValueError("ERROR! Problem is not regularized, cannot evaluate epsilon value!")
        if self.g_op_reg is not None and self.h_op_reg is None:
            # Problem is non-linearly regularized
            msg = "Epsilon Scale evaluation"
            if verbose: print(msg)
            if logger: logger.addToLog("REGULARIZED PROBLEM log file\n" + msg)
            # Keeping the initial model vector
            prblm_mdl = self.get_model()
            # Keeping user-predefined epsilon if any
            epsilon = self.epsilon
            # Setting epsilon to one to evaluate the scale
            self.epsilon = 1.0
            prblm_res = self.get_res(prblm_mdl)  # Compute residual arising from the gradient
            # Balancing the two terms of the objective function
            res_data_norm = prblm_res.vecs[0].norm()
            res_model_norm = prblm_res.vecs[1].norm()
            if isnan(res_model_norm) or isnan(res_data_norm):
                raise ValueError("ERROR! Obtained NaN: Residual-data-side-norm = %s, Residual-model-side-norm = %s" % (
                    res_data_norm, res_model_norm))
            if res_model_norm == 0.0:
                msg = "Model residual component norm is zero, cannot find epsilon scale! Provide a different initial model"
                if (logger): logger.addToLog(msg)
                raise ValueError(msg)
            # Resetting user-predefined epsilon if any
            self.epsilon = epsilon
            epsilon_balance = res_data_norm / res_model_norm
            # Resetting problem
            self.setDefaults()
            msg = "	Epsilon balancing the the two objective function terms is: %s" % (epsilon_balance)
            if verbose: print(msg)
            if logger: logger.addToLog(msg + "\nREGULARIZED PROBLEM end log file")
        elif self.h_op_reg is not None:
            # Setting non-linear component of the model
            self.h_op.set_nl(self.model)
            self.h_op_reg.set_nl(self.model)
            # Problem is linearly regularized (fixing non-linear part and evaluating the epsilon on the linear
            # component)
        return self.vp_linear_prob.estimate_epsilon(verbose, logger)
    
    def res_func(self, model):
        """Method to return residual vector"""
        # Zero-out residual vector
        self.res.zero()
        ###########################################
        # Applying full non-linear modeling operator
        res = self.res
        if self.epsilon is not None: res = self.res.vecs[0]
        # Computing non-linear part g(m) (if any)
        if self.g_op is not None:
            self.g_op.nl_op.forward(False, model, res)
        # Computing non-linear part g_reg(m) (if any)
        if self.g_op_reg is not None:
            self.g_op_reg.nl_op.forward(False, model, self.res.vecs[1])
        
        ##################################
        # Setting data for linear inversion
        # data term = data - [g(m) if any]
        res.scaleAdd(self.data, -1.0, 1.0)
        # Setting data within first term
        self.vp_linear_prob.data = res
        
        # regularization data term = [g_reg(m) - data_reg if any]
        if self.data_reg is not None:
            self.res.vecs[1].scaleAdd(self.data_reg, 1.0, -1.0)
        # Data term for linear regularization term
        if "epsilon" in dir(self.vp_linear_prob):
            self.res.vecs[1].scale(-1.0)
            self.vp_linear_prob.prior_model = self.res.vecs[1]
        
        ##################################
        # Running linear inversion
        # Getting fevals for saving linear inversion results
        fevals = self.get_fevals()
        # Setting initial linear inversion model
        if not self.warm_start:
            self.lin_model.zero()
        self.vp_linear_prob.set_model(self.lin_model)
        # Setting non-linear component of the model
        self.h_op.set_nl(model)
        if self.h_op_reg is not None:
            self.h_op_reg.set_nl(model)
        # Resetting inversion problem variables
        self.vp_linear_prob.setDefaults()
        # Saving linear inversion results if requested
        if self.lin_solver_prefix is not None:
            self.lin_solver.setPrefix(self.lin_solver_prefix + "_feval%s" % fevals)
        
        # Printing non-linear inversion information
        if self.lin_solver.logger is not None:
            # Writing linear inversion log information if requested (i.e., a logger is present in the solver)
            msg = "NON_LINEAR INVERSION INFO:\n	objective function evaluation\n"
            msg += "#########################################################################################\n"
            self.lin_solver.logger.addToLog(msg + "Linear inversion for non-linear function evaluation # %s" % (fevals))
        self.lin_solver.run(self.vp_linear_prob, verbose=False)
        if self.lin_solver.logger is not None:
            self.lin_solver.logger.addToLog(
                "#########################################################################################\n")
        # Copying inverted linear optimal model
        self.lin_model.copy(self.vp_linear_prob.get_model())
        # Flushing internal saved results of the linear inversion
        self.lin_solver.flush_results()
        
        ##################################
        # Obtaining the residuals
        if (self.epsilon is not None) and not ("epsilon" in dir(self.vp_linear_prob)):
            # Regularization contains a non-linear operator only
            self.res.vecs[0].copy(self.vp_linear_prob.get_res(self.lin_model))
            self.res.vecs[1].scale(self.epsilon)
        else:
            self.res.copy(self.vp_linear_prob.get_res(self.lin_model))
        return self.res
    
    def grad_func(self, model, residual):
        r"""
        Method to return gradient vector
    
        .. math::
           \mathbf{g} = [\mathbf{G}(\mathbf{m})' + \mathbf{H}(\mathbf{m}_nl, \; \mathbf{m}_lin)'] \mathbf{r}_d +
           \varepsilon [\mathbf{G}'(\mathbf{m}_nl)' + \mathbf{H}'(\mathbf{m}_nl, \; \mathbf{m}_lin)'] \mathbf{r}_m
        """
        # Zero-out gradient vector
        self.grad.zero()
        # Setting the optimal linear model component and background of the Jacobian matrices
        self.h_op.set_lin_jac(self.lin_model)  # H(_,m_lin_opt)
        self.h_op.h_nl.set_background(model)  # H(m_nl,m_lin_opt)
        if self.h_op_reg is not None:
            self.h_op_reg.set_lin_jac(self.lin_model)  # H'(_,m_lin_opt)
            self.h_op_reg.h_nl.set_background(model)  # H'(m_nl,m_lin_opt)
        if self.g_op is not None:
            self.g_op.set_background(model)  # G(m_nl)
        if self.g_op_reg is not None:
            self.g_op_reg.set_background(model)  # G'(m_nl)
        # Computing contribuition from the regularization term (if any)
        if self.epsilon is not None:
            # G'(m_nl)' r_m
            if self.g_op_reg is not None:
                self.g_op_reg.lin_op.adjoint(False, self.grad, residual.vecs[1])
            # H'(m_nl,m_lin_opt)' r_m
            if self.h_op_reg is not None:
                self.h_op_reg.h_nl.lin_op.adjoint(True, self.grad, residual.vecs[1])
            # epsilon * [G'(m_nl)' + H'(m_nl,m_lin_opt)'] r_m
            self.grad.scale(self.epsilon)
        residual = self.res if self.epsilon is None else self.res.vecs[0]
        # G(m_nl)' r_d
        if self.g_op is not None:
            self.g_op.lin_op.adjoint(True, self.grad, residual)
        # H(m_nl,m_lin_opt)' r_d
        self.h_op.h_nl.lin_op.adjoint(True, self.grad, residual)
        if self.lin_solver.logger is not None:
            self.lin_solver.logger.addToLog(
                "NON_LINEAR INVERSION INFO:\n	Gradient has been evaluated, current objective function value: %s;\n 	"
                "Stepping!" % (
                    self.get_obj(model)))
        return self.grad
    
    def pert_res_func(self, model, pert_model):
        raise NotImplementedError(
            "ERROR! pert_res_func is not currently supported! Provide an initial step-length value different than zero.")
    
    def obj_func(self, residual):
        r"""
        Method to return objective function value
        
        .. math::
            \phi(\mathbf{m}) = \frac{1}{2} \Vert g(\mathbf{m}_nl) + h(\mathbf{m}_nl)\mathbf{m}_lin - \mathbf{d}\Vert_2^2
            + \frac{\varepsilon^2}{2} \Vert g'(\mathbf{m}_nl) + h'(\mathbf{m}_nl)\mathbf{m}_lin - \mathbf{d}'\Vert_2^2
        """
        if "obj_terms" in dir(self):
            # data term
            val = residual.vecs[0].norm()
            self.obj_terms[0] = 0.5 * val * val
            # model term
            val = residual.vecs[1].norm()
            self.obj_terms[1] = 0.5 * val * val
            obj = self.obj_terms[0] + self.obj_terms[1]
        else:
            val = residual.norm()
            obj = 0.5 * val * val
        return obj
