from math import isnan

from occamypy.operator import Identity, Vstack
from occamypy.problem.base import Problem
from occamypy.vector.base import superVector, Vector


class LeastSquares(Problem):
    r"""
    Linear inverse problem of the form

    .. math::
        \frac{1}{2} \Vert \mathbf{A}\mathbf{m} - \mathbf{d}\Vert_2^2
    """
    
    def __init__(self, model: Vector, data: Vector, op, grad_mask: Vector = None, prec=None,
                 minBound: Vector = None, maxBound: Vector = None, boundProj=None):
        """
        LeastSquares constructor
        
        Args:
            model: initial domain vector
            data: data vector
            op: linear operator
            grad_mask: mask to be applied on the gradient during the inversion
            prec: preconditioner linear operator
            minBound: lower bound vector
            maxBound: upper bound vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(LeastSquares, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Residual vector
        self.res = data.clone()
        self.res.zero()
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting linear operator
        self.op = op
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with domain vector!")
            self.grad_mask = grad_mask.clone()
        # Preconditioning matrix
        self.prec = prec
        # Setting default variables
        self.setDefaults()
        self.linear = True
        return
    
    def __del__(self):
        return
    
    def resf(self, model):
        r"""
        Method to return residual vector
        
        .. math::
            \mathbf{r} = \mathbf{A} \mathbf{m} - \mathbf{d}
        """
        # Computing Am
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()
        # Computing Am - d
        self.res.scaleAdd(self.data, 1., -1.)
        return self.res
    
    def gradf(self, model, res):
        r"""
        Method to return gradient vector
        
        .. math::
            \mathbf{g} = \mathbf{A}'\mathbf{r} = \mathbf{A}'(\mathbf{A} \mathbf{m} - \mathbf{d})
        """
        # Computing A'r = g
        self.op.adjoint(False, self.grad, res)
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad
    
    def dresf(self, model, dmodel):
        r"""
        Method to return residual vector
        
        .. math::
            \mathbf{r}_d = \mathbf{A} \mathbf{r}_m
        """
        self.op.forward(False, dmodel, self.dres)
        return self.dres
    
    def objf(self, residual):
        r"""
        Method to return objective function value
        
        .. math::
            \frac{1}{2} \Vert \mathbf{A}\mathbf{m} - \mathbf{d}\Vert_2^2
        """
        val = residual.norm()
        obj = 0.5 * val * val
        return obj


class LeastSquaresSymmetric(Problem):
    r"""
    Linear inverse problem of the form
    
    .. math::
        \frac{1}{2} \Vert \mathbf{A}\mathbf{m} - \mathbf{d}\Vert_2^2
    
    where A is a symmetric operator (i.e., A' = A)
    """
    
    def __init__(self, model: Vector, data: Vector, op, prec=None,
                 minBound: Vector = None, maxBound: Vector = None, boundProj=None):
        """
        LeastSquaresSymmetric constructor
        
        Args:
            model: initial domain vector
            data: data vector
            op: linear operator
            prec: preconditioner linear operator
            minBound: lower bound vector
            maxBound: upper bound vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(LeastSquaresSymmetric, self).__init__(minBound, maxBound, boundProj)
        # Checking range and domain are the same
        if not model.checkSame(data) and not op.domain.checkSame(op.range):
            raise ValueError("Data and domain vector live in different spaces!")
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Copying the pointer to data vector
        self.data = data
        # Residual vector
        self.res = data.clone()
        self.res.zero()
        # Gradient vector is equal to the residual vector
        self.grad = self.res
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting linear operator
        self.op = op
        # Preconditioning matrix
        self.prec = prec
        # Setting default variables
        self.setDefaults()
        self.linear = True
    
    def resf(self, model):
        r"""
        Method to return residual vector
        
        .. math::
            \mathbf{r} = \mathbf{A} \mathbf{m} - \mathbf{d}
        """
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()

        self.res.scaleAdd(self.data, 1., -1.)
        return self.res
    
    def gradf(self, model, res):
        r"""Method to return gradient vector
        
        .. math::
            \mathbf{g} = \mathbf{r}
        """
        self.grad = self.res
        return self.grad
    
    def dresf(self, model, dmodel):
        r"""
        Method to return residual vector
        
        .. math::
            \mathbf{r}_d = \mathbf{A} \mathbf{r}_m
        """
        # Computing Ldm = dres
        self.op.forward(False, dmodel, self.dres)
        return self.dres
    
    def objf(self, residual):
        r"""
        Method to return objective function value
        
        .. math::
            \frac{1}{2}  [ \mathbf{m}'\mathbf{A}\mathbf{m} - \mathbf{m}'\mathbf{d}]
        """
        obj = 0.5 * (self.model.dot(residual) - self.model.dot(self.data))
        return obj


class LeastSquaresRegularized(Problem):
    r"""
    Linear regularized inverse problem of the form
    
    .. math::
        \frac{1}{2} \Vert \mathbf{A}\mathbf{m} - \mathbf{d}\Vert_2^2 + \frac{\varepsilon^2}{2} \Vert \mathbf{R m} - \mathbf{m}_p \Vert_2^2
    """
    
    def __init__(self, model: Vector, data: Vector, op, epsilon: float, grad_mask: Vector = None, reg_op=None,
                 prior_model: Vector = None, prec=None,
                 minBound: Vector = None, maxBound: Vector = None, boundProj=None):
        """
        LeastSquaresRegularized constructor
        
        Args:
            model: initial domain vector
            data: data vector
            op: linear operator
            epsilon: regularization weight
            grad_mask: mask to be applied on the gradient during the inversion
            reg_op: regularization operator (default: Identity)
            prior_model: prior vector for the regularization term
            prec: preconditioner linear operator
            minBound: lower bound vector
            maxBound: upper bound vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(LeastSquaresRegularized, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Setting a prior domain (if any)
        self.prior_model = prior_model
        # Setting linear operators
        # Assuming identity operator if regularization operator was not provided
        if reg_op is None:
            reg_op = Identity(self.model)
        # Checking if space of the prior domain is consistent with range of
        # regularization operator
        if self.prior_model is not None:
            if not self.prior_model.checkSame(reg_op.range):
                raise ValueError("Prior domain space no consistent with range of regularization operator")
        self.op = Vstack(op, reg_op)  # Modeling operator
        self.epsilon = epsilon  # Regularization weight
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with domain vector!")
            self.grad_mask = grad_mask.clone()
        # Residual vector (data and domain residual vectors)
        self.res = self.op.range.clone()
        self.res.zero()
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = True
        # Preconditioning matrix
        self.prec = prec
        # Objective function terms (useful to analyze each term)
        self.obj_terms = [None, None]
    
    def estimate_epsilon(self, verbose: bool = False, logger=None) -> float:
        """
        Estimate the epsilon that balances the first gradient in the 'extended-data' space or initial data residuals
        
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
        # Keeping the initial domain vector
        prblm_mdl = self.get_model()
        mdl_tmp = prblm_mdl.clone()
        # Keeping user-predefined epsilon if any
        epsilon = self.epsilon
        # Setting epsilon to one to evaluate the scale
        self.epsilon = 1.0
        if self.model.norm() != 0.:
            prblm_res = self.get_res(self.model)
            msg = "	Epsilon balancing data and regularization residuals is: %.2e"
        else:
            prblm_grad = self.get_grad(self.model)  # Compute first gradient
            prblm_res = self.get_res(prblm_grad)  # Compute residual arising from the gradient
            # Balancing the first gradient in the 'extended-data' space
            prblm_res.vecs[0].scaleAdd(self.data)  # Remove data vector (Lg0 - d + d)
            if self.prior_model is not None:
                prblm_res.vecs[1].scaleAdd(self.prior_model)  # Remove prior domain vector (Ag0 - m_prior + m_prior)
            msg = "	Epsilon balancing the data-space gradients is: %.2e"
        res_data_norm = prblm_res.vecs[0].norm()
        res_model_norm = prblm_res.vecs[1].norm()
        if isnan(res_model_norm) or isnan(res_data_norm):
            raise ValueError("Obtained NaN: Residual-data-side-norm = %.2e, Residual-domain-side-norm = %.2e"
                             % (res_data_norm, res_model_norm))
        if res_model_norm == 0.:
            raise ValueError("Model residual component norm is zero, cannot find epsilon scale")
        # Resetting user-predefined epsilon if any
        self.epsilon = epsilon
        # Resetting problem initial domain vector
        self.set_model(mdl_tmp)
        del mdl_tmp
        epsilon_balance = res_data_norm / res_model_norm
        # Resetting feval
        self.fevals = 0
        msg = msg % epsilon_balance
        if verbose:
            print(msg)
        if logger:
            logger.addToLog(msg + "\nREGULARIZED PROBLEM end log file")
        return epsilon_balance
    
    def resf(self, model):
        r"""
        Method to return residual vector

        .. math::
            \begin{bmatrix}
                \mathbf{r}_{d}  \\
                \mathbf{r}_{m}  \\
            \end{bmatrix} =
            \begin{bmatrix}
                \mathbf{A}\mathbf{m} - \mathbf{d}  \\
                \varepsilon (\mathbf{R} \mathbf{m} - \mathbf{m}_p)  \\
            \end{bmatrix}
        """
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()
        # Computing r_d = Lm - d
        self.res.vecs[0].scaleAdd(self.data, 1., -1.)
        # Computing r_m = Am - m_prior
        if self.prior_model is not None:
            self.res.vecs[1].scaleAdd(self.prior_model, 1., -1.)
        # Scaling by epsilon epsilon*r_m
        self.res.vecs[1].scale(self.epsilon)
        return self.res
    
    def gradf(self, model, res):
        r"""
        Method to return gradient vector

        .. math::
            \mathbf{g} = \mathbf{A}' \mathbf{r}_d + \varepsilon \mathbf{R}' \mathbf{r}_m
        """
        # Scaling by epsilon the domain residual vector (saving temporarily residual regularization)
        # g = epsilon*A'r_m
        self.op.ops[1].adjoint(False, self.grad, res.vecs[1])
        self.grad.scale(self.epsilon)
        # g = L'r_d + epsilon*A'r_m
        self.op.ops[0].adjoint(True, self.grad, res.vecs[0])
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad
    
    def dresf(self, model, dmodel):
        r"""
        Method to return residual vector
        
        .. math::
             \mathbf{d}_r = [\mathbf{A} + \varepsilon \mathbf{R}] \mathbf{d}_m
        """
        self.op.forward(False, dmodel, self.dres)
        # Scaling by epsilon
        self.dres.vecs[1].scale(self.epsilon)
        return self.dres
    
    def objf(self, residual):
        r"""
        Method to return objective function value
        
        .. math::
            \frac{1}{2} \Vert \mathbf{r}_m \Vert_2^2 + \frac{1}{2} \Vert \mathbf{r}_m \Vert_2^2
        """
        for idx in range(residual.n):
            val = residual.vecs[idx].norm()
            self.obj_terms[idx] = 0.5 * val * val
        return sum(self.obj_terms)


class Lasso(Problem):
    r"""
    Least Absolute Shrinkage and Selection Operator (LASSO) problem

    .. math::
        \frac{1}{2} \Vert \mathbf{A}\mathbf{m} - \mathbf{d}\Vert_2^2 + \lambda \Vert \mathbf{m}\Vert_1
    """
    
    def __init__(self, model: Vector, data: Vector, op, op_norm: float = None, lambda_value: float = None,
                 minBound: Vector = None, maxBound: Vector = None, boundProj=None):
        """
        Lasso constructor
        
        Args:
            model: initial domain vector
            data: data vector
            op: linear operator
            op_norm: operator norm that will be computed with the power method if not provided
            lambda_value: regularization weight
            minBound: lower bound vector
            maxBound: upper bound vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        # Setting the bounds (if any)
        super(Lasso, self).__init__(minBound, maxBound, boundProj)
        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Setting linear operator
        self.op = op  # Modeling operator
        # Residual vector (data and domain residual vectors)
        self.res = superVector(op.range.clone(), op.domain.clone())
        self.res.zero()
        # Dresidual vector
        self.dres = None  # Not necessary for the inversion
        # Setting default variables
        self.setDefaults()
        self.linear = True
        if op_norm is not None:
            # Using user-provided A operator norm
            self.op_norm = op_norm  # Operator Norm necessary for solver
        else:
            # Evaluating operator norm using power method
            self.op_norm = self.op.powerMethod()
        self.lambda_value = lambda_value
        # Objective function terms (useful to analyze each term)
        self.obj_terms = [None, None]
        return
    
    def set_lambda(self, lambda_in):
        # Set lambda
        self.lambda_value = lambda_in
        return
    
    def objf(self, residual):
        r"""
        Method to return objective function value
        
        .. math::
            \frac{1}{2} \Vert \mathbf{A}\mathbf{m} - \mathbf{d}\Vert_2^2 + \lambda \Vert \mathbf{m}\Vert_1
        """
        # data term
        val = residual.vecs[0].norm()
        self.obj_terms[0] = 0.5 * val * val
        # domain term
        self.obj_terms[1] = self.lambda_value * residual.vecs[1].norm(1)
        return sum(self.obj_terms)
    
    def resf(self, model):
        r"""
        Compute the residuals from the model
        
        .. math::
            \begin{bmatrix}
                \mathbf{r}_{d}  \\
                \mathbf{r}_{m}  \\
            \end{bmatrix} =
            \begin{bmatrix}
                \mathbf{A}\mathbf{m} - \mathbf{d}  \\
                \mathbf{m} \\
            \end{bmatrix}
        """
        if model.norm() != 0.:
            self.op.forward(False, model, self.res.vecs[0])
        else:
            self.res.zero()
        # Computing r_d = Lm - d
        self.res.vecs[0].scaleAdd(self.data, -1., 1.)
        # Run regularization part
        self.res.vecs[1].copy(model)
        return self.res
    
    def dresf(self, model, dmodel):
        """Linear projection of the domain perturbation onto the data space. Method not implemented"""
        raise NotImplementedError("dresf is not necessary for ISTC; DO NOT CALL THIS METHOD")
    
    # function to compute gradient (Soft thresholding applied outside in the solver)
    def gradf(self, model, res):
        r"""Compute the gradient (the soft-thresholding is applied by the solver!)
        
        .. math::
            \mathbf{g} = - \mathbf{A}'\mathbf{r}
        """
        # Apply an adjoint modeling
        self.op.adjoint(False, self.grad, res.vecs[0])
        # Applying negative scaling
        self.grad.scale(-1.0)
        return self.grad


class GeneralizedLasso(Problem):
    r"""
     Linear L1-regularized inverse problem of the form

    .. math::
        \frac{1}{2} \Vert \mathbf{A}\mathbf{m} - \mathbf{d}\Vert_2^2 + \varepsilon \Vert \mathbf{Rm}\Vert_1
    """
    
    def __init__(self, model: Vector, data: Vector, op, eps: float = 1., reg=None,
                 minBound: Vector = None, maxBound: Vector = None, boundProj=None):
        """
        GeneralizedLasso constructor
        
        Args:
            model: initial domain vector
            data: data vector
            op: linear operator
            eps: regularization weight
            reg: regularizer operator (default: Identity)
            minBound: lower bound vector
            maxBound: upper bound vector
            boundProj: class with a method `apply(input_vec)` to project input_vec onto some convex set
        """
        super(GeneralizedLasso, self).__init__(minBound, maxBound, boundProj)
        self.model = model
        self.dmodel = model.clone().zero()
        self.grad = self.dmodel.clone()
        self.data = data
        self.op = op
        
        self.minBound = minBound
        self.maxBound = maxBound
        self.boundProj = boundProj
        
        # L1 Regularization
        self.reg_op = reg if reg is not None else Identity(model)
        self.eps = eps
        
        # Last settings
        self.obj_terms = [None] * 2
        self.linear = True
        # store the "residuals" (for computing the objective function)
        self.res_data = self.op.range.clone().zero()
        self.res_reg = self.reg_op.range.clone().zero()
        # this last superVector is instantiated with pointers to res_data and res_reg!
        self.res = superVector(self.res_data, self.res_reg)
    
    def objf(self, residual, eps=None):
        r"""
        Compute objective function based on the residual (super)vector

        .. math::
            \frac{1}{2} \Vert \mathbf{r}_d\Vert_2^2 + \varepsilon \Vert \mathbf{r}_m\Vert_1
        """
        res_data = residual.vecs[0]
        res_reg = residual.vecs[1]
        eps = eps if eps is not None else self.eps
        
        # data fidelity
        self.obj_terms[0] = .5 * res_data.norm(2) ** 2
        
        # regularization penalty
        self.obj_terms[1] = eps * res_reg.norm(1)
        
        return sum(self.obj_terms)
    
    def resf(self, model):
        r"""
        Compute residuals from model
        
        .. math::
            \begin{bmatrix}
                \mathbf{r}_{d}  \\
                \mathbf{r}_{m}  \\
            \end{bmatrix} =
            \begin{bmatrix}
                \mathbf{A}\mathbf{m} - \mathbf{d}  \\
                \mathbf{R}\mathbf{m} \\
            \end{bmatrix}
        """
        # compute data residual: A m - d
        if model.norm() != 0:
            self.op.forward(False, model, self.res_data)  # rd = A * m
        else:
            self.res_data.zero()
        self.res_data.scaleAdd(self.data, 1., -1.)  # rd = rd - d
        
        # compute L1 reg residuals
        if model.norm() != 0. and self.reg_op is not None:
            self.reg_op.forward(False, model, self.res_reg)
        else:
            self.res_reg.zero()
        
        return self.res
