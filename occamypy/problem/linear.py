from math import isnan

from .base import Problem
from occamypy import operator as O
from occamypy.vector import superVector


class LeastSquares(Problem):
    """Linear inverse problem of the form 1/2*|Lm-d|_2"""

    def __init__(self, model, data, op, grad_mask=None, prec=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
           Constructor of linear problem:
           model    	= [no default] - vector class; Initial model vector
           data     	= [no default] - vector class; Data vector
           op       	= [no default] - linear operator class; L operator
           grad_mask	= [None] - vector class; Mask to be applied on the gradient during the inversion
           minBound     = [None] - vector class; Minimum value bounds
           maxBound     = [None] - vector class; Maximum value bounds
           boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
           prec       	= [None] - linear operator class; Preconditioning matrix
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
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Preconditioning matrix
        self.prec = prec
        # Setting default variables
        self.setDefaults()
        self.linear = True
        return

    def __del__(self):
        """Default destructor"""
        return

    def resf(self, model):
        """Method to return residual vector r = Lm - d"""
        # Computing Lm
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()
        # Computing Lm - d
        self.res.scaleAdd(self.data, 1., -1.)
        return self.res

    def gradf(self, model, res):
        """Method to return gradient vector g = L'r = L'(Lm - d)"""
        # Computing L'r = g
        self.op.adjoint(False, self.grad, res)
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad

    def dresf(self, model, dmodel):
        """Method to return residual vector dres = Ldm"""
        # Computing Ldm = dres
        self.op.forward(False, dmodel, self.dres)
        return self.dres

    def objf(self, residual):
        """Method to return objective function value 1/2|Lm-d|_2"""
        val = residual.norm()
        obj = 0.5 * val * val
        return obj


class LeastSquaresSymmetric(Problem):
    """Linear inverse problem of the form 1/2m'Am - m'b"""

    def __init__(self, model, data, op, prec=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Constructor of linear symmetric problem:
        model    	= [no default] - vector class; Initial model vector
        data     	= [no default] - vector class; Data vector
        op       	= [no default] - linear operator class; A symmetric operator (i.e., A = A')
        minBound		= [None] - vector class; Minimum value bounds
        maxBound		= [None] - vector class; Maximum value bounds
        boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
        prec       	= [None] - linear operator class; Preconditioning matrix
        """
        # Setting the bounds (if any)
        super(LeastSquaresSymmetric, self).__init__(minBound, maxBound, boundProj)
        # Checking range and domain are the same
        if not model.checkSame(data) and not op.domain.checkSame(op.range):
            raise ValueError("Data and model vector live in different spaces!")
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

    def __del__(self):
        """Default destructor"""
        return

    def resf(self, model):
        """Method to return residual vector r = Am - b"""
        # Computing Lm
        if model.norm() != 0.:
            self.op.forward(False, model, self.res)
        else:
            self.res.zero()
        # Computing Lm - d
        self.res.scaleAdd(self.data, 1., -1.)
        return self.res

    def gradf(self, model, res):
        """Method to return gradient vector equal to residual one"""
        # Assigning g = r
        self.grad = self.res
        return self.grad

    def dresf(self, model, dmodel):
        """Method to return residual vector dres = Adm"""
        # Computing Ldm = dres
        self.op.forward(False, dmodel, self.dres)
        return self.dres

    def objf(self, residual):
        """Method to return objective function value 1/2m'Am - m'b"""
        obj = 0.5 * (self.model.dot(residual) - self.model.dot(self.data))
        return obj


class LeastSquaresRegularized(Problem):
    """Linear inverse problem regularized of the form 1/2*|Lm-d|_2 + epsilon^2/2*|Am-m_prior|_2"""

    def __init__(self, model, data, op, epsilon, grad_mask=None, reg_op=None, prior_model=None, prec=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Constructor of linear regularized problem:
        model    	= [no default] - vector class; Initial model vector
        data     	= [no default] - vector class; Data vector
        op       	= [no default] - linear operator class; L operator
        epsilon      = [no default] - float; regularization weight
        grad_mask	= [None] - vector class; Mask to be applied on the gradient during the inversion
        reg_op       = [Identity] - linear operator class; A regularization operator
        prior_model  = [None] - vector class; Prior model for regularization term
        minBound		= [None] - vector class; Minimum value bounds
        maxBound		= [None] - vector class; Maximum value bounds
        boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
        prec       	= [None] - linear operator class; Preconditioning matrix
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
        # Setting a prior model (if any)
        self.prior_model = prior_model
        # Setting linear operators
        # Assuming identity operator if regularization operator was not provided
        if reg_op is None:
            reg_op = O.Identity(self.model)
        # Checking if space of the prior model is consistent with range of
        # regularization operator
        if self.prior_model is not None:
            if not self.prior_model.checkSame(reg_op.range):
                raise ValueError("Prior model space no consistent with range of regularization operator")
        self.op = O.Vstack(op, reg_op)  # Modeling operator
        self.epsilon = epsilon  # Regularization weight
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Residual vector (data and model residual vectors)
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

    def __del__(self):
        """Default destructor"""
        return
    
    # TODO make estimate_epsilon a function that takes a problem as input
    def estimate_epsilon(self, verbose=False, logger=None):
        """
        Method returning epsilon that balances the first gradient in the 'extended-data' space or initial data residuals
        """
        msg = "Epsilon Scale evaluation"
        if verbose:
            print(msg)
        if logger:
            logger.addToLog("REGULARIZED PROBLEM log file\n" + msg)
        # Keeping the initial model vector
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
                prblm_res.vecs[1].scaleAdd(self.prior_model)  # Remove prior model vector (Ag0 - m_prior + m_prior)
            msg = "	Epsilon balancing the data-space gradients is: %.2e"
        res_data_norm = prblm_res.vecs[0].norm()
        res_model_norm = prblm_res.vecs[1].norm()
        if isnan(res_model_norm) or isnan(res_data_norm):
            raise ValueError("Obtained NaN: Residual-data-side-norm = %.2e, Residual-model-side-norm = %.2e"
                             % (res_data_norm, res_model_norm))
        if res_model_norm == 0.:
            raise ValueError("Model residual component norm is zero, cannot find epsilon scale")
        # Resetting user-predefined epsilon if any
        self.epsilon = epsilon
        # Resetting problem initial model vector
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
        """Method to return residual vector r = [r_d; r_m]: r_d = Lm - d; r_m = epsilon * (Am - m_prior) """
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
        """Method to return gradient vector g = L'r_d + epsilon*A'r_m"""
        # Scaling by epsilon the model residual vector (saving temporarily residual regularization)
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
        """Method to return residual vector dres = (L + epsilon * A)dm"""
        # Computing Ldm = dres_d
        self.op.forward(False, dmodel, self.dres)
        # Scaling by epsilon
        self.dres.vecs[1].scale(self.epsilon)
        return self.dres

    def objf(self, residual):
        """Method to return objective function value 1/2|Lm-d|_2 + epsilon^2/2*|Am-m_prior|_2"""
        for idx in range(residual.n):
            val = residual.vecs[idx].norm()
            self.obj_terms[idx] = 0.5 * val*val
        return sum(self.obj_terms)


class Lasso(Problem):
    """Convex problem 1/2*| y - Am |_2 + lambda*| m |_1"""

    def __init__(self, model, data, op, op_norm=None, lambda_value=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
           Constructor of convex L1-norm LASSO inversion problem:
           model    	= [no default] - vector class; Initial model vector
           data     	= [no default] - vector class; Data vector
           op       	= [no default] - linear operator class; L operator
           lambda_value	= [None] - Regularization weight. Not necessary for ISTC solver but required for ISTA and FISTA
           op_norm		= [None] - float; A operator norm that will be evaluated with the power method if not provided
           minBound		= [None] - vector class; Minimum value bounds
           maxBound		= [None] - vector class; Maximum value bounds
           boundProj	= [None] - Bounds class; Class with a function "apply(input_vec)" to project input_vec onto some convex set
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
        # Residual vector (data and model residual vectors)
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
        """Method to return objective function value 1/2*| y - Am |_2 + lambda*| m |_1"""
        # data term
        val = residual.vecs[0].norm()
        self.obj_terms[0] = 0.5 * val*val
        # model term
        self.obj_terms[1] = self.lambda_value * residual.vecs[1].norm(1)
        return sum(self.obj_terms)

    # define function that computes residuals
    def resf(self, model):
        """ y - alpha * A m = rd (self.res[0]) and m = rm (self.res[1]);"""
        if model.norm() != 0.:
            self.op.forward(False, model, self.res.vecs[0])
        else:
            self.res.zero()
        # Computing r_d = Lm - d
        self.res.vecs[0].scaleAdd(self.data, -1., 1.)
        # Run regularization part
        self.res.vecs[1].copy(model)
        return self.res

    # function that projects search direction into data space (Not necessary for ISTC)
    def dresf(self, model, dmodel):
        """Linear projection of the model perturbation onto the data space. Method not implemented"""
        raise NotImplementedError("dresf is not necessary for ISTC; DO NOT CALL THIS METHOD")

    # function to compute gradient (Soft thresholding applied outside in the solver)
    def gradf(self, model, res):
        """- A'r_data (residual[0]) = g"""
        # Apply an adjoint modeling
        self.op.adjoint(False, self.grad, res.vecs[0])
        # Applying negative scaling
        self.grad.scale(-1.0)
        return self.grad


class GeneralizedLasso(Problem):
    def __init__(self, model, data, op, eps=1., reg=None,
                 minBound=None, maxBound=None, boundProj=None):
        """
        Linear Problem with L1 regularizer:

        .. math ::
            1 / 2 |Op m - d|_2^2 + eps |R m|_1

        :param model        : vector; initial model
        :param data         : vector; data
        :param op           : LinearOperator; modeling operator
        :param eps          : float; weight of L1 regularizer [1.]
        :param reg          : LinearOperator; L1 regularizer [Identity]
        :param minBound     : vector; minimum value bounds
        :param maxBound     : vector; maximum value bounds
        :param boundProj    : Bounds; object with a method "apply(x)" to project x onto some convex set
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
        self.reg_op = reg if reg is not None else O.Identity(model)
        self.eps = eps

        # Last settings
        self.obj_terms = [None] * 2
        self.linear = True
        # store the "residuals" (for computing the objective function)
        self.res_data = self.op.range.clone().zero()
        self.res_reg = self.reg_op.range.clone().zero()
        # this last superVector is instantiated with pointers to res_data and res_reg!
        self.res = superVector(self.res_data, self.res_reg)

    def __del__(self):
        """Default destructor"""
        return

    def objf(self, residual, eps=None):
        """
        Compute objective function based on the residual (super)vector

        .. math ::
            1 / 2 |Op m - d|_2^2 + eps |R m|_1

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
        """Compute residuals from current model"""

        # compute data residual: Op * m - d
        if model.norm() != 0:
            self.op.forward(False, model, self.res_data)  # rd = Op * m
        else:
            self.res_data.zero()
        self.res_data.scaleAdd(self.data, 1., -1.)  # rd = rd - d

        # compute L1 reg residuals
        if model.norm() != 0. and self.reg_op is not None:
            self.reg_op.forward(False, model, self.res_reg)
        else:
            self.res_reg.zero()

        return self.res
