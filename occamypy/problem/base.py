class Bounds:
    """
    Class used to enforce boundary constraints during the inversion

    Methods:
        apply(in_content): apply bounds to input vector
    """
    
    def __init__(self, minBound=None, maxBound=None):
        """
        Bounds constructor
        
        Args:
            minBound: vector containing minimum values of the domain vector
            maxBound: vector containing maximum values of the domain vector
        """
        self.minBound = minBound
        self.maxBound = maxBound
        if minBound is not None:
            self.minBound = minBound.clone()
        if maxBound is not None:
            self.maxBound = maxBound.clone()
        # If only the lower bound was provided we use the opposite of the lower bound to clip the values
        if self.minBound is not None and self.maxBound is None:
            self.minBound.scale(-1.0)
        return

    def apply(self, in_content):
        """
        Apply bounds to the input vector
        
        Args:
            in_content: vector to be processed
        """
        if self.minBound is not None and self.maxBound is None:
            if not in_content.checkSame(self.minBound):
                raise ValueError("Input vector not consistent with bound space")
            in_content.scale(-1.0)
            in_content.clip(in_content, self.minBound)
            in_content.scale(-1.0)
        elif self.minBound is None and self.maxBound is not None:
            if not in_content.checkSame(self.maxBound):
                raise ValueError("Input vector not consistent with bound space")
            in_content.clip(in_content, self.maxBound)
        elif self.minBound is not None and self.maxBound is not None:
            if (not (in_content.checkSame(self.minBound) and in_content.checkSame(
                    self.maxBound))):
                raise ValueError("Input vector not consistent with bound space")
            in_content.clip(self.minBound, self.maxBound)
        return


class Problem:
    """Base problem class"""

    def __init__(self, minBound=None, maxBound=None, boundProj=None):
        """
        Problem constructor
        
        Args:
            minBound: vector containing minimum values of the domain vector
            maxBound: vector containing maximum values of the domain vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
        """
        if minBound is not None or maxBound is not None:
            # Simple box bounds
            self.bounds = Bounds(minBound, maxBound)  # Setting the bounds of the problem (if necessary)
        elif boundProj is not None:
            # Projection operator onto the bounds
            self.bounds = boundProj
        # Setting common variables
        self.obj_updated = False
        self.res_updated = False
        self.grad_updated = False
        self.dres_updated = False
        self.fevals = 0
        self.gevals = 0
        self.counter = 0
        self.linear = False  # By default all problem are non-linear

    def __del__(self):
        return

    def setDefaults(self):
        """Default common variables for any inverse problem"""
        self.obj_updated = False
        self.res_updated = False
        self.grad_updated = False
        self.dres_updated = False
        self.fevals = 0
        self.gevals = 0
        self.counter = 0
        return

    def set_model(self, in_content):
        """Setting internal domain vector
        
        Args:
            in_content: domain vector to be copied
        """
        if in_content.isDifferent(self.model):
            self.model.copy(in_content)
            self.obj_updated = False
            self.res_updated = False
            self.grad_updated = False
            self.dres_updated = False

    def set_residual(self, in_content):
        """Setting internal residual vector
        
        Args:
            in_content: residual vector to be copied
        """
        # Useful for linear inversion (to avoid residual computation)
        if self.res.isDifferent(in_content):
            self.res.copy(in_content)
            # If residuals have changed, recompute gradient and objective function value
            self.grad_updated = False
            self.obj_updated = False
        self.res_updated = True
        return

    def get_model(self):
        """Get the domain vector"""
        return self.model

    def get_dmodel(self):
        """Get the domain vector"""
        return self.dmodel

    def get_rnorm(self, model) -> float:
        """Compute the residual vector norm
        Args:
            model: domain vector
        """
        self.get_res(model)
        return self.get_res(model).norm()

    def get_gnorm(self, model) -> float:
        """Compute the gradient vector norm
        
        Args:
            model: domain vector
        """
        return self.get_grad(model).norm()

    def get_obj(self, model) -> float:
        """Compute the objective function
        
        Args:
            model: domain vector
        """
        self.set_model(model)
        if not self.obj_updated:
            self.res = self.get_res(self.model)
            self.obj = self.objf(self.res)
            self.obj_updated = True
        return self.obj

    def get_res(self, model):
        """Compute the residual vector
        
        Args:
            model: domain vector
        """
        self.set_model(model)
        if not self.res_updated:
            self.fevals += 1
            self.res = self.resf(self.model)
            self.res_updated = True
        return self.res

    def get_grad(self, model):
        """Compute the gradient vector
        
        Args:
            model: domain vector
        """
        self.set_model(model)
        if not self.grad_updated:
            self.res = self.get_res(self.model)
            self.grad = self.gradf(self.model, self.res)
            self.gevals += 1
            if self.linear:
                self.fevals += 1
            self.grad_updated = True
        return self.grad

    def get_dres(self, model, dmodel):
        """Compute the dresidual vector (i.e., application of the Jacobian to Dmodel vector)
        
        Args:
            model: domain vector
            dmodel: dmodel vector
        """
        self.set_model(model)
        if not self.dres_updated or dmodel.isDifferent(self.dmodel):
            self.dmodel.copy(dmodel)
            self.dres = self.dresf(self.model, self.dmodel)
            if self.linear:
                self.fevals += 1
            self.dres_updated = True
        return self.dres

    def get_fevals(self):
        """Get the number of objective function evalutions"""
        return self.fevals

    def get_gevals(self):
        """Get the number of gradient evalutions"""
        return self.gevals

    def objf(self, residual) -> float:
        """Compute the objective function
        
        Args:
            residual: residual vector
        Returns:
            objective function value
        """
        raise NotImplementedError("Implement objf for problem in the derived class!")

    def resf(self, model):
        """
        Compute the residual vector
        
        Args:
            model: domain vector

        Returns: residual vector based on the domain
        """
        raise NotImplementedError("Implement resf for problem in the derived class!")

    def dresf(self, model, dmodel):
        """
        Compute the residual vector
        Args:
            model: domain vector
            dmodel: dmodel vector

        Returns: residual vector
        """
        raise NotImplementedError("Implement dresf for problem in the derived class!")

    def gradf(self, model, residual):
        """
        Compute the gradient vector from the residual (i.e., g = A' r = A'(Am - d))
        Args:
            model: domain vector
            residual: residual vector

        Returns: gradient vector
        """
        raise NotImplementedError("Implement gradf for problem in the derived class!")
