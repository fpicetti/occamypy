__all__ = [
    "Bounds",
    "Problem",
]


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

    def __init__(self, model, data, minBound=None, maxBound=None, boundProj=None, name: str = "Problem"):
        """
        Problem constructor

        Args:
            model: model vector to be optimized
            data: data vector
            minBound: vector containing minimum values of the domain vector
            maxBound: vector containing maximum values of the domain vector
            boundProj: class with a function "apply(input_vec)" to project input_vec onto some convex set
            name: problem name
        """
        self.minBound = minBound
        self.maxBound = maxBound
        self.boundProj = boundProj
        
        if minBound is not None or maxBound is not None:
            # Simple box bounds
            self.bounds = Bounds(minBound, maxBound)  # Setting the bounds of the problem (if necessary)
        elif boundProj is not None:
            # Projection operator onto the bounds
            self.bounds = boundProj
        
        # use model and data as pointers, not clone
        self.model = model
        self.data = data
        
        # Setting common variables
        self.pert_model = model.clone().zero()
        self.res = data.clone().zero()
        self.pert_res = self.res.clone()
        
        self.obj_updated = False
        self.res_updated = False
        self.grad_updated = False
        self.pert_res_updated = False
        self.fevals = 0
        self.gevals = 0
        self.counter = 0
        self.linear = False
        self.name = str(name)
    
    def __str__(self):
        return self.name
    
    def setDefaults(self):
        """Default common variables for any inverse problem"""
        self.obj_updated = False
        self.res_updated = False
        self.grad_updated = False
        self.pert_res_updated = False
        self.fevals = 0
        self.gevals = 0
        self.counter = 0
        return

    def set_model(self, model):
        """
        Setting internal domain vector

        Args:
            model: domain vector to be copied
        """
        if model.isDifferent(self.model):
            self.model.copy(model)
            self.obj_updated = False
            self.res_updated = False
            self.grad_updated = False
            self.pert_res_updated = False

    def set_residual(self, residual):
        """
        Setting internal residual vector

        Args:
            residual: residual vector to be copied
        """
        # Useful for linear inversion (to avoid residual computation)
        if self.res.isDifferent(residual):
            self.res.copy(residual)
            # If residuals have changed, recompute gradient and objective function value
            self.grad_updated = False
            self.obj_updated = False
        self.res_updated = True
        return

    def get_model(self):
        """Geet the domain vector"""
        return self.model

    def get_pert_model(self):
        """Get the model perturbation vector"""
        return self.pert_model

    def get_rnorm(self, model):
        """
        Compute the residual vector norm
        
        Args:
            model: domain vector
        """
        self.get_res(model)
        return self.get_res(model).norm()

    def get_gnorm(self, model):
        """
        Compute the gradient vector norm

        Args:
            model: domain vector
        """
        return self.get_grad(model).norm()

    def get_obj(self, model):
        """Compute the objective function
        
        Args:
            model: domain vector
        """
        self.set_model(model)
        if not self.obj_updated:
            self.res = self.get_res(self.model)
            self.obj = self.obj_func(self.res)
            self.obj_updated = True
        return self.obj

    def get_res(self, model):
        """
        Compute the residual vector

        Args:
            model: domain vector
        """
        self.set_model(model)
        if not self.res_updated:
            self.fevals += 1
            self.res = self.res_func(self.model)
            self.res_updated = True
        return self.res

    def get_grad(self, model):
        """
        Compute the gradient vector

        Args:
            model: domain vector
        """
        self.set_model(model)
        if not self.grad_updated:
            self.res = self.get_res(self.model)
            self.grad = self.grad_func(self.model, self.res)
            self.gevals += 1
            if self.linear:
                self.fevals += 1
            self.grad_updated = True
        return self.grad

    def get_pert_res(self, model, pert_model):
        """Compute the perturbation residual vector (i.e., application of the Jacobian to model perturbation vector)

        Args:
            model: domain vector
            pert_model: domain perturbation vector
        """
        self.set_model(model)
        if not self.pert_res_updated or pert_model.isDifferent(self.pert_model):
            self.pert_model.copy(pert_model)
            self.pert_res = self.pert_res_func(self.model, self.pert_model)
            if self.linear:
                self.fevals += 1
            self.pert_res_updated = True
        return self.pert_res

    def get_fevals(self):
        """Get the number of objective function evalutions"""
        return self.fevals

    def get_gevals(self):
        """Get the number of gradient evalutions"""
        return self.gevals

    def obj_func(self, residual):
        """
        Compute the objective function

        Args:
            residual: residual vector
        Returns:
            objective function value
        """
        raise NotImplementedError("Implement objf for problem in the derived class!")

    def res_func(self, model):
        """
        Compute the residual vector
        
        Args:
            model: domain vector

        Returns: residual vector based on the domain
        """
        raise NotImplementedError("Implement resf for problem in the derived class!")

    def pert_res_func(self, model, pert_model):
        """
        Compute the residual vector
        Args:
            model: domain vector
            pert_model: domain perturbation vector

        Returns: residual vector
        """
        raise NotImplementedError("Implement pert_res_func for problem in the derived class!")

    def grad_func(self, model, residual):
        """
        Compute the gradient vector from the residual (i.e., g = A' r = A'(Am - d))
        Args:
            model: domain vector
            residual: residual vector

        Returns: gradient vector
        """
        raise NotImplementedError("Implement gradf for problem in the derived class!")
