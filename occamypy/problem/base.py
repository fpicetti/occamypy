class Bounds:
    """Class used to enforce boundary constraints during the inversion"""

    def __init__(self, minBound=None, maxBound=None):
        """
        Bounds constructor
        minBound    = [None] - vector class; vector containing minimum values of the model vector
        maxBound    = [None] - vector class; vector containing maximum values of the model vector
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

    def apply(self, input_vec):
        """
        Function for applying the model bounds
        """
        if self.minBound is not None and self.maxBound is None:
            if not input_vec.checkSame(self.minBound):
                raise ValueError("Input vector not consistent with bound space")
            input_vec.scale(-1.0)
            input_vec.clipVector(input_vec, self.minBound)
            input_vec.scale(-1.0)
        elif self.minBound is None and self.maxBound is not None:
            if not input_vec.checkSame(self.maxBound):
                raise ValueError("Input vector not consistent with bound space")
            input_vec.clipVector(input_vec, self.maxBound)
        elif self.minBound is not None and self.maxBound is not None:
            if (not (input_vec.checkSame(self.minBound) and input_vec.checkSame(
                    self.maxBound))):
                raise ValueError("Input vector not consistent with bound space")
            input_vec.clipVector(self.minBound, self.maxBound)
        return


class Problem:
    """Problem parent object"""

    # Default class methods/functions
    def __init__(self, minBound=None, maxBound=None, boundProj=None):
        """Default class constructor for Problem"""
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
        """Default destructor"""
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

    def set_model(self, model):
        """Setting internal model vector"""
        if model.isDifferent(self.model):
            self.model.copy(model)
            self.obj_updated = False
            self.res_updated = False
            self.grad_updated = False
            self.dres_updated = False

    def set_residual(self, residual):
        """Setting internal residual vector"""
        # Useful for linear inversion (to avoid residual computation)
        if self.res.isDifferent(residual):
            self.res.copy(residual)
            # If residuals have changed, recompute gradient and objective function value
            self.grad_updated = False
            self.obj_updated = False
        self.res_updated = True
        return

    def get_model(self):
        """Accessor for model vector"""
        return self.model

    def get_dmodel(self):
        """Accessor for model vector"""
        return self.dmodel

    def get_rnorm(self, model):
        """Accessor for residual vector norm"""
        self.get_res(model)
        return self.get_res(model).norm()

    def get_gnorm(self, model):
        """Accessor for gradient vector norm"""
        return self.get_grad(model).norm()

    def get_obj(self, model):
        """Accessor for objective function"""
        self.set_model(model)
        if not self.obj_updated:
            self.res = self.get_res(self.model)
            self.obj = self.objf(self.res)
            self.obj_updated = True
        return self.obj

    def get_res(self, model):
        """Accessor for residual vector"""
        self.set_model(model)
        if not self.res_updated:
            self.fevals += 1
            self.res = self.resf(self.model)
            self.res_updated = True
        return self.res

    def get_grad(self, model):
        """Accessor for gradient vector"""
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
        """Accessor for dresidual vector (i.e., application of the Jacobian to Dmodel vector)"""
        self.set_model(model)
        if not self.dres_updated or dmodel.isDifferent(self.dmodel):
            self.dmodel.copy(dmodel)
            self.dres = self.dresf(self.model, self.dmodel)
            if self.linear:
                self.fevals += 1
            self.dres_updated = True
        return self.dres

    def get_fevals(self):
        """Accessor for number of objective function evalutions"""
        return self.fevals

    def get_gevals(self):
        """Accessor for number of gradient evalutions"""
        return self.gevals

    def objf(self, residual):
        """Dummy objf running method, must be overridden in the derived class"""
        raise NotImplementedError("Implement objf for problem in the derived class!")

    def resf(self, model):
        """Dummy resf running method, must be overridden in the derived class"""
        raise NotImplementedError("Implement resf for problem in the derived class!")

    def dresf(self, model, dmodel):
        """Dummy dresf running method, must be overridden in the derived class"""
        raise NotImplementedError("Implement dresf for problem in the derived class!")

    def gradf(self, model, residual):
        """Dummy gradf running method, must be overridden in the derived class"""
        raise NotImplementedError("Implement gradf for problem in the derived class!")
