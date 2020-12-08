import numpy as np

from occamypy.operator import Operator, Vstack
from occamypy.operator.base import _sumOperator, _prodOperator


def dummy_set_background(dummy_arg):
    """
    Dummy function to use Non-linear operator class for Linear ones (it takes one argument and does nothing)
    """
    return


class NonlinearOperator(Operator):
    """
    Non-linear operator class
    """
    
    def __init__(self, nl_op, lin_op=None, set_background_func=dummy_set_background):
        """
           Constructor for non-linear operator class:
           nl_op                = [no default] - operator class;
                                Non-linear operator class where only the forward is overwritten
           lin_op               = [no default] - operator class;
                                Linear Jacobian operator class where only the forward is
                                overwritten (if not necessary, use pyOperator.ZeroOp)
           set_background_func  = [dummy_set_background] - function pointer;
                                Function to set the model vector on which the
                                Jacobian operator is evaluated
        """
        # Setting non-linear and linearized operators
        self.nl_op = nl_op
        self.lin_op = lin_op if lin_op is not None else nl_op
        self.set_background = set_background_func
        # Checking if domain of the operators is the same
        if not self.nl_op.domain.checkSame(self.lin_op.domain):
            raise ValueError("ERROR! The two provided operators have different domains")
        if not self.nl_op.range.checkSame(self.lin_op.range):
            raise ValueError("ERROR! The two provided operators have different ranges")
        super(NonlinearOperator, self).__init__(self.nl_op.domain, self.nl_op.range)
    
    def dotTest(self, **kwargs):
        """
        Raising an exception, dot-product tests must be performed directly onto linear operator.
        """
        raise NotImplementedError("Perform dot-product tests directly on the linear operator.")
    
    def linTest(self, background, pert=None, alpha=np.logspace(-6, 0, 100), plot=False):
        """
        Linearization tests function. It plots the model-perturbation norm vs linearization error norm
        :param background: vector class - Background model used during the linearization tests
        :param pert: vector class - Model-perturbation vector to be used during linearization tests [None]
                     if not provided a random perturbation is employed
        :param alpha: array - array of scalars to scale the pert vector during the tests [np.logspace(-6,0,100)]
        :param plot: boolean - whether to plot the linearization error vs perturbation scale or not
        :return:
        :param scale
        :param lin_err array - array containing linearization error for each scale value
        """
        # Creating model perturbation if not provided
        if pert is None:
            pert = self.getDomain().clone().rand()
        # List containing linearization error and perturbation scale/norm
        lin_err = []
        # temporary vectors
        m0 = background.clone()
        m = background.clone()
        d0 = self.nl_op.getRange().clone()
        d1 = self.nl_op.getRange().clone()
        dlin = self.nl_op.getRange().clone()
        # computing f(m0) = d0
        self.nl_op.forward(False, m0, d0)
        # setting m0 for the Jacobian matrix
        self.set_background(m0)
        # computing F(m0)dm = dlin
        self.lin_op.forward(False, pert, dlin)
        for sc in alpha:
            # print(sc)
            # computing f(m0+dm) = d1
            m.copy(m0)
            m.scaleAdd(pert, 1.0, sc)
            self.nl_op.forward(False, m, d1)
            # computing f(m0+dm) - f(m0)
            d1.scaleAdd(d0, 1.0, -1.0)
            # computing f(m0+dm) - f(m0) - F(m0)dm (i.e., linearization error)
            d1.scaleAdd(dlin, 1.0, -sc)
            lin_err.append(d1.norm())
        lin_err = np.array(lin_err)
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            plt.plot(alpha, lin_err, 'r')
            ax.autoscale(enable=True, axis='y', tight=True)
            ax.autoscale(enable=True, axis='x', tight=True)
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$|f(m_0+\alpha dm) - f(m_0) - \alpha F(m_0)dm|_2$")
            plt.title('Linearization error')
            plt.show()
        return alpha, lin_err
    
    # unary operators
    def __add__(self, other):  # self + other
        if isinstance(other, NonlinearOperator):
            return _sumNlOperator(self, other)
        else:
            raise TypeError('Argument must be an Operator')


class NonlinearComb(NonlinearOperator):
    """
    Combination of non-linear opeartors: f(g(m))
    """
    
    def __init__(self, f, g):
        """
        Constructor for non-linear operator class
        """
        # Checking if non-linear operators were provided
        if not (isinstance(f, NonlinearOperator) and isinstance(g, NonlinearOperator)):
            raise TypeError("Provided operators has to be NonLinearOperator")
        # Defining f(g(m))
        self.nl_op = _prodOperator(f.nl_op, g.nl_op)
        # Defining F(g(m0))G(m0)
        self.lin_op = _prodOperator(f.lin_op, g.lin_op)
        # Defining internal set_background functions
        self.set_background_f = f.set_background
        self.set_background_g = g.set_background
        # Defining non_linear operator g(m) for Jacobian definition
        self.g_nl_op = g.nl_op
        self.g_range_tmp = g.nl_op.range.clone()
        super(NonlinearComb, self).__init__(self.nl_op, self.lin_op, self.set_background)
    
    def set_background(self, model):
        """
        Set background function for the chain of Jacobian matrices
        """
        # Setting G(m0)
        self.set_background_g(model)
        # Setting F(g(m0))
        self.g_nl_op.forward(False, model, self.g_range_tmp)
        self.set_background_f(self.g_range_tmp)


# Necessary for backward compatibility
def CombNonlinearOp(g, f):
    """Combination of non-linear operators: f(g(m))"""
    return NonlinearComb(f, g)


class _sumNlOperator(NonlinearOperator):
    """
    Sum of two non-linear operators
        h = g + f
    """
    
    def __init__(self, g, f):
        """Sum operator constructor"""
        if not isinstance(g, NonlinearOperator) or not isinstance(f, NonlinearOperator):
            raise TypeError('Both operands have to be a NonLinearOperator')
        if not f.range.checkSame(g.range) or not f.domain.checkSame(g.domain):
            raise ValueError('Cannot add operators: shape mismatch')
        
        self.args = (g, f)
        # Defining f(m) + g(m)
        self.nl_op = _sumOperator(f.nl_op, g.nl_op)
        # Defining F(m0) and G(m0)
        self.lin_op = _sumOperator(f.lin_op, g.lin_op)
        # Defining internal set_background functions
        self.set_background_f = f.set_background
        self.set_background_g = g.set_background
        # Defining non_linear operator g(m) for Jacobian definition
        self.g_nl_op = g.nl_op
        self.g_range_tmp = g.nl_op.range.clone()
        super(_sumNlOperator, self).__init__(self.nl_op, self.lin_op, self.set_background)
    
    def __str__(self):
        return self.args[0].__str__()[:3] + "+" + self.args[1].__str__()[:4]
    
    def set_background(self, model):
        """
        Set background function for the sum of Jacobian matrices
        """
        # Setting G(m0)
        self.set_background_g(model)
        # Setting F(m0)
        self.set_background_f(model)


def NonlinearSum(f, g):
    """Combination of non-linear operators: f(m) + g(m)"""
    return _sumNlOperator(f, g)


class NonlinearVstack(NonlinearOperator):
    """
    Stack of operators class
            | d1 |   | f(m) |
     h(m) = |    | = |      |
            | d2 |   | g(m) |
    """
    
    def __init__(self, nl_op1, nl_op2):
        """Constructor for the stacked operator"""
        # Checking if domain of the operators is the same
        if not (isinstance(nl_op1, NonlinearOperator) and isinstance(nl_op2, NonlinearOperator)):
            raise TypeError("Provided operators must be NonLinearOperator instances")
        self.nl_op1 = nl_op1  # f(m)
        self.nl_op2 = nl_op2  # g(m)
        # Defining f(g(m))
        self.nl_op = Vstack(nl_op1.nl_op, nl_op2.nl_op)
        # Defining F(g(m0))G(m0)
        self.lin_op = Vstack(nl_op1.lin_op, nl_op2.lin_op)
        # Defining internal set_background functions
        self.set_background1 = nl_op1.set_background
        self.set_background2 = nl_op2.set_background
        super(NonlinearVstack, self).__init__(self.nl_op, self.lin_op, self.set_background)
    
    def __str__(self):
        return "NLVstack"
    
    def set_background(self, model):
        """
        Set background function for the stack of Jacobian matrices
        """
        # Setting F(m0)
        self.set_background1(model)
        # Setting G(m0)
        self.set_background2(model)

# TODO add Hstack and Dstack


# Variable Projection Operator
class VarProOperator(Operator):
    """
        Operator of the form: h(m_nl)m_lin, for Variable-projection method
    """
    
    def __init__(self, h_nl, h_lin, set_nl, set_lin_jac, set_lin=None):
        """
            Constructor for an operator with a linear and non-linear model component
            Required arguments:
            h_nl    	= [no default] - non-linear operator class; Non-linear operator class
            h_lin   	= [no default] - operator class; Linear operator class
            set_nl  	= [no default] - class function pointer; Class function to set non-linear part within h_lin
            set_lin_jac = [no default] - class function pointer; Class function to set linear part within the Jacobian h_nl (if not necessary, use pyOperator.dummy_set_background)
            #Optional arguments:
            set_lin 	= [None] - class function pointer; Class function to set linear part within h_nl (not used during an inversion if ProblemL2VpReg is used)
        """
        if not isinstance(h_nl, NonlinearOperator):
            raise TypeError("ERROR! Not provided a non-linear operator class for h_nl")
        self.h_nl = h_nl
        self.h_lin = h_lin
        # Checking the range spaces
        if not h_nl.nl_op.range.checkSame(h_lin.range):
            raise ValueError("ERROR! The two provided operators have different ranges")
        self.set_nl = set_nl  # Function to set the non-linear component of the h(m_nl)
        self.set_lin_jac = set_lin_jac  # Function to set the non-linear component of the Jacobian H(m_nl;m_lin)
        self.set_lin = set_lin  # Function to set the non-linear component h(m_nl)m_lin
    
    def dotTest(self, verb=False, maxError=.0001):
        """
           Raising an exception, dot-product tests must be performed directly onto linear operator and the Jacobian of h(m_nl).
        """
        raise NotImplementedError(
            "ERROR! Perform dot-product tests directly onto linear operator and Jacobian of h(m_nl).")


# simple non-linear operator to tests linTest method
class cosOperator(Operator):
    """Cosine non-linear operator"""
    
    def __init__(self, domain):
        super(cosOperator, self).__init__(domain, domain)
    
    def forward(self, add, model, data):
        """Forward operator cos(x)"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        data.getNdArray()[:] += np.cos(model.getNdArray())
        return


class cosJacobian(Operator):
    """Jacobian of cosine non-linear operator (i.e., -sin(x0)dx)"""
    
    def __init__(self, domain):
        super(cosJacobian, self).__init__(domain, domain)
        self.background = domain.clone()
        self.backgroundNd = self.background.getNdArray()
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        data.getNdArray()[:] -= np.sin(self.backgroundNd) * model.getNdArray()
        return
    
    def adjoint(self, add, model, data):
        """Adjoint operator"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        model.getNdArray()[:] -= np.sin(self.backgroundNd) * data.getNdArray()
        return
    
    def set_background(self, background):
        """ Setting -sin(x0)"""
        self.background.copy(background)
        return
