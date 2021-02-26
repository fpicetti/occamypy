from __future__ import division, print_function, absolute_import

from time import time
from copy import deepcopy

import numpy as np

from occamypy import problem as P
from occamypy import solver as S
from occamypy.vector import Vector, superVector


class Operator:
    """Abstract python operator class"""
    
    # Default class methods/functions
    def __init__(self, domain, range):
        """Generic class for operator"""
        self.domain = domain.cloneSpace()
        self.range = range.cloneSpace()
    
    def __del__(self):
        """Default destructor"""
        return
    
    def __str__(self):
        return "Operator"
    
    def __repr__(self):
        return self.__str__()
    
    # unary operators
    def __add__(self, other):  # self + other
        if isinstance(other, Operator):
            return _sumOperator(self, other)
        else:
            raise TypeError('Argument must be an Operator')
    
    def __sub__(self, other):  # self - other
        self.__add__(-other)
    
    def __neg__(self):  # -self
        return _scaledOperator(self, -1)
    
    def __mul__(self, other):  # self * other
        return self.dot(other)
    
    __rmul__ = __mul__  # other * self
    
    def __truediv__(self, other, niter=2000):
        """x = A / y through CG"""
        
        if not self.range.checkSame(other):
            raise ValueError('Operator range and data domain mismatch')
        
        stopper = S.BasicStopper(niter=niter)
        problem = P.LeastSquares(model=self.domain.clone(), data=other, op=self)
        CGsolver = S.CG(stopper)
        CGsolver.run(problem, verbose=False)
        
        return problem.model
    
    # main function for all kinds of multiplication
    def dot(self, other):
        """Matrix-matrix or matrix-vector or matrix-scalar multiplication."""
        if isinstance(other, Operator):  # A * B
            return _prodOperator(self, other)
        elif type(other) in [int, float]:  # A * c or c * A
            return _scaledOperator(self, other)
        elif isinstance(other, list) and isinstance(self, Vstack):
            assert len(other) == self.n, "Other lenght and self lenght mismatch"
            return Vstack([_scaledOperator(self.ops[i], other[i]) for i in range(self.n)])
        elif isinstance(other, list) and isinstance(self, Hstack):
            assert len(other) == self.n, "Other lenght and self lenght mismatch"
            return Hstack([_scaledOperator(self.ops[i], other[i]) for i in range(self.n)])
        elif isinstance(other, Vector) or isinstance(other, superVector):  # A * x
            temp = self.range.clone()
            self.forward(False, other, temp)
            return temp
        else:
            raise TypeError('Expected Operator, (super)Vector or scalar, got %r' % other)
    
    def getDomain(self):
        """Function to return operator domain"""
        return self.domain
    
    def getRange(self):
        """Function to return operator range"""
        return self.range
    
    def setDomainRange(self, domain, range):
        """Function to set (cloning space) domain and range of the operator"""
        self.domain = domain.cloneSpace()
        self.range = range.cloneSpace()
        return
    
    def checkDomainRange(self, x, y):
        """Function to check model and data vector sizes"""
        if not self.domain.checkSame(x):
            raise ValueError("Provided x vector does not match operator domain")
        if not self.range.checkSame(y):
            raise ValueError("Provided y vector does not match operator range")
    
    def powerMethod(self, verbose=False, tol=1e-8, niter=None, eval_min=False, return_vec=False):
        """
        Function to estimate maximum eigenvalue of the operator:

        :param return_vec: boolean - Return the estimated eigenvectors [False]
        :param niter: int - Maximum number of operator applications [None]
            if not provided, the function will continue until the tolerance is reached)
        :param eval_min: boolean - Compute the minimum eigenvalue [False]
        :param verbose: boolean - Print information to screen as the method is being run [False]
        :param tol: float - Tolerance on the change of the estimated eigenvalues [1e-6]
        """
        # Cloning input and output vectors
        if verbose:
            print('Running power method to estimate maximum eigenvalue (operator L2 norm)')
        x = self.domain.clone()
        # Checking if matrix is square
        square = False
        try:
            if self.domain.checkSame(self.range):
                square = True
        except:
            pass
        if not square:
            if verbose:
                print("Note: operator is not square, the eigenvalue is associated to A'A not A!")
            d_temp = self.range.clone()
        y = self.domain.clone()
        # randomize the input vector
        x.rand()
        x.scale(1.0 / x.norm())  # Normalizing the initial vector
        y.zero()
        iiter = 0
        eigen_old = 0.0  # Previous estimated eigenvalue
        # Estimating maximum eigenvalue
        if verbose:
            print("Starting iterative process for maximum eigenvalue")
        # Starting the power iteration loop
        while True:
            # Applying adjoint if forward not square
            if square:
                self.forward(False, x, y)  # y = A x
            else:
                self.forward(False, x, d_temp)  # d = A x
                self.adjoint(False, y, d_temp)  # y = A' d = A' A x
            
            # Estimating eigenvalue (Rayleigh quotient)
            eigen = x.dot(y)  # eigen_i = x' A x / (x'x = 1.0)
            # x = y
            x.copy(y)
            # Normalization of the operator
            x.scale(1.0 / x.norm())
            # Stopping criteria (first number of iterations and then tolerance)
            iiter += 1
            if verbose:
                print("Estimated maximum eigenvalue at iter %d: %.2e" % (iiter, eigen))
            if niter is not None:
                if iiter >= niter:
                    if verbose:
                        print("Maximum number of iteration reached! Stopping iterative process!")
                    break
            # Checking change on the eigenvalue estimated value
            if abs(eigen - eigen_old) < abs(tol * eigen_old):
                if verbose:
                    print("Tolerance value reached! Stopping iterative process!")
                break
            # eigen_(i-1) = eigen_i
            eigen_old = eigen
        if eval_min:
            x_max = x.clone()  # Cloning "maximum" eigenvector
            eigen_max = deepcopy(eigen)
            # Re-initialize variables
            x.rand()
            x.scale(1.0 / x.norm())  # Normalizing the initial vector
            y.zero()
            iiter = 0
            eigen = 0.0  # Current estimated eigenvalue
            eigen_old = 0.0  # Previous estimated eigenvalue
            # Estimating the minimum eigenvalue
            # Shifting all eigenvalues by maximum one (i.e., A_min = A-muI)
            if verbose:
                print("Starting iterative process for minimum eigenvalue")
            while True:
                # Applying adjoint if forward not square
                if not square:
                    self.forward(False, x, d_temp)  # d = A x
                    self.adjoint(False, y, d_temp)  # y = A' d = A' A x
                else:
                    self.forward(False, x, y)  # y = A x
                # y = Ax - mu*Ix
                y.scaleAdd(x, 1.0, -eigen_max)
                # Estimating eigenvalue (Rayleigh quotient)
                eigen = x.dot(y)  # eigen_i = x' A_min x / (x'x = 1.0)
                # x = y
                x.copy(y)
                # Normalization of the operator
                x.scale(1.0 / x.norm())
                # Stopping criteria (first number of iterations and then tolerance)
                iiter += 1
                if verbose:
                    print("Estimated minimum eigenvalue at iter %d: %.2e"
                          % (iiter, eigen + eigen_max))
                if niter is not None:
                    if iiter >= niter:
                        if verbose:
                            print("Maximum number of iteration reached! Stopping iterative process!")
                        break
                # Checking change on the eigenvalue estimated value
                if abs(eigen - eigen_old) < abs(tol * eigen_old):
                    if verbose:
                        print("Tolerance value reached! Stopping iterative process!")
                    break
                # eigen_(i-1) = eigen_i
                eigen_old = eigen
            x_min = x.clone()  # Cloning "minimum" eigenvector
            eigen_min = deepcopy(eigen + eigen_max)
            eigen = [eigen_max, eigen_min]
            x = [x_max, x_min]
        return (eigen, x) if return_vec else eigen
    
    def dotTest(self, verbose=False, tol=1e-4):
        """
        Function to perform dot-product tests.
        :param verbose  : boolean; Flag to print information to screen as the method is being run [False]
        :param tol      : float; The function throws a Warning if the relative error is greater than maxError [1e-4]
        """
        
        def _testing(add, dt1, dt2, tol, verbose=False):
            if isinstance(dt2, np.complex):
                dt2 = np.conj(dt2)
            err_abs = dt1 - dt2
            err_rel = err_abs / abs(dt2)
            if verbose:
                print("Dot products add=%s: domain=%.6e range=%.6e " % (str(add), abs(dt1), abs(dt2)))
                print("Absolute error: %.6e" % abs(err_abs))
                print("Relative error: %.6e \n" % abs(err_rel))
            if abs(err_rel) > tol:
                # # Deleting temporary vectors
                # del d1, d2, r1, r2
                raise Warning("\tDot products failure add=%s; relative error %.2e is greater than tolerance %.2e"
                              % (str(add), err_rel, tol))
        
        if verbose:
            msg = "Dot-product tests of forward and adjoint operators"
            print(msg + "\n" + "-" * len(msg))
        
        # Allocating temporary vectors for dot-product tests
        d1 = self.domain.clone()
        d2 = self.domain.clone()
        r1 = self.range.clone()
        r2 = self.range.clone()
        
        # Randomize the input vectors
        d1.rand()
        r1.rand()
        
        # Applying forward and adjoint operators with add=False
        if verbose:
            print("Applying forward operator add=False")
        start = time()
        self.forward(False, d1, r2)
        end = time()
        if verbose:
            print(" Runs in: %s seconds" % (end - start))
            print("Applying adjoint operator add=False")
        start = time()
        self.adjoint(False, d2, r1)
        end = time()
        if verbose:
            print(" Runs in: %s seconds" % (end - start))
        
        # Computing dot products
        dt1 = d1.dot(d2)
        dt2 = r1.dot(r2)
        _testing(False, dt1, dt2, tol, verbose)
        
        # Applying forward and adjoint operators with add=True
        if verbose:
            print("Applying forward operator add=True")
        start = time()
        self.forward(True, d1, r2)
        end = time()
        if verbose:
            print(" Runs in: %s seconds" % (end - start))
            print("Applying adjoint operator add=True")
        start = time()
        self.adjoint(True, d2, r1)
        end = time()
        if verbose:
            print(" Runs in: %s seconds" % (end - start))
        
        # Computing dot products
        dt1 = d1.dot(d2)
        dt2 = r1.dot(r2)
        _testing(True, dt1, dt2, tol, verbose)
        
        if verbose:
            print("-" * 49)
        
        # Deleting temporary vectors
        del d1, d2, r1, r2
        return
    
    def forward(self, add, model, data):
        """Forward operator"""
        raise NotImplementedError("Forward must be defined")
    
    def adjoint(self, add, model, data):
        """Adjoint operator"""
        raise NotImplementedError("Adjoint must be defined")
    
    def hermitian(self):
        """Instantiate the Hermitian operator"""
        return _Hermitian(self)
    
    H = property(hermitian)
    T = H  # misleading (H is the conjugate transpose), probably we can delete it


class _Hermitian(Operator):
    
    def __init__(self, op):
        super(_Hermitian, self).__init__(op.range, op.domain)
        self.op = op
    
    def forward(self, add, model, data):
        return self.op.adjoint(add, data, model)
    
    def adjoint(self, add, model, data):
        return self.op.forward(add, data, model)


class _CustomOperator(Operator):
    """Linear operator defined in terms of user-specified operations."""
    
    def __init__(self, domain, range, forward_function, adjoint_function):
        super(_CustomOperator, self).__init__(domain, range)
        self.forward_function = forward_function
        self.adjoint_function = adjoint_function
    
    def forward(self, add, model, data):
        return self.forward_function(add, model, data)
    
    def adjoint(self, add, model, data):
        return self.adjoint_function(add, model, data)


class Vstack(Operator):
    """
    Vertical stack of operators
        y1 = | A | x
        y2   | B |
    """
    
    def __init__(self, *args):
        """Constructor for the stacked operator"""
        
        self.ops = []
        for _, arg in enumerate(args):
            if arg is None:
                continue
            elif isinstance(arg, Operator):
                self.ops.append(arg)
            elif isinstance(arg, list):
                for op in arg:
                    if op is None:
                        continue
                    elif isinstance(op, Operator):
                        self.ops.append(op)
            else:
                raise TypeError('Argument must be either Operator or Vstack')
        
        # check range
        self.n = len(self.ops)
        op_range = []
        for idx in range(self.n):
            if idx < self.n - 1:
                if not self.ops[idx].domain.checkSame(self.ops[idx + 1].domain):
                    raise ValueError('Domain incompatibility between Op %d and Op %d' % (idx, idx + 1))
            op_range += [self.ops[idx].range]
        
        super(Vstack, self).__init__(domain=self.ops[0].domain, range=superVector(op_range))
    
    def __str__(self):
        return " VStack "
    
    def forward(self, add, model, data):
        """Forward operator Cm"""
        self.checkDomainRange(model, data)
        for idx in range(self.n):
            self.ops[idx].forward(add, model, data.vecs[idx])
    
    def adjoint(self, add, model, data):
        """Adjoint operator C'r = A'r1 + B'r2"""
        self.checkDomainRange(model, data)
        self.ops[0].adjoint(add, model, data.vecs[0])
        for idx in range(1, self.n):
            self.ops[idx].adjoint(True, model, data.vecs[idx])


class Hstack(Operator):
    """
    Horizontal stack of operators
        y = [A  B]  x1
                    x2
    """
    
    def __init__(self, *args):
        """Constructor for the stacked operator"""
        
        self.ops = []
        for _, arg in enumerate(args):
            if arg is None:
                continue
            elif isinstance(arg, Operator):
                self.ops.append(arg)
            elif isinstance(arg, list):
                for op in arg:
                    if op is None:
                        continue
                    elif isinstance(op, Operator):
                        self.ops.append(op)
            else:
                raise TypeError('Argument must be either Operator or Hstack')
        
        # check domain
        self.n = len(self.ops)
        domain = []
        for idx in range(self.n):
            if idx < self.n - 1:
                if not self.ops[idx].range.checkSame(self.ops[idx + 1].range):
                    raise ValueError('Range incompatibility between Op %d and Op %d' % (idx, idx + 1))
            domain += [self.ops[0].domain]
        super(Hstack, self).__init__(domain=superVector(domain), range=self.ops[0].range)
    
    def __str__(self):
        return " HStack "
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        self.ops[0].forward(add, model.vecs[0], data)
        for idx in range(1, self.n):
            self.ops[idx].forward(True, model.vecs[idx], data)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        self.ops[0].adjoint(add, model.vecs[0], data)
        for idx in range(1, self.n):
            self.ops[idx].adjoint(add, model.vecs[idx], data)


class Dstack(Operator):
    """
    Diagonal stack of operators
    y1 = | A  0 |  x1
    y2   | 0  B |  x2
    """
    
    def __init__(self, *args):
        """Constructor for the stacked operator"""
        
        self.ops = []
        for _, arg in enumerate(args):
            if arg is None:
                continue
            elif isinstance(arg, Operator):
                self.ops.append(arg)
            elif isinstance(arg, list):
                for op in arg:
                    if op is None:
                        continue
                    elif isinstance(op, Operator):
                        self.ops.append(op)
            else:
                raise TypeError('Argument must be either Operator or list of Operators')
        
        # build domain and range
        self.n = len(self.ops)
        op_range = []
        op_domain = []
        for idx in range(self.n):
            op_domain += [self.ops[idx].domain]
            op_range += [self.ops[idx].range]
        
        super(Dstack, self).__init__(domain=superVector(op_domain), range=superVector(op_range))
    
    def __str__(self):
        return " DStack "
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        for idx in range(self.n):
            self.ops[idx].forward(add, model.vecs[idx], data.vecs[idx])
    
    def adjoint(self, add, model, data):
        """Adjoint operator"""
        self.checkDomainRange(model, data)
        for idx in range(self.n):
            self.ops[idx].adjoint(add, model.vecs[idx], data.vecs[idx])


class _sumOperator(Operator):
    """
    Sum of two operators
        C = A + B
        C.H = A.H + B.H
    """
    
    def __init__(self, A, B):
        """Sum operator constructor"""
        if not isinstance(A, Operator) or not isinstance(B, Operator):
            raise TypeError('Both operands have to be a Operator')
        if not A.range.checkSame(B.range) or not A.domain.checkSame(B.domain):
            raise ValueError('Cannot add operators: shape mismatch')
        
        super(_sumOperator, self).__init__(A.domain, A.range)
        self.args = (A, B)
    
    def __str__(self):
        return self.args[0].__str__()[:3] + "+" + self.args[1].__str__()[:4]
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        self.args[0].forward(add, model, data)
        self.args[1].forward(True, model, data)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        self.args[0].adjoint(add, model, data)
        self.args[1].adjoint(True, model, data)


class _prodOperator(Operator):
    """
    Multiplication of two operators
    C = A * B
    C.H = B.H * A.H
    """
    
    def __init__(self, A, B):
        if not isinstance(A, Operator) or not isinstance(B, Operator):
            raise TypeError('Both operands have to be a Operator')
        if not A.domain.checkSame(B.range):
            raise ValueError('Cannot multiply operators: shape mismatch')
        super(_prodOperator, self).__init__(B.domain, A.range)
        self.args = (A, B)
        self.temp = B.getRange().clone()
    
    def __str__(self):
        return self.args[0].__str__()[:3] + "*" + self.args[1].__str__()[:4]
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        self.args[1].forward(False, model, self.temp)
        self.args[0].forward(add, self.temp, data)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        self.args[0].adjoint(False, self.temp, data)
        self.args[1].adjoint(add, model, self.temp)


class _scaledOperator(Operator):
    """
    Scalar matrix multiplication
    """
    
    def __init__(self, A, const):
        if not isinstance(A, Operator):
            raise TypeError('Operator expected as A')
        if not type(const) in [int, float]:
            raise ValueError('scalar expected as const')
        super(_scaledOperator, self).__init__(A.domain, A.range)
        self.const = const
        self.op = A
    
    def __str__(self):
        op_name = self.op.__str__().replace(" ", "")
        op_name_len = len(op_name)
        if op_name_len <= 6:
            name = "sc" + op_name + "" * (6 - op_name_len)
        else:
            name = "sc" + op_name[:6]
        return name
    
    def forward(self, add, model, data):
        self.op.forward(add, model.clone().scale(self.const), data)
    
    def adjoint(self, add, model, data):
        self.op.adjoint(add, model, data.clone().scale(np.conj(self.const)))


def Chain(A, B):
    """
         Chain of two operators
                d = B A m
    """
    return _prodOperator(B, A)


# for backward compatibility
Transpose = Operator.H
sumOperator = _sumOperator
stackOperator = Vstack
