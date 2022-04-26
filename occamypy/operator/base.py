from __future__ import division, print_function, absolute_import

from time import time
from copy import deepcopy

import numpy as np
import torch

from occamypy.vector.base import Vector, superVector


class Operator:
    """
    Abstract python operator class

    Args:
        domain: domain vector
        range: range vector
        name: string that describes the operator

    Attributes:
        domain: domain vector space
        range: range vector space
        H: hermitian operator

    Methods:
        dot: dot-product with input object
        getDomain: get domain vector space
        getRange: get range vector space
        setDomainRange: set both domain and range space
        checkDomainRange: check whether the input domain and range match with operator
        powerMethod: estimate the maximum eigenvalue iteratively
        dotTest: dot-product test, or adjointness test
        forward: forward operation
        adjoint: adjoint (conjugate-tranpose) operation
    """
    def __init__(self, domain, range, name: str = "Operator"):
        """
        Operator constructor

        Args:
            domain: domain vector
            range: range vector
        """
        self.domain = domain.cloneSpace()
        self.range = range.cloneSpace()
        self.name = str(name)
    
    def __str__(self):
        return self.name
    
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
    
    def __truediv__(self, other, niter: int = 2000):
        """x = op / y through CG"""
        from occamypy.problem.linear import LeastSquares
        from occamypy.solver.stopper import BasicStopper
        from occamypy.solver.linear import CG
        
        if not self.range.checkSame(other):
            raise ValueError('Operator range and data domain mismatch')
        
        stopper = BasicStopper(niter=niter)
        problem = LeastSquares(model=self.domain.clone(), data=other, op=self)
        CGsolver = CG(stopper)
        CGsolver.run(problem, verbose=False)
        
        return problem.model
    
    # main function for all kinds of multiplication
    def dot(self, other):
        """Matrix-matrix or matrix-vector or matrix-scalar multiplication."""
        if isinstance(other, Operator):  # op * B
            return _prodOperator(self, other)
        elif type(other) in [int, float]:  # op * c or c * op
            return _scaledOperator(self, other)
        elif isinstance(other, list) and isinstance(self, Vstack):
            if len(other) != self.n:
                raise ValueError("Other lenght and self lenght mismatch")
            return Vstack([_scaledOperator(self.ops[i], other[i]) for i in range(self.n)])
        elif isinstance(other, list) and isinstance(self, Hstack):
            if len(other) != self.n:
                raise ValueError("Other lenght and self lenght mismatch")
            return Hstack([_scaledOperator(self.ops[i], other[i]) for i in range(self.n)])
        elif isinstance(other, Vector) or isinstance(other, superVector):  # op * x
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
        Function to estimate maximum eigenvalue of the operator

        Args:
            verbose: verbosity flag
            tol: stopping tolerance on the change of the estimated eigenvalues
            niter: maximum number of operator applications;
                if not provided, the function will continue until the tolerance is reached
            eval_min: whether to compute the minimum eigenvalue
            return_vec: whether to return the estimated eigenvectors

        Returns:
            (eigenvalues, eigenvectors) if return_vec else (eigenvalues)
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
                print("Note: operator is not square, the eigenvalue is associated to op'op not op!")
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
                self.forward(False, x, y)  # y = op x
            else:
                self.forward(False, x, d_temp)  # d = op x
                self.adjoint(False, y, d_temp)  # y = op' d = op' op x
            
            # Estimating eigenvalue (Rayleigh quotient)
            eigen = x.dot(y)  # eigen_i = x' op x / (x'x = 1.0)
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
            # Shifting all eigenvalues by maximum one (i.e., A_min = op-muI)
            if verbose:
                print("Starting iterative process for minimum eigenvalue")
            while True:
                # Applying adjoint if forward not square
                if not square:
                    self.forward(False, x, d_temp)  # d = op x
                    self.adjoint(False, y, d_temp)  # y = op' d = op' op x
                else:
                    self.forward(False, x, y)  # y = op x
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
        Perform the dot-product test
        
        Args:
            verbose: verbosity flag
            tol: the function throws a Warning if the relative error is greater than tol
        """
        def _process_complex(x):
            if isinstance(x, complex):
                x = np.conj(x)
            elif isinstance(x, torch.Tensor) and x.dtype in [torch.complex64, torch.complex128]:
                x = x.real
            return x
            
        def _testing(add, dt1, dt2, tol, verbose=False):
            dt2 = _process_complex(dt2)
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
        super(_Hermitian, self).__init__(op.range, op.domain, name=op.name)
        self.op = op
    
    def forward(self, add, model, data):
        return self.op.adjoint(add, data, model)
    
    def adjoint(self, add, model, data):
        return self.op.forward(add, data, model)


class _CustomOperator(Operator):
    """Linear operator defined in terms of user-specified operations."""
    
    def __init__(self, domain, range, fwd_fn, adj_fn):
        """
        CustomOperator constructor

        Args:
            domain: domain vector
            range: range vectorr
            fwd_fn: callable function of the kind f(add, domain, data)
            adj_fn: callable function of the kind f(add, domain, data)
        """
        super(_CustomOperator, self).__init__(domain, range)
        self.forward_function = fwd_fn
        self.adjoint_function = adj_fn
    
    def forward(self, add, model, data):
        return self.forward_function(add, model, data)
    
    def adjoint(self, add, model, data):
        return self.adjoint_function(add, model, data)


class Vstack(Operator):
    """
    Vertical stack of operators
        y1 = | op | x
        y2   | B |
    """
    
    def __init__(self, *args):
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
        
        super(Vstack, self).__init__(domain=self.ops[0].domain, range=superVector(op_range), name="Vstack")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        for idx in range(self.n):
            self.ops[idx].forward(add, model, data.vecs[idx])
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        self.ops[0].adjoint(add, model, data.vecs[0])
        for idx in range(1, self.n):
            self.ops[idx].adjoint(True, model, data.vecs[idx])


class Hstack(Operator):
    """
    Horizontal stack of operators
        y = [op  B]  x1
                    x2
    """
    
    def __init__(self, *args):
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
        super(Hstack, self).__init__(domain=superVector(domain), range=self.ops[0].range, name="Hstack")
    
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
    y1 = | op  0 |  x1
    y2   | 0  B |  x2
    """
    
    def __init__(self, *args):
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
        
        super(Dstack, self).__init__(domain=superVector(op_domain), range=superVector(op_range), name="Dstack")
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        for idx in range(self.n):
            self.ops[idx].forward(add, model.vecs[idx], data.vecs[idx])
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        for idx in range(self.n):
            self.ops[idx].adjoint(add, model.vecs[idx], data.vecs[idx])


class _sumOperator(Operator):
    """
    Sum of two operators
        C = op + B
        C.H = op.H + B.H
    """
    
    def __init__(self, A, B):
        if not isinstance(A, Operator) or not isinstance(B, Operator):
            raise TypeError('Both operands have to be a Operator')
        if not A.range.checkSame(B.range) or not A.domain.checkSame(B.domain):
            raise ValueError('Cannot add operators: shape mismatch')
        
        super(_sumOperator, self).__init__(A.domain, A.range, name=A.name+"+"+B.name)
        self.args = (A, B)
    
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
    C = op * B
    C.H = B.H * op.H
    """
    
    def __init__(self, A, B):
        if not isinstance(A, Operator) or not isinstance(B, Operator):
            raise TypeError('Both operands have to be a Operator')
        if not A.domain.checkSame(B.range):
            raise ValueError('Cannot multiply operators: shape mismatch')
        super(_prodOperator, self).__init__(B.domain, A.range, name=A.name+"*"+B.name)
        self.args = (A, B)
        self.temp = B.getRange().clone()
    
    def forward(self, add, model, data):
        self.checkDomainRange(model, data)
        self.args[1].forward(False, model, self.temp)
        self.args[0].forward(add, self.temp, data)
    
    def adjoint(self, add, model, data):
        self.checkDomainRange(model, data)
        self.args[0].adjoint(False, self.temp, data)
        self.args[1].adjoint(add, model, self.temp)


class _scaledOperator(Operator):
    """Scaled operator B = c op"""
    
    def __init__(self, op, const):
        """
        ScaledOperator constructor.

        Args:
            op: operator
            const: scaling factor
        """
        if not isinstance(op, Operator):
            raise TypeError('Operator expected as op')
        if not type(const) in [int, float]:
            raise ValueError('scalar expected as const')
        super(_scaledOperator, self).__init__(op.domain, op.range, name="sc"+op.name)
        self.const = const
        self.op = op
    
    def forward(self, add, model, data):
        self.op.forward(add, model.clone().scale(self.const), data)
    
    def adjoint(self, add, model, data):
        self.op.adjoint(add, model, data.clone().scale(np.conj(self.const)))


def Chain(A, B):
    """
    Chain of two linear operators:
        d = B op m

    Notes:
        this function is deprecated, as you can simply write (B * op). Watch out for the order!
    """
    return _prodOperator(B, A)


# for backward compatibility
Transpose = Operator.H
sumOperator = _sumOperator
stackOperator = Vstack
