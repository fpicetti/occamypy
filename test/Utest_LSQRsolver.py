#!/usr/bin/env python3
import sys
sys.path.insert(0, "../../python")
import pyVector
import pyOperator
import pyNpOperator
from pyLinearSolver import LSQRsolver, LCGsolver
from pyProblem import ProblemL2Linear, ProblemL2LinearReg
from pyStopper import BasicStopper
import numpy as np
try:
    from scipy.sparse.linalg import lsqr
except ImportError:
    import subprocess
    import sys
    subprocess.call([sys.executable, "-m", "pip", "install", "--user", "scipy"])
    from scipy.sparse.linalg import lsqr
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    # same example as scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)
    model = pyVector.VectorIC(np.zeros(2, dtype=float))
    data = pyVector.VectorIC(np.array([1., 0.01, -1.], dtype=float))
    A = pyNpOperator.MatrixOp(np.array([[1., 0.], [1., 1.], [0., 1.]], dtype=float), model, data)
    
    # CG for benchmarking
    L2ProbCG = ProblemL2Linear(model, data, A)
    CG = LCGsolver(BasicStopper(niter=1000))
    CG.run(L2ProbCG, verbose=False)
    print('CG result: \t\t', L2ProbCG.model.getNdArray())  # should be near [1, -1]
    
    # LSQR from scipy
    print('scipy result: \t', lsqr(A.getNdArray(), data.getNdArray(), show=False)[0])
    
    # LSQR
    L2Prob = ProblemL2Linear(model, data, A)
    LSQR = LSQRsolver(BasicStopper(niter=1000))
    LSQR.run(L2Prob, verbose=False)
    print('LSQR result: \t', L2Prob.model.getNdArray())  # should be near [1, -1]
    
    # another example
    np.random.seed(1)
    nx = 101
    x = pyVector.VectorIC((nx,)).zero()
    x.getNdArray()[:nx // 2] = 10
    x.getNdArray()[nx // 2:3 * nx // 4] = -5
    Iop = pyOperator.IdentityOp(x)
    L = pyNpOperator.SecondDerivative(x)
    n = x.clone()
    n.getNdArray()[:] = np.random.normal(0, 1, nx)
    y = Iop * (x.clone() + n)

    plt.figure(figsize=(5, 4))
    plt.plot(x.getNdArray(), 'k', lw=1, label='x')
    plt.plot(y.getNdArray(), '.k', label='y=x+n')
    plt.legend()
    plt.title('Model, Data and Derivative')
    plt.show()

    # CG solver
    problemCG = ProblemL2Linear(x.clone().zero(), y, Iop)
    CG = LCGsolver(BasicStopper(niter=30))
    CG.run(problemCG, verbose=False)

    plt.figure(figsize=(5, 4))
    plt.plot(x.getNdArray(), 'k', lw=1, label='x')
    plt.plot(y.getNdArray(), '.k', label='y=x+n')
    plt.plot(problemCG.model.getNdArray(), 'r', lw=1, label='x_inv')
    plt.legend()
    plt.title('CG')
    plt.show()

    # LSQR solver
    problemLSQR = ProblemL2Linear(x.clone().zero(), y, Iop)
    LSQR = LSQRsolver(BasicStopper(niter=1000))
    LSQR.run(problemLSQR, verbose=False)

    plt.figure(figsize=(5, 4))
    plt.plot(x.getNdArray(), 'k', lw=1, label='x')
    plt.plot(y.getNdArray(), '.k', label='y=x+n')
    plt.plot(problemLSQR.model.getNdArray(), 'r', lw=1, label='x_inv')
    plt.legend()
    plt.title('LSQR')
    plt.show()

    # CG solver with L2 regularization
    problemCGL = ProblemL2LinearReg(x.clone().zero(), y, Iop, np.sqrt(50), reg_op=L)
    CG = LCGsolver(BasicStopper(niter=30))
    CG.run(problemCGL, verbose=False)
    
    plt.figure(figsize=(5, 4))
    plt.plot(x.getNdArray(), 'k', lw=1, label='x')
    plt.plot(y.getNdArray(), '.k', label='y=x+n')
    plt.plot(problemCGL.model.getNdArray(), 'r', lw=1, label='x_inv')
    plt.legend()
    plt.title('CG with Laplacian reg')
    plt.show()

    # LSQR solver with L2 regularization
    problemLSQRL = ProblemL2LinearReg(x.clone().zero(), y, Iop, np.sqrt(50), reg_op=L)
    LSQR = LSQRsolver(BasicStopper(niter=30))
    LSQR.run(problemLSQRL, verbose=False)

    plt.figure(figsize=(5, 4))
    plt.plot(x.getNdArray(), 'k', lw=1, label='x')
    plt.plot(y.getNdArray(), '.k', label='y=x+n')
    plt.plot(problemLSQRL.model.getNdArray(), 'r', lw=1, label='x_inv')
    plt.legend()
    plt.title('LSQR with Laplacian reg')
    plt.show()