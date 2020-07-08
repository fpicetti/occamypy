#!/usr/bin/env python3
import sys

sys.path.insert(0, "../../python")
import pyVector as Vec
import pyOperator as Op
from pyNpOperator import MatrixOp
from pyLinearSolver import LCGsolver as LCG
from pyLinearSolver import SymLCGsolver as SymLCGsolver
import pyProblem as Prblm
from pyStopper import BasicStopper as Stopper
from sys_util import logger
import sep_util as sep
import numpy as np

# Importing scipy to compare CG behavior
from scipy.sparse.linalg import cg

# Testing the NLCG to solver a regularized linear problem treated as if it was non linear
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as BFGS


if __name__ == '__main__':
    # In-core run
    # Creating model vector
    model_vec = Vec.VectorIC(np.zeros((100, 1)))
    model_vec.zero()
    # Creating data vector
    data_vec = Vec.VectorIC(np.zeros((200, 1)))
    data_vec.rand()
    # Matrix to be inverted
    A = np.random.rand(200, 100)
    # Create operator
    MatMult = MatrixOp(A, model_vec, data_vec)
    # Create L2-norm linear problem
    L2Prob = Prblm.LeastSquares(model_vec, data_vec, MatMult)
    # L2ProbReg = Prblm.ProblemL2LinearReg(model_vec, data_vec, MatMult, 0.0001)
    # Create stopper
    niter = 2000
    Stop = Stopper(niter=niter)  # ,tolobjchng=1e-15)
    # Create solver
    LCGsolver = LCG(Stop)
    LCGsolver.setDefaults(iter_sampling=1, save_obj=True, save_model=True, prefix="test_junk")
    # Running the solver
    # LCGsolver.run(L2Prob, verbose=True)

    # Out-of-core run
    # Creating model vector
    # model_vecOC = Vec.vectorOC(model_vec)
    # Creating data vector
    # data_vecOC  = Vec.vectorOC(data_vec)
    # Create operator
    # MatMultOC = MatMult_outcore(A,model_vecOC,data_vecOC)
    # Create L2-norm linear problem
    # L2Prob_outcore = Prblm.ProblemL2Linear(model_vecOC,data_vecOC,MatMultOC)
    # Running the solver
    # LCGsolver.setDefaults()
    # LCGsolver.run(L2Prob_outcore,True)

    # Testing inversion of a symmetric matrix (second-order derivative operator)
    n = 200
    A = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(A, -2)
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    model_vec_sym = Vec.VectorIC(np.zeros((n, 1), dtype=np.float64))
    data_vec_sym = Vec.VectorIC(np.zeros((n, 1), dtype=np.float64))
    # Constant derivative
    data_vec_sym.set(1.)
    # Create operator
    MatMultSym = MatrixOp(A, model_vec_sym, data_vec_sym)
    # Inverse of A as preconditioning
    Prec = MatrixOp(np.linalg.inv(A), model_vec_sym, data_vec_sym)
    # Computing max and min eigenvalues using power method
    # eg,vec=MatMultSym.powerMethod(verbose=False,eval_min=True,return_vec=True,tol=1e-18)
    # print("power",eg)
    # eigenValues, eigenVectors = np.linalg.eig(A)
    # idx = eigenValues.argsort()[::-1]
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]
    # print(eigenValues[-1],eigenValues[0])
    # print("max eigen vec",np.append(vec[0].getNdArray(),eigenVectors[:,-1],axis=1))
    # print("min eigen vec",np.append(vec[1].getNdArray(),eigenVectors[:,0],axis=1))
    # quit()
    # Create L2-norm linear problem
    L2Prob_sym = Prblm.LeastSquares(model_vec_sym, data_vec_sym, MatMultSym)
    # Running the solver
    # LCGsolver.setDefaults(iter_buffer_size=None,iter_sampling=1000,save_obj=True,save_model=True,prefix="sym_test")
    # LCGsolver.run(L2Prob_sym,True)

    L2Prob_sym = Prblm.LeastSquares(model_vec_sym, data_vec_sym, MatMultSym, prec=Op.ChainOperator(Prec, Prec))
    # LCGsolver.run(L2Prob_sym,True)

    # Testing LCG with regularized problem
    L2Prob_reg = Prblm.RegularizedL2(model_vec_sym, data_vec_sym, MatMultSym, 0.0001)
    # L2Prob_reg.estimate_epsilon(True)
    # Running the solver
    # LCGsolver.setDefaults(iter_sampling=100,iter_buffer_size=1,save_obj=True,save_model=True,save_grad=True,save_res=True,prefix="lin_test")
    # LCGsolver.run(L2Prob_reg,verbose=True)

    # Testing estimate_epsilon when initial model different than zero
    model_vec_sym.rand()
    L2Prob_reg1 = Prblm.RegularizedL2(model_vec_sym, data_vec_sym, MatMultSym, 0.0001)
    # L2Prob_reg1.estimate_epsilon(True)
    model_vec_sym.zero()

    # Testing LCG for symmetric systems
    low_bound = model_vec_sym.clone()
    low_bound.set(-2000.)
    SymProb = Prblm.LeastSquaresSymmetric(model_vec_sym, data_vec_sym, MatMultSym)  # ,minBound=low_bound)
    SLCG = SymLCGsolver(Stop)
    # SLCG.setDefaults(iter_sampling=5,save_obj=True,save_res=True,save_grad=True,save_model=True,prefix="test")
    SLCG.run(SymProb,verbose=True)
    print('CG result: \t',SymProb.model.getNdArray())

    # Testing LCG solving A'A
    SymProb_squared = Prblm.LeastSquaresSymmetric(model_vec_sym, MatMultSym * data_vec_sym, MatMultSym * MatMultSym)
    SLCG.run(SymProb_squared, verbose=True)
    print('CG result squared: \t', SymProb_squared.model.getNdArray())


    # Testing LCG from scipy
    # b = data_vec_sym.getNdArray()
    # scipy_res = cg(A, b)[0]
    # print('scipy result: \t', scipy_res)
    # quit(0)

    # Testing preconditioned CG
    SymProbPrec = Prblm.LeastSquaresSymmetric(model_vec_sym, data_vec_sym, MatMultSym, prec=Prec)
    # SLCG.run(SymProbPrec,verbose=True)

    # Testing Linear steepest-descent algorithm for symmetric systems
    SymProb1 = Prblm.LeastSquaresSymmetric(model_vec_sym, data_vec_sym, MatMultSym)
    SLSD = SymLCGsolver(Stop, steepest=True)
    SLSD.setDefaults(iter_sampling=100)
    # SLSD.run(SymProb1)

    # Testing non-linear regularized problem
    non_lin_op = Op.NonLinearOperator(MatMultSym, MatMultSym)
    L2NLRegProb = Prblm.ProblemL2NonLinearReg(model_vec_sym, data_vec_sym, non_lin_op, 0.)
    L2NLRegProb.estimate_epsilon()
    NLCGsolver = NLCG(Stop)
    # NLCGsolver.setDefaults(iter_sampling=5, save_obj=True, save_res=True, save_grad=True, save_model=True, prefix="test_nl")
    # NLCGsolver.run(L2NLRegProb, verbose=True)

    # Testing non-linear bounded problem with NLCG
    L2NLProb = Prblm.ProblemL2NonLinear(model_vec_sym, data_vec_sym, non_lin_op, minBound=low_bound)
    NLCGsolver = NLCG(Stop)
    # NLCGsolver.run(L2NLProb,verbose=False)
    # print(L2NLProb.model.arr)

    # Testing non-linear bounded problem with BFGS
    L2NLProb = Prblm.ProblemL2NonLinear(model_vec_sym, data_vec_sym, non_lin_op, minBound=low_bound)
    BFGSsolver = BFGS(Stop)
    # BFGSsolver.run(L2NLProb,verbose=True)
    # print(L2NLProb.model.arr)

    # Bounded problem
    # Creating the bounds
    model_vec_sym.zero()
    # Create L2-norm linear problem
    L2Prob_sym = Prblm.LeastSquares(model_vec_sym, data_vec_sym, MatMultSym, minBound=low_bound)
    # L2Prob_sym = Prblm.ProblemL2Linear(model_vec_sym,data_vec_sym,MatMultSym)
    # Running the solver
    # LCGsolver.run(L2Prob_sym,verbose=False)
    # print(L2Prob_sym.model.arr)

#
