# Module containing the definition of inverse problems where the ADMM method is used
from math import isnan
from occamypy import Operator, Vector,
import pyOperator as pyOp
import pyVector as pyVec
from pyLinearSolver import LCGsolver, LSQRsolver
from pySparseSolver import ISTAsolver
from pyProblem import Problem, ProblemL1Lasso, ProblemL2LinearReg, ProblemL2Linear, ProblemLinearReg
from pySolver import Solver
from pySparseSolver import *
from pyStopper import BasicStopper
from sys_util import logger


class ADMM(Solver):
    """Alternate Directions of Multipliers Method (ADMM) for GeneralizedLasso problems"""
    
    # Default class methods/functions
    def __init__(self, stopper, logger=None, niter_linear=5, niter_lasso=5, rho=None,
                 rho_auto=True, rho_ratio=10., rho_weight=2., warm_start=False, linear_solver='CG'):
        """
        Constructor for ADMM Solver
        :param stopper       : stopper object
        :param logger        : logger object
        :param niter_linear  : int; number of iterations for solving the linear problem [5]
        :param niter_lasso   : int; number of iterations for solving the lasso problem [5]
        :param rho           : float; penalty parameter rho (if None it is initialized as 2*gamma+.1)
        :param rho_auto      : bool; update rho automatically
        :param rho_ratio     : float; norm ratio between residuals for updating rho [10]
        :param rho_weight    : float; scaling factor for updating rho [2]
        :param warm_start    : bool; linear solver uses previous solution [False]
        :param linear_solver : str; linear solver to be used [CG, SD, LSQR]
        """
        # Calling parent construction
        super(ADMM, self).__init__()
        
        self.stopper = stopper
        self.logger = logger
        self.stopper.logger = self.logger
        self.niter_linear = niter_linear
        self.niter_lasso = niter_lasso
        self.warm_start = warm_start
        
        if linear_solver == 'CG':
            self.solver_linear = LCGsolver(BasicStopper(niter=self.niter_linear), steepest=False, logger=self.logger)
        elif linear_solver == 'SD':
            self.solver_linear = LCGsolver(BasicStopper(niter=self.niter_linear), steepest=True, logger=self.logger)
        elif linear_solver == 'LSQR':
            self.solver_linear = LSQRsolver(BasicStopper(niter=self.niter_linear), logger=self.logger)
        else:
            raise ValueError('ERROR! Solver has to be CG, SD or LSQR')
        
        self.solver_lasso = ISTAsolver(BasicStopper(niter=self.niter_lasso), fast=True, logger=self.logger)
        
        self.rho = rho  # ADMM penalty parameter
        self.rho_ratio = rho_ratio  # norm ratio between residuals for updating rho
        self.rho_weight = rho_weight  # scaling factor for updating rho
        self.rho_auto = rho_auto  # whether to update rho
        self.primal = None  # primal residual vector (r = A x + B z - c)
        self.dual = None  # dual residual vector (s = rho A.H B r)
        
        # print formatting
        self.iter_msg = "iter = %s, obj = %.5e, df_obj = %.2e, reg_obj = %.2e, resnorm = %.2e"
    
    def __del__(self):
        print('Destructor called, ADMM deleted')
    
    def update_rho(self):
        """update penalty parameter rho as suggested in boyd2010distributed (3.13)"""
        if self.primal.norm(2) > self.rho_ratio * self.dual.norm(2):
            self.rho = self.rho * self.rho_weight
        elif self.dual.norm(2) > self.rho_ratio * self.primal.norm(2):
            self.rho = self.rho / self.rho_weight
        else:
            pass
    
    def init_rho(self, gamma):
        self.rho = 2 * gamma + .1
    
    def run(self, problem, verbose=False, inner_verbose=False, restart=False, initial_guess=None):
        
        assert type(problem) == ProblemLinearReg, 'problem has to be a ProblemLinearReg'
        if problem.nregsL1 == 0:
            raise ValueError('ERROR! Provide at least one L1 regularizer!')
        
        self.create_msg = verbose or self.logger
        
        # I want to set dfw=1, so:
        gamma = max(problem.epsL1)
        
        # A is the Vstack of the L1 reg operators, scaled by their respective eps
        # we divide by gamma as gamma becomes the lambda value for the FISTA problem
        A = problem.regL1_op * [e / gamma for e in problem.epsL1]
        
        # initialize all others variables
        if self.rho is None:
            self.init_rho(gamma)
        
        admm_mdl = problem.model.clone().zero() if initial_guess is None else initial_guess.clone()
        y = A.range.clone().zero()
        u = y.clone()
        self.dual = A.domain.clone().zero()
        
        if restart:
            self.restart.read_restart()
            outer_iter = self.restart.retrieve_parameter("iter")
            initial_obj_value = self.restart.retrieve_parameter("obj_initial")
            admm_mdl = self.restart.retrieve_vector("admm_mdl")
            if self.create_msg:
                msg = "Restarting previous solver run from: %s" % self.restart.restart_folder
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog(msg)
        else:
            outer_iter = 0
            if self.create_msg:
                msg = 90 * '#' + '\n'
                msg += "\t\t\t\t\tADMM ALGORITHM log file\n\n"
                msg += "\tRestart folder: %s\n" % self.restart.restart_folder
                msg += "\tModeling Operator:\t\t%s\n" % problem.op
                if problem.nregsL2 != 0:
                    msg += "\tL2 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL2_op.ops]) + "\n"
                    msg += "\tL2 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL2]) + "\n"
                msg += "\tL1 Regularizer ops:\t\t" + ", ".join(["%s" % op for op in problem.regL1_op.ops]) + "\n"
                msg += "\tL1 Regularizer weights:\t" + ", ".join(["{:.2e}".format(e) for e in problem.epsL1]) + "\n"
                msg += "\tPenalty parameter:\t\t%.2e\n" % self.rho
                msg += 90 * '#' + '\n'
                if verbose:
                    print(msg.replace(" log file", ""))
                if self.logger:
                    self.logger.addToLog(msg)
        
        # Main iteration loop
        while True:
            obj0 = problem.get_obj(admm_mdl)
            
            if outer_iter == 0:
                initial_obj_value = obj0
                self.restart.save_parameter("obj_initial", initial_obj_value)
                if self.create_msg:
                    msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                           obj0,
                                           problem.obj_terms[0],
                                           obj0 - problem.obj_terms[0],
                                           problem.get_rnorm(admm_mdl))
                    if verbose:
                        print(msg)
                    if self.logger:
                        self.logger.addToLog("\n" + msg)
                
                if isnan(obj0):
                    raise ValueError("Objective function values NaN!")
            
            if obj0 == 0:
                print("Objective function is 0!")
                break
            
            self.save_results(outer_iter, problem, force_save=False)
            
            # 1) update x
            # Linear Problem:       1/2 | Op x - d| + epsL2   | R2 x -  dr  |
            #                                         rho/2   | A  x - (y-u)|
            regL2_op_scaled_list = [problem.epsL2[i] * problem.regL2_op.ops[i] for i in range(problem.nregsL2)]
            regA_op_scaled_list = [self.rho / 2 * A.ops[i] for i in range(A.n)]
            reg_op = pyOp.Vstack(
                pyOp.Vstack(regL2_op_scaled_list) if len(regL2_op_scaled_list) != 0 else None,
                pyOp.Vstack(regA_op_scaled_list) if len(regA_op_scaled_list) != 0 else None,
            )
            prior = pyVec.superVector(problem.dataregsL2, y.clone().scaleAdd(u, 1., -1.))
            linear_problem = ProblemL2LinearReg(
                model=admm_mdl.clone().zero() if not self.warm_start else admm_mdl.clone(),
                data=problem.data,
                op=problem.op,
                reg_op=reg_op,  # pyOp.Vstack(problem.regL2_op, A),
                epsilon=1.,  # problem.epsL2 + [self.rho / 2] * A.n,
                prior_model=prior,
                minBound=problem.minBound, maxBound=problem.maxBound, boundProj=problem.boundProj
            )
            if outer_iter == 0 and initial_guess is not None:
                linear_problem.model = initial_guess.clone()
            
            self.solver_linear.run(linear_problem, verbose=inner_verbose)
            
            admm_mdl = linear_problem.model.clone()
            
            # 2) update y
            # lasso problem: rho/2 | A x - z + u|_2^2 + gamma | y |_1
            # this means to solve: 1/2 | I y - (Ax + u)| + gamma/rho | y |_1
            Ax = A * admm_mdl
            lasso_problem = ProblemL1Lasso(  # TODO it stops at the second iteration
                model=y.clone().zero(),
                data=Ax.clone() + u,
                op=pyOp.Identity(y),
                op_norm=1.,
                lambda_value=gamma / self.rho,
                minBound=problem.minBound, maxBound=problem.maxBound, boundProj=problem.boundProj
            )
            self.solver_lasso.setDefaults()
            self.solver_lasso.run(lasso_problem, verbose=inner_verbose)
            y = lasso_problem.model.clone()
            
            # 3) update penalty parameter and scaled dual variable
            self.primal = Ax.clone() - y
            
            A.adjoint(False, self.dual, self.primal)
            self.dual.scale(-self.rho)
            self.update_rho()
            u.__add__(self.primal).scale(1 / self.rho)  # u = (u + r)/rho
            
            outer_iter += 1
            # check objective function
            obj1 = problem.get_obj(admm_mdl)
            # if obj1 >= obj0:
            #     msg = "Objective function didn't reduce, will terminate solver:\n\t" \
            #           "obj_new = %.2e\tobj_cur = %.2e" % (obj1, obj0)
            #     if verbose:
            #         print(msg)
            #     if self.logger:
            #         self.logger.addToLog(msg)
            #     break
            
            # iteration info
            if self.create_msg:
                msg = self.iter_msg % (str(outer_iter).zfill(self.stopper.zfill),
                                       obj1,
                                       problem.obj_terms[0],
                                       obj1 - problem.obj_terms[0],
                                       problem.get_rnorm(admm_mdl))
                if verbose:
                    print(msg)
                if self.logger:
                    self.logger.addToLog("\n" + msg)
            
            # saving in case of restart
            self.restart.save_parameter("iter", outer_iter)
            self.restart.save_vector("admm_mdl", admm_mdl)
            
            if self.stopper.run(problem, outer_iter, initial_obj_value, verbose):
                break
        
        # writing last inverted model
        self.save_results(outer_iter, problem, model=None, force_save=False, force_write=False)
        
        # ending message and log file
        if self.create_msg:
            msg = 90 * '#' + '\n'
            msg += "\t\t\t\t\tADMM ALGORITHM log file end\n"
            msg += 90 * '#'
            if verbose:
                print(msg.replace(" log file", ""))
            if self.logger:
                self.logger.addToLog("\n" + msg)
        
        # Clear restart object
        self.restart.clear_restart()


def main():
    from sys import path
    path.insert(0, '.')
    import numpy as np
    import matplotlib.pyplot as plt
    # plt.style.use('ggplot')
    import pyNpOperator
    from pyProblem import ProblemLinearReg
    from pySparseSolver import SplitBregmanSolver
    
    PLOT = True
    EXAMPLE = 'monarch'  # must be noisy, gaussian1D, gaussian2D, medical or monarch
    
    if EXAMPLE == 'noisy':
        # data examples
        np.random.seed(1)
        nx = 101
        x = pyVec.vectorIC((nx,)).zero()
        x.getNdArray()[:nx // 2] = 10
        x.getNdArray()[nx // 2:3 * nx // 4] = -5
        
        Iop = pyOp.Identity(x)
        TV = pyNpOperator.FirstDerivative(x)
        L = pyNpOperator.SecondDerivative(x)
        
        n = x.clone()
        n.getNdArray()[:] = np.random.normal(0, 1.0, nx)
        y = Iop * (x.clone() + n)
        
        derivative = TV * x
        
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(derivative.getNdArray(), '.b', lw=2, label='∂x')
        #     plt.legend()
        #     plt.title('Model, Data and Derivative')
        #     plt.show()
        #
        # # CG solver
        # problemLS = ProblemL2Linear(x.clone().zero(), y, Iop)
        # CG = LCGsolver(BasicStopper(niter=30))
        # CG.run(problemLS, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLS.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('Least-Squares CG')
        #     plt.show()
        #
        # # LSQR solver
        # problemLSQR = ProblemL2Linear(x.clone().zero(), y, Iop)
        # LSQR = LSQRsolver(BasicStopper(niter=30))
        # LSQR.run(problemLSQR, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLSQR.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('Least-Squares LSQR')
        #     plt.show()
        #
        # # CG solver with L2 regularization
        # problemLSR = ProblemL2LinearReg(x.clone().zero(), y, Iop, np.sqrt(50), L)
        # CG = LCGsolver(BasicStopper(niter=30))
        # CG.run(problemLSR, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLSR.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('CG with Laplacian reg')
        #     plt.show()
        #
        # # LSQR solver with L2 regularization
        # problemLSR_1 = ProblemL2LinearReg(x.clone().zero(), y, Iop, np.sqrt(50), L)
        # LSQR = LSQRsolver(BasicStopper(niter=30))
        # LSQR.run(problemLSR_1, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLSR_1.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('LSQR with Laplacian reg')
        #     plt.show()
        #
        # # FISTA
        # problemFISTA = ProblemL1Lasso(x.clone().zero(), y, Iop, lambda_value=1, op_norm=1)
        # FISTA = ISTAsolver(BasicStopper(niter=300), fast=True)
        # FISTA.run(problemFISTA, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemFISTA.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('FISTA inversion')
        #     plt.show()
        
        # # SplitBregman
        problemSB = ProblemLinearReg(x.clone().zero(), y, Iop, regsL1=TV, epsL1=3.0)
        SB = SplitBregmanSolver(BasicStopper(niter=100), lambd=0.03, niter_inner=1, niter_solver=20,
                                linear_solver='LSQR', breg_weight=1.,
                                warm_start=True)  # , logger=logger("test_SB1.txt"))
        SB.setDefaults(save_obj=True)
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=1, label='x')
            plt.plot(y.getNdArray(), '.k', label='y=x+n')
            plt.plot(derivative.getNdArray(), ':k', lw=1, label='∂x')
            plt.plot(problemSB.model.getNdArray(), 'r', lw=2, label='x_inv')
            plt.plot((TV * problemSB.model).getNdArray(), ':r', lw=2, label='∂(x_inv)')
            plt.legend()
            plt.title('SB inversion')
            plt.show()
            # Objective function convergence
            plt.figure(figsize=(5, 4))
            plt.plot(np.log10(SB.obj / SB.obj[0]), 'r', lw=1, label='SplitBregman')
            obj_true = problemSB.get_obj(x)
            plt.plot([np.log10(obj_true / SB.obj[0])] * len(SB.obj), 'k--', lw=1, label='true solution obj value')
            plt.legend()
            plt.title('Convergence curve')
            plt.show()
        
        # ADMM
        # problemADMM = ProblemLinearReg(x.clone().zero(), y, Iop, regsL1=TV, epsL1=3.)
        #
        # ADMM = ADMMsolver(BasicStopper(niter=30), niter_linear=10, niter_lasso=50)
        # ADMM.run(problemADMM, verbose=True, inner_verbose=False)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(derivative.getNdArray(), ':k', lw=1, label='∂x')
        #     plt.plot(problemADMM.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.plot((TV * problemADMM.model).getNdArray(), ':r', lw=2, label='∂(x_inv)')
        #     plt.legend()
        #     plt.title('ADMM inversion')
        #     plt.show()
    
    if EXAMPLE == 'gaussian1D':
        # data examples
        np.random.seed(1)
        nx = 101
        x = pyVec.vectorIC((nx,)).zero()
        x.getNdArray()[:nx // 2] = 10
        x.getNdArray()[nx // 2:3 * nx // 4] = -5
        
        Op = pyNpOperator.GaussianFilter(x, sigma=(5))
        TV = pyNpOperator.FirstDerivative(x)
        L = pyNpOperator.SecondDerivative(x)
        y = Op * x
        dx = TV * x
        
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=1, label='x')
            plt.plot(y.getNdArray(), 'b', label='y=Gx')
            plt.plot(dx.getNdArray(), '.k', lw=2, label='∂x')
            plt.legend()
            plt.title('Model, Data and Derivative')
            plt.show()
        
        # # CG solver
        problemLS = ProblemL2Linear(x.clone().zero(), y, Op)
        CG = LCGsolver(BasicStopper(niter=500))
        CG.run(problemLS)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', label='x')
            plt.plot(y.getNdArray(), 'b', label='y=Gx')
            plt.plot(problemLS.model.getNdArray(), 'r', label='x_inv')
            plt.legend()
            plt.title('Least-Squares CG')
            plt.show()
        
        # LSQR solver
        problemLSQR = ProblemL2Linear(x.clone().zero(), y, Op)
        LSQR = LSQRsolver(BasicStopper(niter=500))
        LSQR.run(problemLSQR, verbose=True)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', label='x')
            plt.plot(y.getNdArray(), 'b', label='y=Gx')
            plt.plot(problemLSQR.model.getNdArray(), 'r', label='x_inv')
            plt.legend()
            plt.title('Least-Squares LSQR')
            plt.show()
        #
        # CG solver with L2 regularization
        problemLSR = ProblemL2LinearReg(x.clone().zero(), y, Op, 1, L)
        CG = LCGsolver(BasicStopper(niter=500))
        CG.run(problemLSR, verbose=True)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', label='x')
            plt.plot(y.getNdArray(), 'b', label='y=Gx')
            plt.plot(problemLSR.model.getNdArray(), 'r', label='x_inv')
            plt.legend()
            plt.title('CG with Laplacian reg')
            plt.show()
        #
        # # LSQR solver with L2 regularization
        # problemLSR_1 = ProblemL2LinearReg(x.clone().zero(), y, Iop, np.sqrt(50), L)
        # LSQR = LSQRsolver(BasicStopper(niter=30))
        # LSQR.run(problemLSR_1, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.plot(x.getNdArray(), 'k', lw=1, label='x')
        #     plt.plot(y.getNdArray(), '.k', label='y=x+n')
        #     plt.plot(problemLSR_1.model.getNdArray(), 'r', lw=2, label='x_inv')
        #     plt.legend()
        #     plt.title('LSQR with Laplacian reg')
        #     plt.show()
        #
        # FISTA
        problemFISTA = ProblemL1Lasso(x.clone().zero(), y, Op, lambda_value=.5, op_norm=1.05)
        FISTA = ISTAsolver(BasicStopper(niter=300), fast=True)
        FISTA.run(problemFISTA, verbose=True)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', label='x')
            plt.plot(y.getNdArray(), 'b', label='y=Gx')
            plt.plot(problemFISTA.model.getNdArray(), 'r', label='x_inv')
            plt.legend()
            plt.title('FISTA inversion')
            plt.show()
        
        # # SplitBregman
        problemSB = ProblemLinearReg(x.clone().zero(), y, Op, regsL1=TV, epsL1=10.0)
        SB = SplitBregmanSolver(BasicStopper(niter=100), lambd=0.03, niter_inner=1, niter_solver=20,
                                linear_solver='CG', breg_weight=1.,
                                warm_start=False)  # , logger=logger("test_SB1.txt"))
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', label='x')
            plt.plot(y.getNdArray(), 'b', label='y=Gx')
            plt.plot(dx.getNdArray(), ':k', lw=1, label='∂x')
            plt.plot(problemSB.model.getNdArray(), 'r', label='x_inv')
            plt.plot((TV * problemSB.model).getNdArray(), ':r', lw=1, label='∂(x_inv)')
            plt.legend()
            plt.title('SB inversion')
            plt.show()
            # Objective function convergence
            plt.figure(figsize=(5, 4))
            plt.plot(np.log10(SB.obj / SB.obj[0]), 'r', lw=1, label='SplitBregman')
            obj_true = problemSB.get_obj(x)
            plt.plot([np.log10(obj_true / SB.obj[0])] * len(SB.obj), 'k--', lw=1, label='true solution obj value')
            plt.legend()
            plt.title('Convergence curve')
            plt.show()
        
        # ADMM
        problemADMM = ProblemLinearReg(x.clone().zero(), y, Iop, regsL1=TV, epsL1=3.)
        
        ADMM = ADMM(BasicStopper(niter=30), niter_linear=10, niter_lasso=50)
        ADMM.run(problemADMM, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.plot(x.getNdArray(), 'k', lw=1, label='x')
            plt.plot(y.getNdArray(), '.k', label='y=x+n')
            plt.plot(derivative.getNdArray(), ':k', lw=1, label='∂x')
            plt.plot(problemADMM.model.getNdArray(), 'r', lw=2, label='x_inv')
            plt.plot((TV * problemADMM.model).getNdArray(), ':r', lw=2, label='∂(x_inv)')
            plt.legend()
            plt.title('ADMM inversion')
            plt.show()
    
    elif EXAMPLE == 'gaussian2D':
        x = pyVec.vectorIC(np.empty((301, 601))).set(0)
        x.getNdArray()[150, 300] = 1.0
        # x.getNdArray()[100, 200] = -5.0
        # x.getNdArray()[280, 400] = 1.0
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(x.getNdArray()), plt.colorbar()
            plt.title('Model')
            plt.show()
        
        G = pyNpOperator.GaussianFilter(x, [25, 15])
        y = G * x
        # y.scale(1./y.norm())
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(y.getNdArray()), plt.colorbar()
            plt.title('Data')
            plt.show()
        
        # CG solver
        problemLS = ProblemL2Linear(x.clone().zero(), y, G)
        CG = LCGsolver(BasicStopper(niter=30))
        CG.setDefaults()
        CG.run(problemLS, verbose=True)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemLS.model.getNdArray()), plt.colorbar()
            plt.title('CG, %d its' % CG.stopper.niter)
            plt.show()
        
        # FISTA
        problemFISTA = ProblemL1Lasso(x.clone().zero(), y, G, lambda_value=1000, op_norm=1.)
        FISTA = ISTAsolver(BasicStopper(niter=1000), fast=True)
        FISTA.setDefaults()
        FISTA.run(problemFISTA, verbose=True)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemFISTA.model.getNdArray()), plt.colorbar()
            plt.title(r'FISTA, $\lambda$=%.2e, %d its' % (problemFISTA.lambda_value, FISTA.stopper.niter))
            plt.show()
        
        # SplitBregman
        I = pyOp.Identity(x)
        problemSB = ProblemLinearReg(x.clone().zero(), y, G, regsL1=I, epsL1=10.)
        
        SB = SplitBregmanSolver(BasicStopper(niter=50), niter_inner=3, niter_solver=30,
                                linear_solver='LSQR', breg_weight=1., use_prev_sol=False)
        SB.setDefaults()
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemSB.model.getNdArray()), plt.colorbar()
            plt.title('SplitBregman')
            plt.show()
        
        # ADMM
        problemADMM = ProblemLinearReg(x.clone().zero(), y, G, regsL1=I, epsL1=2.)
        
        ADMM = ADMM(BasicStopper(niter=10), niter_linear=30, niter_lasso=100)
        ADMM.setDefaults()
        ADMM.run(problemADMM, verbose=True, inner_verbose=True)
        if PLOT:
            plt.figure(figsize=(6, 3))
            plt.imshow(problemADMM.model.getNdArray()), plt.colorbar()
            plt.title('ADMM')
            plt.show()
    
    elif EXAMPLE == 'medical':
        x = pyVec.vectorIC(np.load('../testdata/shepp_logan_phantom.npy', allow_pickle=True).astype(np.float32))
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(x.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title('Model')
            plt.show()
        
        # nh = [5, 10]
        # hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
        # hx = np.exp(-0.03 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
        # hz /= np.trapz(hz)  # normalize the integral to 1
        # hx /= np.trapz(hx)  # normalize the integral to 1
        # h = hz[:, np.newaxis] * hx[np.newaxis, :]
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.imshow(h, aspect='equal'), plt.colorbar()
        #     plt.title('Blurring Kernel')
        #     plt.show()
        # Blurring = pyNpOperator.ConvNDscipy(model=x, kernel=pyVec.vectorIC(h))
        Blurring = pyNpOperator.GaussianFilter(x, [3, 5])
        
        y = Blurring * x
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(y.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title('Data')
            plt.show()
        
        # CG solver
        problemLS = ProblemL2Linear(x.clone().zero(), y, Blurring, minBound=x.clone().set(0.0))
        CG = LCGsolver(BasicStopper(niter=400))
        CG.run(problemLS, verbose=True)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemLS.model.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title('CG, %d iter' % CG.stopper.niter)
            plt.show()
        
        # FISTA
        # problemFISTA = ProblemL1Lasso(x.clone().zero(), y, Blurring, lambda_value=1, op_norm=1.25)
        # FISTA = ISTAsolver(BasicStopper(niter=100), fast=True)
        # FISTA.run(problemFISTA, verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.imshow(problemFISTA.model.getNdArray(), cmap='bone'), plt.colorbar()
        #     plt.title(r'FISTA, $\lambda$=%.2e, %d iter'
        #               % (problemFISTA.lambda_value, FISTA.stopper.niter))
        #     plt.show()
        
        # SplitBregman
        # the gradient of the image is 6e3
        D = pyNpOperator.TotalVariation(x)
        I = pyOp.Identity(x)
        
        problemSB = ProblemLinearReg(x.clone().zero(), y, Blurring, regsL1=D, epsL1=0.00005,
                                     minBound=x.clone().set(0.0))
        
        SB = SplitBregmanSolver(BasicStopper(niter=300), lambd=0.1, niter_inner=1, niter_solver=30,
                                linear_solver='LSQR', breg_weight=1., warm_start=True)
        SB.setDefaults(save_obj=True)
        SB.run(problemSB, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemSB.model.getNdArray(), cmap='bone', vmin=x.min(), vmax=x.max()), plt.colorbar()
            plt.title(r'SB TV, $\varepsilon=%.2e$, %d iter'
                      % (problemSB.epsL1[0], SB.stopper.niter))
            plt.show()
            plt.figure(figsize=(5, 4))
            plt.plot(np.log10(SB.obj / SB.obj[0]), 'r', lw=1, label='SplitBregman')
            obj_true = problemSB.get_obj(x)
            plt.plot([np.log10(obj_true / SB.obj[0])] * len(SB.obj), 'k--', lw=1, label='true solution obj value')
            plt.legend()
            plt.title('Convergence curve')
            plt.show()
        
        # ADMM
        # problemADMM = ProblemLinearReg(x.clone().zero(), y, Blurring,
        #                              regsL1=D, epsL1=.1)
        #
        # ADMM = ADMMsolver(BasicStopper(niter=10), niter_linear=30, niter_lasso=10)
        # ADMM.setDefaults(save_obj=True, save_model=True)
        # ADMM.run(problemADMM, verbose=True, inner_verbose=True)
        # if PLOT:
        #     plt.figure(figsize=(5, 4))
        #     plt.imshow(problemADMM.model.getNdArray(), cmap='bone'), plt.colorbar()
        #     plt.title(r'ADMM TV, $\varepsilon=%.2e$, %d iter'
        #               % (problemADMM.epsL1[0], ADMM.stopper.niter))
        #     plt.show()
    
    elif EXAMPLE == 'monarch':
        x = pyVec.vectorIC(np.load('../testdata/monarch.npy', allow_pickle=True).astype(np.float32))
        np.random.seed(12345)
        sigma = 0.05
        y = x.clone()
        y.getNdArray()[:] += np.random.normal(0.0, sigma, y.shape)
        Op = pyOp.Identity(x)
        TV = pyNpOperator.TotalVariation(x)
        
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(x.arr, cmap='gray'), plt.colorbar()
            plt.title('Model')
            plt.show()
            
            plt.figure(figsize=(5, 4))
            plt.imshow(y.arr, cmap='gray'), plt.colorbar()
            plt.title('Data, std=%.2f' % sigma)
            plt.show()
        
        problemADMM = ProblemLinearReg(x.clone().zero(), y, Op, regsL1=TV, epsL1=.04)
        
        ADMM = ADMM(BasicStopper(niter=200), niter_linear=30, niter_lasso=10)
        ADMM.run(problemADMM, verbose=True, inner_verbose=False)
        if PLOT:
            plt.figure(figsize=(5, 4))
            plt.imshow(problemADMM.model.getNdArray(), cmap='gray'), plt.colorbar()
            plt.title(r'ADMM TV, $\varepsilon=%.2e$, %d iter'
                      % (problemADMM.epsL1[0], ADMM.stopper.niter))
            plt.show()
    
    else:
        raise ValueError("EXAMPLE has to be one of noisy, gaussian, medical or monarch")
    
    return 0


if __name__ == '__main__':
    main()
