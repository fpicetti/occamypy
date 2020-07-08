#!/usr/bin/env python3
import sys, os

sys.path.insert(0, "../../python")
from pyLinearSolver import LCGsolver
from pyNonLinearSolver import NLCGsolver
from pyNonLinearSolver import LBFGSsolver
import pyVector as Vec
import pyOperator as Op
from pyProblem import ProblemL2VpReg
from pyStopper import BasicStopper
from pyStepper import CvSrchStep as StepperMT
import numpy as np
from sys_util import logger


# Exponential fitting (See Golub and Pereyra, 1973)
# y(a,b;t) = a1 + a2 * exp(-b1*t) + a3 * exp(-b2*t)
# a linear component of the problem
# b non-linear component of the problem

class exponential_nl(Op.Operator):

    def __init__(self, b_vec, t_vec):
        """y(b;a,t) = a1 + a2 * exp(-b1*t) + a3 * exp(-b2*t)"""
        self.setDomainRange(b_vec, t_vec)
        self.t_samples = t_vec.arr
        self.lin_model = None
        return

    def forward(self, add, model, data):
        """Forward non-linear"""
        self.checkDomainRange(model, data)
        a = self.lin_model.arr
        b = model.arr
        if (not add): data.zero()
        for it, time in enumerate(self.t_samples):
            data.arr[it] += a[0] + a[1] * np.exp(-b[0] * time) + a[2] * np.exp(-b[1] * time)
        return

    def set_lin(self, lin_model):
        """Function to set the linear component of the operator"""
        del self.lin_model
        self.lin_model = lin_model.clone()
        return


class exponential_nl_jac(Op.Operator):

    def __init__(self, b_vec, t_vec):
        """dy(b;a,t) = -a2 * t * exp(-b1*t) * db1 - a3 * t * exp(-b2*t) * db2"""
        self.setDomainRange(b_vec, t_vec)
        self.t_samples = t_vec.arr
        self.lin_model = None
        self.background = None
        return

    def forward(self, add, model, data):
        """Forward linearized"""
        self.checkDomainRange(model, data)
        a = self.lin_model.arr
        b0 = self.background.arr
        db = model.arr
        if (not add): data.zero()
        for it, time in enumerate(self.t_samples):
            data.arr[it] -= (a[1] * time * np.exp(-b0[0] * time) * db[0] + a[2] * time * np.exp(-b0[1] * time) * db[1])
        return

    def adjoint(self, add, model, data):
        """Adjoint linearized"""
        self.checkDomainRange(model, data)
        a = self.lin_model.arr
        b0 = self.background.arr
        db = model.arr
        if (not add): model.zero()
        for it, time in enumerate(self.t_samples):
            db[0] -= a[1] * time * np.exp(-b0[0] * time) * data.arr[it]
            db[1] -= a[2] * time * np.exp(-b0[1] * time) * data.arr[it]
        return

    def set_background(self, model):
        """Function to set the the model on which the operator is linearized"""
        del self.background
        self.background = model.clone()
        return

    def set_lin(self, lin_model):
        """Function to set the linear component of the operator"""
        del self.lin_model
        self.lin_model = lin_model.clone()
        return


class exponential_lin(Op.Operator):

    def __init__(self, a_vec, t_vec):
        """y(b;a,t) = a1 + a2 * exp(-b1*t) + a3 * exp(-b2*t)"""
        self.setDomainRange(a_vec, t_vec)
        self.t_samples = t_vec.arr
        self.nl_model = None
        return

    def forward(self, add, model, data):
        """Forward linear"""
        self.checkDomainRange(model, data)
        b = self.nl_model.arr
        a = model.arr
        if (not add): data.zero()
        for it, time in enumerate(self.t_samples):
            data.arr[it] += a[0] + a[1] * np.exp(-b[0] * time) + a[2] * np.exp(-b[1] * time)
        return

    def adjoint(self, add, model, data):
        """Adjoint linear"""
        self.checkDomainRange(model, data)
        b = self.nl_model.arr
        a = model.arr
        if (not add): model.zero()
        for it, time in enumerate(self.t_samples):
            a[0] += data.arr[it]
            a[1] += np.exp(-b[0] * time) * data.arr[it]
            a[2] += np.exp(-b[1] * time) * data.arr[it]
        return

    def set_nl(self, nl_model):
        """Function to set the non-linear component of the operator"""
        del self.nl_model
        self.nl_model = nl_model.clone()
        return


if __name__ == '__main__':
    # Exponential fitting example
    time_vec = Vec.VectorIC(np.linspace(0., 2., 200))
    a_true = Vec.VectorIC(np.array([10.0, 21.0, 5.0]))
    b_true = Vec.VectorIC(np.array([0.5, 1.5]))
    data_true = time_vec.clone()
    # Generating true data
    expon_nl = exponential_nl(b_true, time_vec)
    expon_nl.set_lin(a_true)
    expon_nl.forward(False, b_true, data_true)
    # Creating VP operator
    expon_nl_jac = exponential_nl_jac(b_true, time_vec)
    exp_nl_op = Op.NonLinearOperator(expon_nl, expon_nl_jac, expon_nl_jac.set_background)
    expon_lin = exponential_lin(a_true, time_vec)
    exp_vp_op = Op.VpOperator(exp_nl_op, expon_lin, expon_lin.set_nl, expon_nl_jac.set_lin,
                              set_lin=expon_nl.set_lin)
    # Creating VP inversion problem
    a_init = a_true.clone()
    a_init.arr = np.array([5.0, 20.0, 3.5])
    b_init = b_true.clone()
    b_init.arr = np.array([0.4, 1.2])
    # Create stopper
    niter = 500
    # Create solver
    LCG = LCGsolver(BasicStopper(niter=niter), logger=logger("Lintest.txt"))
    # LCG.setDefaults(prefix="lin_inv/test",save_obj=True,save_model=True)
    VPproblem = ProblemL2VpReg(b_init, a_init, exp_vp_op, data_true, LCG)
    # Instantiating NLCG solver
    CvStep = StepperMT(gtol=0.1)
    NLCG = NLCGsolver(BasicStopper(niter=niter), stepper=CvStep, logger=logger("VP_NLCG_log.txt"))
    # NLCGsolver = NLCGsolver(BasicStopper(niter=niter),logger=logger("VP_NLCG_log.txt"))
    NLCG.setDefaults()

    # Intial step-length value
    # NLCGsolver.stepper.alpha=0.5
    NLCG.run(VPproblem, verbose=True)
    print("NLCG a optimal: ", VPproblem.lin_model.arr)
    print("NLCG b optimal: ", VPproblem.model.arr)

    # Testing BFGS
    # BFGSsolver = LBFGSsolver(Stopper.BasicStopper(niter=niter),logger=logger("VP_BFGS_log.txt"))
    BFGS = LBFGSsolver(BasicStopper(niter=niter))  # , logger=logger("VP_BFGS_log.txt"))
    VPproblem = ProblemL2VpReg(b_init, a_init, exp_vp_op, data_true, LCG)
    BFGS.run(VPproblem, verbose=True)
    print("BFGS a optimal: ", VPproblem.lin_model.arr)
    print("BFGS b optimal:", VPproblem.model.arr)

    # Testing regularization term by adding the same problem in the regularization term
    VPproblemReg = ProblemL2VpReg(b_init, a_init, exp_vp_op, data_true, LCG, h_op_reg=exp_vp_op,
                                  epsilon=1.0, data_reg=data_true)
# VPproblemReg.estimate_epsilon(verbose=True)
# NLCG.stepper.alpha=0.25 #Resetting initial step length value
# NLCG.run(VPproblemReg,verbose=True)
# print("a optimal",VPproblemReg.lin_model.arr)
# print("b optimal",VPproblemReg.model.arr)
