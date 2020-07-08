#!/usr/bin/env python3
import sys

sys.path.insert(0, "../../python")
import pyVector as Vec
import pyOperator as Op
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as BFGS
import pyProblem as Prblm
from pyStopper import BasicStopper as Stopper
import pyStepper as Stepper
import numpy as np
from sys_util import logger
# Plotting library
import matplotlib.pyplot as plt


class Rosenbrock_prblm(Prblm.Problem):
    """
	   Rosenbrock function inverse problem
	   f(x,y) = (1 - x)^2 + 100*(y -x^2)^2
	   m = [x y]'
	   res = objective function value
	"""

    def __init__(self, x_initial, y_initial):
        """Constructor of linear problem"""
        # Setting the bounds (if any)
        super(Rosenbrock_prblm, self).__init__(None, None)
        # Setting initial model
        self.model = Vec.VectorIC(np.array((x_initial, y_initial)))
        self.dmodel = self.model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Residual vector
        self.res = Vec.VectorIC(np.array((0.,)))
        # Dresidual vector
        self.dres = self.res.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = False
        return

    def objf(self, model):
        """Objective function computation"""
        m = model.arr  # Getting ndArray of the model
        obj = self.res.arr[0]
        return obj

    def resf(self, model):
        """Residual function"""
        m = model.arr  # Getting ndArray of the model
        self.res.arr[0] = (1.0 - m[0]) * (1.0 - m[0]) + 100.0 * (m[1] - m[0] * m[0]) * (m[1] - m[0] * m[0])
        return self.res

    def gradf(self, model, res):
        """Gradient computation"""
        m = model.arr  # Getting ndArray of the model
        self.grad.arr[0] = - 2.0 * (1.0 - m[0]) - 400.0 * m[0] * (m[1] - m[0] * m[0])
        self.grad.arr[1] = 200.0 * (m[1] - m[0] * m[0])
        return self.grad

    def dresf(self, model, dmodel):
        """Linear variation of the objective function"""
        m = model.arr  # Getting ndArray of the model
        dm = dmodel.arr  # Getting ndArray of the model
        self.dres.arr[0] = (- 2.0 * (1.0 - m[0]) - 400.0 * m[0] * (m[1] - m[0] * m[0])) * dm[0] + (
                200.0 * (m[1] - m[0] * m[0])) * dm[1]
        return self.dres


if __name__ == '__main__':
    x_init = -1.0
    y_init = -1.0
    # Testing solver on Rosenbrock function
    Ros_prob = Rosenbrock_prblm(x_init, y_init)
    # Create stopper
    niter = 500
    # Stop  = Stopper.BasicStopper(niter=niter,tolr=1e-32,tolg=1e-32,tolobjchng=1e-6)
    Stop = Stopper(niter=niter, tolr=1e-32, tolg=1e-32)
    # Create solver
    NLCGsolver = NLCG(Stop, logger=logger("Rosenbrock_NLCG_log.txt"))
    # NLCGsolver.setDefaults(save_obj=True,save_model=True,prefix="NLCG_ros",iter_sampling=1)
    NLCGsolver.stepper.eval_parab = True
    NLCGsolver.run(Ros_prob, verbose=True)
    print("optimal NLCG x: ", Ros_prob.model.arr[0])
    print("optimal NLCG y: ", Ros_prob.model.arr[1])
    # plt.plot(NLCGsolver.obj)
    # plt.show()

    # Testing Steepest-descent method
    Ros_prob = Rosenbrock_prblm(x_init, y_init)
    NLSDsolver = NLCG(Stop, beta_type="SD", logger=logger("Rosenbrock_NLSD_log.txt"))
    # NLSDsolver.setDefaults(save_obj=True,save_model=True,prefix="NLSD_ros",iter_sampling=1)
    # NLSDsolver.run(Ros_prob,verbose=True)
    print("optimal NLSD x: ", Ros_prob.model.arr[0])
    print("optimal NLSD y: ", Ros_prob.model.arr[1])

    # Testing BFGS algorithm
    ParabStep = Stepper.ParabolicStep()
    Ros_prob = Rosenbrock_prblm(x_init, y_init)
    BFGSsolver = BFGS(Stop, stepper=ParabStep, logger=logger("Rosenbrock_BFGS_log.txt"))
    # BFGSsolver.setDefaults(save_obj=True,save_model=True,prefix="BFGSsolver_ros")
    BFGSsolver.stepper.eval_parab = True
    BFGSsolver.run(Ros_prob, verbose=True)
    print("optimal BFGS x: ", Ros_prob.model.arr[0])
    print("optimal BFGS y: ", Ros_prob.model.arr[1])

    # Testing LBFGS algorithm
    Ros_prob = Rosenbrock_prblm(x_init, y_init)
    LBFGSsolver = BFGS(Stop, stepper=ParabStep, m_steps=1, logger=logger("Rosenbrock_LBFGS_log.txt"))
    # LBFGSsolver.setDefaults(save_obj=True,save_model=True,prefix="LBFGSsolver_ros")
    LBFGSsolver.run(Ros_prob, verbose=True)
    print("optimal LBFGS x: ", Ros_prob.model.arr[0])
    print("optimal LBFGS y: ", Ros_prob.model.arr[1])

    # Testing BFGS algorithm using different parabolic stepper
    Ros_prob = Rosenbrock_prblm(x_init, y_init)
    ParStep = Stepper.ParabolicStepConst()
    BFGSsolver2 = BFGS(Stop, stepper=ParStep, logger=logger("Rosenbrock_BFGS_parab_log.txt"))
    # BFGSsolver.setDefaults(save_obj=True,save_model=True,prefix="BFGSsolver_ros")
    BFGSsolver2.run(Ros_prob, verbose=True)
    print("optimal BFGS x: ", Ros_prob.model.arr[0])
    print("optimal BFGS y: ", Ros_prob.model.arr[1])

    # Testing BFGS algorithm using CvSrch stepper
    Ros_prob = Rosenbrock_prblm(x_init, y_init)
    BFGSsolver3 = BFGS(Stop, logger=logger("Rosenbrock_BFGS_CvSrch_log.txt"))
    # BFGSsolver.setDefaults(save_obj=True,save_model=True,prefix="BFGSsolver_ros")
    BFGSsolver3.run(Ros_prob, verbose=True)
    print("optimal BFGS x: ", Ros_prob.model.arr[0])
    print("optimal BFGS y: ", Ros_prob.model.arr[1])

    # Testing BFGS algorithm using CvSrch stepper and running it twice for testing
    Ros_prob = Rosenbrock_prblm(x_init, y_init)
    BFGSsolver4 = BFGS(Stopper(niter=18, tolr=1e-32, tolg=1e-32), logger=logger("Rosenbrock_BFGS_CvSrch_log.txt"))
    # BFGSsolver.setDefaults(save_obj=True,save_model=True,prefix="BFGSsolver_ros")
    BFGSsolver4.run(Ros_prob, verbose=True)
    print("optimal BFGS x: ", Ros_prob.model.arr[0])
    print("optimal BFGS y: ", Ros_prob.model.arr[1])
    BFGSsolver4.run(Ros_prob, verbose=True, keep_hessian=True)
    print("optimal BFGS x: ", Ros_prob.model.arr[0])
    print("optimal BFGS y: ", Ros_prob.model.arr[1])

# Computing the objective function for plotting
# x_samples = np.linspace(-2.0,2.0,1000)
# y_samples = np.linspace(-2.0,2.0,1000)
# obj_ros = Vec.vectorIC(np.zeros((len(x_samples),len(y_samples))))
# obj_ros.ax_info = [[len(y_samples),-2.0,y_samples[1]-y_samples[0],"y"],[len(x_samples),-2.0,x_samples[1]-x_samples[0],"x"]]
# obj_ros_np = obj_ros.getNdArray()
# model_test = Vec.vectorIC(np.array((0.0,0.0)))
# model_test_np = model_test.getNdArray()
# for ix,x_value in enumerate(x_samples):
# 	for iy,y_value in enumerate(y_samples):
# 		model_test_np[0] = x_value
# 		model_test_np[1] = y_value
# 		obj_ros_np[ix,iy]=Ros_prob.get_obj(model_test)
# obj_ros.writeVec("Ros_func.H")

#
