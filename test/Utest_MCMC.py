#!/usr/bin/env python3
import sys

sys.path.insert(0, "../../python")
import pyVector as Vec
from pyNonLinearSolver import MCMCsolver as MCMC
from pyStopper import SamplingStopper
import pyProblem as Prblm
import numpy as np
# Plotting library
import matplotlib.pyplot as plt


class multi_gauss_prblm(Prblm.Problem):
    """
	   Objective function containing two Gaussian functions at two different centers in the 2D plane
	"""

    def __init__(self, mu1, mu2, sigma1, sigma2):
        """
        Objective function containing two gaussian functions with mu1, mu2, sigma1, and sigma2 parameters
        :param mu1: 1D array - mean of the gaussian function 1
        :param mu2: 1D array - mean of the gaussian function 2
        :param sigma1: 2D array - variance matrix of gaussian function 1
        :param sigma2: 2D array - variance matrix of gaussian function 2
        """
        # Setting the bounds (if any)
        super(multi_gauss_prblm, self).__init__(None, None)
        # Gaussian parameters
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1_inv = np.linalg.inv(sigma1)
        self.sigma2_inv = np.linalg.inv(sigma2)
        self.scale1 = 0.5 / (np.sqrt(np.linalg.det(sigma1) * (2.0 * np.pi) ** self.mu1.shape[0]))
        self.scale2 = 0.5 / (np.sqrt(np.linalg.det(sigma2) * (2.0 * np.pi) ** self.mu2.shape[0]))
        # Setting initial model
        self.model = Vec.VectorIC(mu1.shape)
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
        m1 = model.getNdArray() - self.mu1
        m2 = model.getNdArray() - self.mu2
        self.res.getNdArray()[0] = self.scale1 * np.exp(-0.5 * np.dot(m1, np.dot(self.sigma1_inv, m1))) \
                            + self.scale2 * np.exp(-0.5 * np.dot(m2, np.dot(self.sigma2_inv, m2)))
        return self.res


if __name__ == '__main__':
    mu1 = np.array([2.0, 1.5])
    mu2 = np.array([-1.5, -3.0])
    sigma1 = np.array([[1., 3. / 5.], [3. / 5., 2.]])
    sigma2 = np.array([[2., -3. / 5.], [-3. / 5., 1.]])
    prblm = multi_gauss_prblm(mu1, mu2, sigma1, sigma2)
    MCMC1 = MCMC(stopper=SamplingStopper(1000), prop_distr="Uni", max_step=np.array([0.1,0.2]), min_step=np.array([-0.1,-0.2]))
    MCMC1.setDefaults(save_obj=True, save_model=True)
    MCMC1.run(prblm, verbose=True)
    plt.plot(MCMC1.obj)
    plt.show()
