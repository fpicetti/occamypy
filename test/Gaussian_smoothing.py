# Simple self-adjoint Gaussian smoothing operator
import numpy as np
import pyOperator
from scipy.ndimage.filters import gaussian_filter


# Creating Gaussian smoothing operator
class Gauss_smooth(pyOperator.Operator):
    def __init__(self, model, sigmax, sigmaz):
        """
        Gaussian 2D smoothing operator:
        model    = [no default] - vector class; domain vector
        sigmax   = [no default] - float; standard deviation along the x direction
        sigmaz   = [no default] - float; standard deviation along the z direction
        """
        self.setDomainRange(model, model)
        self.sigmax = sigmax
        self.sigmaz = sigmaz
        # Precomputing Gaussian filters
        model_arr = model.getNdArray()
        nz = model_arr.shape[1]
        nx = model_arr.shape[0]
        epsilon = 1e-32  # Cut-off value for removing small filter coefficients
        # Filter on z axis
        arg = np.linspace(0., nz)
        arg = -arg * arg / (sigmaz * sigmaz)
        filterz = np.exp(arg)
        # Removing extremely small values
        filterz = filterz[filterz > epsilon]
        # Symmetrizing filter
        self.filterz = np.append(np.flip(filterz[1:], 0), filterz)
        # Filter on x axis
        arg = np.linspace(0., nx)
        arg = -arg * arg / (sigmax * sigmax)
        filterx = np.exp(arg)
        # Removing extremely small values
        filterx = filterx[filterx > epsilon]
        # Symmetrizing filter
        self.filterx = np.append(np.flip(filterx[1:], 0), filterx)
        # Allocating temporary array for dealing with the boundaries
        self.tmp_array = np.zeros(
            (nx + self.filterx.size - 1, nz + self.filterz.size - 1))
        self.halfx = int(self.filterx.size / 2)
        self.halfz = int(self.filterz.size / 2)
        self.data_tmp = model.clone()
        return
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if add:
            self.data_tmp.copy(data)
        data.zero()
        # Getting Ndarrays
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        nz = model_arr.shape[1]
        nx = model_arr.shape[0]
        self.tmp_array[self.halfx:self.halfx + nx,
        self.halfz:self.halfz + nz] = model_arr[:]
        # Applying filter along z
        for ix in range(nx):
            for iz in range(nz):
                data_arr[ix, iz] = np.dot(
                    self.tmp_array[ix + self.halfx, iz:iz + self.filterz.size],
                    self.filterz)
        # Applying filter along x
        self.tmp_array[self.halfx:self.halfx + nx, self.halfz:self.halfz + nz] = data_arr[
                                                                                 :]
        for ix in range(nx):
            for iz in range(nz):
                data_arr[ix, iz] = np.dot(
                    self.tmp_array[ix:ix + self.filterx.size, iz + self.halfz],
                    self.filterx)
        if add:
            data.scaleAdd(self.data_tmp)
        return
    
    def adjoint(self, add, model, data):
        """Self-adjoint operator"""
        self.forward(add, data, model)
        return


class Gauss_smooth_scipy(pyOperator.Operator):
    def __init__(self, model, sigmax, sigmaz):
        """
        Gaussian 2D smoothing operator using scipy smoothing:
        model    = [no default] - vector class; domain vector
        sigmax   = [no default] - float; standard deviation along the x direction
        sigmaz   = [no default] - float; standard deviation along the z direction
        """
        self.setDomainRange(model, model)
        self.sigmax = sigmax
        self.sigmaz = sigmaz
        self.scaling = 2.0 * np.pi * sigmax * sigmaz
        return
    
    def __str__(self):
        return "GauSmoot"
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        data_arr[:] = self.scaling * gaussian_filter(model_arr, sigma=[self.sigmax, self.sigmaz])
        return
    
    def adjoint(self, add, model, data):
        """Self-adjoint operator"""
        self.forward(add, data, model)
        return
