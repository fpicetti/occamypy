# Operators to perform 2D Eikonal tomography
import numpy as np
import occamypy
from numba import jit
import psutil 
import pykonal
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed
from scipy.ndimage import gaussian_filter


class GaussianFilter(occamypy.Operator):
    def __init__(self, model, sigma):
        """
        Gaussian smoothing operator using scipy smoothing:
        model    = [no default] - vector class; domain vector
        sigma   = [no default] - scalar or sequence of scalars; standard deviation along the model directions
        """
        self.setDomainRange(model, model)
        self.sigma = sigma
        self.scaling = np.sqrt(np.prod(np.array(self.sigma)/np.pi))  # in order to have the max amplitude 1
        return
    
    def __str__(self):
        return "GausFilt"
    
    def forward(self, add, model, data):
        """Forward operator"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Getting Ndarrays
        model_arr = model.getNdArray()
        data_arr = data.getNdArray()
        data_arr += self.scaling * gaussian_filter(model_arr, sigma=self.sigma)
        return
    
    def adjoint(self, add, model, data):
        """Self-adjoint operator"""
        self.forward(add, data, model)
        return

def compute_tt(vel, SouPos, RecPos, dx, dz, data, ishot):
    """Function to compute traveltimes by solving Eikonal equation"""
    nx = vel.shape[0]
    nz = vel.shape[1]
    nRec = RecPos.shape[0]
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = 0.0, 0.0, 0.0
    velocity.node_intervals = 1.0, dx, dz
    velocity.npts = 1, vel.shape[0], vel.shape[1]
    # Set Eikonal solver
    solver_ek = pykonal.EikonalSolver(coord_sys="cartesian")
    solver_ek.vv.min_coords = velocity.min_coords
    solver_ek.vv.node_intervals = velocity.node_intervals
    solver_ek.vv.npts = velocity.npts
    solver_ek.vv.values[:] = vel
    # Initial conditions
    solver_ek.tt.values[:] = np.inf
    solver_ek.known[:] = False
    solver_ek.unknown[:] = True
    # Source location initial conditions
    eq_iz = SouPos[ishot,1]
    eq_iy = 0
    eq_ix = SouPos[ishot,0]
    src_idx = (eq_iy, eq_ix, eq_iz)
    solver_ek.tt.values[src_idx] = 0.0
    solver_ek.unknown[src_idx] = False
    solver_ek.trial.push(*src_idx)
    solver_ek.solve()
    for iRec in range(nRec):
        data[iRec] += solver_ek.tt.values[0, RecPos[iRec,0], RecPos[iRec,1]]
    return data, solver_ek.tt.values[0,:,:]

class EikonalTT_2D(occamypy.Operator):

    def __init__(self, vel, tt_data, dx, dz, SouPos, RecPos):
        """2D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        self.setDomainRange(vel, tt_data)
        # Setting acquisition geometry
        self.nSou = SouPos.shape[0]
        self.nRec = RecPos.shape[0]
        self.SouPos = SouPos.copy()
        self.RecPos = RecPos.copy()
        dataShape = tt_data.shape
        if dataShape[0] != self.nSou:
            raise ValueError("Number of sources inconsistent with traveltime vector (shape[0])")
        if dataShape[1] != self.nRec:
            raise ValueError("Number of receivers inconsistent with traveltime vector (shape[1])")
        self.dx = dx
        self.dz = dz
        self.nx = vel.shape[0]
        self.nz = vel.shape[1]
        velNd = vel.getNdArray()
        # General unsorted traveltime indices
        idx_1d = np.arange(velNd[:,:].size)
        idx,idz = np.unravel_index(idx_1d, velNd[:,:].shape)
        self.tt_idx = np.array([idx,idz]).T
        # Traveltime maps
        self.tt_maps = []
        for _ in range(self.nSou):
            self.tt_maps.append(np.zeros_like(velNd[:,:]))

    def forward(self, add, model, data):
        """Forward non-linear traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        velNd = model.getNdArray()
        # Initialization
        tt = np.zeros((self.nx, self.nz))
        for iShot in range(self.nSou):
            dataNd[iShot,:], self.tt_maps[iShot] = compute_tt(velNd, self.SouPos, self.RecPos, self.dx, self.dz, dataNd[iShot,:], iShot)
        return


# Eikonal tomography-related operator
def sorting2D(tt, idx_l, ordering="a"):
    idx1 = idx_l[:,0]
    idx2 = idx_l[:,1]
    idx = np.ravel_multi_index((idx1, idx2), tt.shape)
    if ordering == "a":
        sorted_indices = np.argsort(tt.ravel()[idx])
    elif ordering == "d":
        sorted_indices = np.argsort(-tt.ravel()[idx])
    else:
        raise ValueError("Unknonw ordering: %s! Provide a or d for ascending or descending" % ordering)       
    # Sorted indices for entire array
    sorted_indices = idx[sorted_indices]
    # Sorting indices
    idx1, idx2 = np.unravel_index(sorted_indices, tt.shape)
    idx_sort = np.array([idx1,idx2], dtype=np.int64).T
    return idx_sort

@jit(nopython=True, cache=True)
def select_upwind_der2D(tt, idx_t0, vv, ds_inv, iax):
    """Find upwind derivative along iax"""
    nx = vv.shape[0]
    nz = vv.shape[1]
    ns = np.array([nx, nz])
    nb = np.zeros(2, dtype=np.int64)
    shift = np.zeros(2, dtype=np.int64)
    drxns = [-1, 1]
    fdt = np.zeros(2)
    order = np.zeros(2, dtype=np.int64)
    
    # Computing derivative for the neighboring points along iax
    for idx in range(2):
        shift[iax] = drxns[idx]
        nb[:] = idx_t0[:] + shift[:]
        # If point is outside the domain skip it
        if np.any(nb < 0) or np.any(nb >= ns):
            continue
        if vv[nb[0], nb[1]] > 0.0:
            order[idx] = 1
            fdt[idx] = drxns[idx] * (tt[nb[0], nb[1]]-tt[idx_t0[0], idx_t0[1]]) * ds_inv[iax]
        else:
            order[idx] = 0
    # Selecting upwind derivative 
    if fdt[0] > -fdt[1] and order[0] > 0:
        fd, idrx = fdt[0], -1
    elif fdt[0] <= -fdt[1] and order[1] > 0:
        fd, idrx = fdt[1], 1
    else:
        fd, idrx = 0.0, 0
    return fd, idrx

@jit(nopython=True, cache=True)
def FMM_tt_lin_fwd2D(delta_v, delta_tt, vv, tt, tt_idx, dx, dz):
    """Fast-marching method linearized forward"""
    nx = delta_v.shape[0]
    nz = delta_v.shape[1]
    ns = np.array([nx, nz])
    drxns = [-1, 1]
    dx_inv = 1.0 / dx
    dz_inv = 1.0 / dz
    ds_inv = np.array([dx_inv, dz_inv])
    
    # Shift variables
    order = np.zeros(2, dtype=np.int64)
    shift = np.zeros(2, dtype=np.int64)
    idrx = np.zeros(2, dtype=np.int64)
    fdt0 = np.zeros(2)
    
    # Scaling the velocity perturbation
    delta_v_scaled = - 2.0 * delta_v / (vv * vv * vv)
    
    # Looping over all indices to solve linear equations from increasing traveltime values
    for idx_t0 in tt_idx:
        # If T = 0 or v = 0, then assuming zero to avoid singularity
        if tt[idx_t0[0], idx_t0[1]] == 0.0 or vv[idx_t0[0], idx_t0[1]] == 0.0:
            continue
        
        # Looping over 
        fdt0.fill(0.0)
        idrx.fill(0)
        for iax in range(2):
            # Loop over neighbourning points to find up-wind direction
            fdt = np.zeros(2)
            order.fill(0)
            shift.fill(0)
            for idx in range(2):
                shift[iax] = drxns[idx]
                nb = idx_t0[:] + shift[:]
                # If point is outside the domain skip it
                if np.any(nb < 0) or np.any(nb >= ns):
                    continue
                if vv[nb[0], nb[1]] > 0.0:
                    order[idx] = 1
                    fdt[idx] = drxns[idx] * (tt[nb[0], nb[1]] - tt[idx_t0[0], idx_t0[1]]) * ds_inv[iax]
                else:
                    order[idx] = 0
            # Selecting upwind derivative 
            shift.fill(0)
            if fdt[0] > -fdt[1] and order[0] > 0:
                idrx[iax], shift[iax] = -1, -1
            elif fdt[0] <= -fdt[1] and order[1] > 0:
                idrx[iax], shift[iax] = 1, 1
            else:
                idrx[iax] = 0
            nb = idx_t0[:] + shift[:]
            # Computing t0 space derivative
            fdt0[iax] = idrx[iax] * (tt[nb[0], nb[1]] - tt[idx_t0[0], idx_t0[1]]) * ds_inv[iax] * ds_inv[iax]
        # Using single stencil along z direction to update value
        if tt[idx_t0[0] + idrx[0], idx_t0[1]] > tt[idx_t0[0], idx_t0[1]]:
            denom = - 2.0 * idrx[1] * fdt0[1]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
        # Using single stencil along x direction to update value
        elif tt[idx_t0[0], idx_t0[1] + idrx[1]] > tt[idx_t0[0], idx_t0[1]]:
            denom = - 2.0 * idrx[0] * fdt0[0]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
        else:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[1] * fdt0[1])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1]] +
                                                   - idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
    return

# Adjoint operator
@jit(nopython=True, cache=True)
def FMM_tt_lin_adj2D(delta_v, delta_tt, vv, tt, tt_idx, dx, dz):
    """Fast-marching method linearized forward"""
    nx = delta_v.shape[0]
    nz = delta_v.shape[1]
    ns = np.array([nx, nz])
    drxns = [-1, 1]
    dx_inv = 1.0 / dx
    dz_inv = 1.0 / dz
    ds_inv = np.array([dx_inv, dz_inv])
    
    # Internal variables
    order = np.zeros(2, dtype=np.int64)
    shift = np.zeros(2, dtype=np.int64)
    nbrs = np.zeros((4,2), dtype=np.int64)
    fdt_nb = np.zeros(4)
    order_nb = np.zeros(4, dtype=np.int64)
    idrx_nb = np.zeros(4, dtype=np.int64)
    
    # Looping over all indices to solve linear equations from increasing traveltime values
    for idx_t0 in tt_idx:
        # If T = 0 or v = 0, then assuming zero to avoid singularity
        if tt[idx_t0[0], idx_t0[1]] == 0.0 or vv[idx_t0[0], idx_t0[1]] == 0.0:
            continue
        
        # Creating indices of neighbouring points
        # Order left/right bottom/top
        inbr = 0
        for iax in range(2):
            shift.fill(0)
            for idx in range(2):
                shift[iax] = drxns[idx]
                nbrs[inbr][:] = idx_t0[:] + shift[:]
                inbr += 1
        
        # Looping over neighbouring points
        fdt_nb.fill(0)
        idrx_nb.fill(0)
        for ib, nb in enumerate(nbrs):
            # Point outside of modeling domain
            if np.any(nb < 0) or np.any(nb >= ns):
                order_nb[ib] = 0
                continue
            # Point with lower traveltime compared to current point
            if tt[idx_t0[0], idx_t0[1]] > tt[nb[0], nb[1]]:
                order_nb[ib] = 0
                continue
            order_nb[ib] = 1
            # Getting derivative along given axis
            iax = 0 if ib in [0,1] else 1
            fdt_nb[ib], idrx_nb[ib] = select_upwind_der2D(tt, nb, vv, ds_inv, iax)
            # Removing point if derivative at nb did not use idx_t0
            if ib in [0,1]:
                # Checking x direction
                if idx_t0[0] != nb[0] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
            else:
                # Checking z direction
                if idx_t0[1] != nb[1] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
        
        # Updating delta_v according to stencil
        fdt_nb *= -idrx_nb
        fdt0 = 0.0
        fdt_nb[0] *= dx_inv
        fdt_nb[1] *= dx_inv
        fdt_nb[2] *= dz_inv
        fdt_nb[3] *= dz_inv

        if np.all(order_nb[:2]):
            fdt0, idrx0 = select_upwind_der2D(tt, idx_t0, vv, ds_inv, 1)
            fdt0 *= np.sign(idrx0) * dz_inv
        elif np.all(order_nb[2:]):
            fdt0, idrx0 = select_upwind_der2D(tt, idx_t0, vv, ds_inv, 0)
            fdt0 *= np.sign(idrx0) * dx_inv
        else:
            fdt0x, idrx0x = select_upwind_der2D(tt, idx_t0, vv, ds_inv, 0)
            fdt0z, idrx0z = select_upwind_der2D(tt, idx_t0, vv, ds_inv, 1)
            # Necessary to consider correct stencil central value
            if tt[idx_t0[0], idx_t0[1]] < tt[idx_t0[0] + idrx0x, idx_t0[1]]: 
                fdt0x, idrx0x = 0.0, 0
            if tt[idx_t0[0], idx_t0[1]] < tt[idx_t0[0], idx_t0[1] + idrx0z]: 
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0x * fdt0x * dx_inv + idrx0z * fdt0z * dz_inv
        
        # Update delta_v value
        delta_v[idx_t0[0], idx_t0[1]] -= (  fdt_nb[0] * delta_v[idx_t0[0]-order_nb[0], idx_t0[1]] 
                                          + fdt_nb[1] * delta_v[idx_t0[0]+order_nb[1], idx_t0[1]] 
                                          + fdt_nb[2] * delta_v[idx_t0[0], idx_t0[1]-order_nb[2]] 
                                          + fdt_nb[3] * delta_v[idx_t0[0], idx_t0[1]+order_nb[3]] 
                                          - 0.5 * delta_tt[idx_t0[0], idx_t0[1]]) / fdt0
    
    # Scaling the velocity perturbation
    delta_v[:] = 2.0 * delta_v / (vv * vv * vv)
            
    return


def solve_linearized_fwd(vel0, delta_v, ishot, dx, dz, tt0, tt_idx, RecPos):
    """Function to solve linearized problem"""
    # Sorting traveltime in ascending order
    tt_idx = sorting2D(tt0, tt_idx)
    delta_tt = np.zeros_like(vel0)
    FMM_tt_lin_fwd2D(delta_v, delta_tt, vel0, tt0, tt_idx, dx, dz)
    return delta_tt

def solve_linearized_adj(vel0, data, ishot, dx, dz, tt0, tt_idx, RecPos):
    delta_tt = np.zeros_like(vel0)
    delta_v = np.zeros_like(vel0)
    # Sorting traveltime in ascending order
    tt_idx = sorting2D(tt0, tt_idx, ordering="d")
    # Injecting traveltime to correct grid positions
    for iRec in range(RecPos.shape[0]):
        delta_tt[RecPos[iRec, 0], RecPos[iRec, 1]] = data[ishot, iRec]
    FMM_tt_lin_adj2D(delta_v, delta_tt, vel0, tt0, tt_idx, dx, dz)
    return delta_v

class EikonalTT_lin_2D(occamypy.Operator):

    def __init__(self, vel, tt_data, dx, dz, SouPos, RecPos, tt_maps=None):
        """2D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        self.setDomainRange(vel, tt_data)
        # Setting acquisition geometry
        self.nSou = SouPos.shape[0]
        self.nRec = RecPos.shape[0]
        self.SouPos = SouPos.copy()
        self.RecPos = RecPos.copy()
        dataShape = tt_data.shape
        if dataShape[0] != self.nSou:
            raise ValueError("Number of sources inconsistent with traveltime vector (shape[0])")
        if dataShape[1] != self.nRec:
            raise ValueError("Number of receivers inconsistent with traveltime vector (shape[1])")
        self.dx = dx
        self.dz = dz
        self.nx = vel.shape[0]
        self.nz = vel.shape[1]
        # Background model
        self.vel = vel.clone()
        self.vel.copy(vel)
        self.velNd = vel.getNdArray()
        # General unsorted traveltime indices
        idx_1d = np.arange(self.velNd.size)
        idx,idz = np.unravel_index(idx_1d, vel.shape)
        self.tt_idx = np.array([idx,idz]).T
        # Traveltime maps
        if tt_maps is None:
            self.tt_maps = []
            for _ in range(self.nSou):
                self.tt_maps.append(np.zeros_like(self.velNd))
        else:
            self.tt_maps = tt_maps

    def forward(self, add, model, data):
        """Forward linearized traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        vel0Nd = self.vel.getNdArray()
        dummyData = np.zeros_like(dataNd[0,:])
        ###################################
        # Computing background traveltime #
        ###################################
        if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.nSou)]):
            for iShot in range(self.nSou):
                _, self.tt_maps[iShot] = compute_tt(vel0Nd, self.SouPos, self.RecPos, self.dx, self.dz, dummyData, iShot)
        ###################################
        # Computing linearized traveltime #
        ###################################
        for iShot in range(self.nSou):
            delta_tt = solve_linearized_fwd(vel0Nd, modelNd, iShot, self.dx, self.dz, self.tt_maps[iShot], self.tt_idx, self.RecPos)
            for iRec in range(self.nRec):
                dataNd[iShot, iRec] += delta_tt[self.RecPos[iRec, 0], self.RecPos[iRec, 1]]
        return
    
    def adjoint(self, add, model, data):
        """Adjoint linearized traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add: 
            model.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        vel0Nd = self.vel.getNdArray()
        dummyData = np.zeros_like(dataNd[0,:])
        ###################################
        # Computing background traveltime #
        ###################################
        if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.nSou)]):
            for iShot in range(self.nSou):
                _, self.tt_maps[iShot] = compute_tt(vel0Nd, self.SouPos, self.RecPos, self.dx, self.dz, dummyData, iShot)
        ###################################
        # Computing velocity perturbation #
        ###################################
        for iShot in range(self.nSou):
            modelNd[:] += solve_linearized_adj(vel0Nd, dataNd, iShot, self.dx, self.dz, self.tt_maps[iShot], self.tt_idx, self.RecPos)
        return
    
    def set_vel(self, vel):
        """Function to set background velocity model"""
        self.vel.copy(vel)