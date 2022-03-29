"""
Operators to perform 2D Eikonal tomography

@Author: Ettore Biondi - ebiondi@caltech.edu
"""
from psutil import cpu_count
import numpy as np
from numba import jit
import occamypy as o

try:
    import pykonal
except ModuleNotFoundError:
    print("Please install pykonal==0.3.2b3 in order to use these operators.")

__all__ = [
    "EikonalTT_2D",
    "EikonalTT_lin_2D",
]


def _compute_traveltime(vel: np.ndarray, SouPos: np.ndarray, RecPos: np.ndarray, dx: float, dz: float, data: np.ndarray, ishot: int):
    """Function to compute traveltimes by solving Eikonal equation"""
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
    eq_iz = SouPos[ishot, 1]
    eq_iy = 0
    eq_ix = SouPos[ishot, 0]
    src_idx = (eq_iy, eq_ix, eq_iz)
    solver_ek.tt.values[src_idx] = 0.0
    
    solver_ek.unknown[src_idx] = False
    solver_ek.trial.push(*src_idx)
    solver_ek.solve()
    
    for iRec in range(nRec):
        data[iRec] += solver_ek.tt.values[0, RecPos[iRec, 0], RecPos[iRec, 1]]
    return data, solver_ek.tt.values[0, :, :]


# Eikonal tomography-related operator
def _sorting2D(tt: np.ndarray, idx_l: np.ndarray, ordering="a") -> np.ndarray:
    idx1 = idx_l[:, 0]
    idx2 = idx_l[:, 1]
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
    idx_sort = np.array([idx1, idx2], dtype=np.int64).T
    return idx_sort


@jit(nopython=True, cache=True)
def _select_upwind_der2D(tt: np.ndarray, idx_t0: np.ndarray, vv: np.ndarray, ds_inv, iax: int):
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
            fdt[idx] = drxns[idx] * (tt[nb[0], nb[1]] - tt[idx_t0[0], idx_t0[1]]) * ds_inv[iax]
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
def _fmm_tt_lin_fwd(delta_v: np.ndarray, delta_tt: np.ndarray, vv: np.ndarray, tt: np.ndarray, tt_idx: np.ndarray, dx: float, dz: float):
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
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[1] * 2.0 * fdt0[1] * delta_tt[
                    idx_t0[0], idx_t0[1] + idrx[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
        # Using single stencil along x direction to update value
        elif tt[idx_t0[0], idx_t0[1] + idrx[1]] > tt[idx_t0[0], idx_t0[1]]:
            denom = - 2.0 * idrx[0] * fdt0[0]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[
                    idx_t0[0] + idrx[0], idx_t0[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
        else:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[1] * fdt0[1])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[
                    idx_t0[0] + idrx[0], idx_t0[1]] +
                                                   - idrx[1] * 2.0 * fdt0[1] * delta_tt[
                                                       idx_t0[0], idx_t0[1] + idrx[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
    return


@jit(nopython=True, cache=True)
def _fmm_tt_lin_adj(delta_v: np.ndarray, delta_tt: np.ndarray, vv: np.ndarray, tt: np.ndarray, tt_idx: np.ndarray, dx: float, dz: float):
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
    nbrs = np.zeros((4, 2), dtype=np.int64)
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
            iax = 0 if ib in [0, 1] else 1
            fdt_nb[ib], idrx_nb[ib] = _select_upwind_der2D(tt, nb, vv, ds_inv, iax)
            # Removing point if derivative at nb did not use idx_t0
            if ib in [0, 1]:
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
            fdt0, idrx0 = _select_upwind_der2D(tt, idx_t0, vv, ds_inv, 1)
            fdt0 *= np.sign(idrx0) * dz_inv
        elif np.all(order_nb[2:]):
            fdt0, idrx0 = _select_upwind_der2D(tt, idx_t0, vv, ds_inv, 0)
            fdt0 *= np.sign(idrx0) * dx_inv
        else:
            fdt0x, idrx0x = _select_upwind_der2D(tt, idx_t0, vv, ds_inv, 0)
            fdt0z, idrx0z = _select_upwind_der2D(tt, idx_t0, vv, ds_inv, 1)
            # Necessary to consider correct stencil central value
            if tt[idx_t0[0], idx_t0[1]] < tt[idx_t0[0] + idrx0x, idx_t0[1]]:
                fdt0x, idrx0x = 0.0, 0
            if tt[idx_t0[0], idx_t0[1]] < tt[idx_t0[0], idx_t0[1] + idrx0z]:
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0x * fdt0x * dx_inv + idrx0z * fdt0z * dz_inv
        
        # Update delta_v value
        delta_v[idx_t0[0], idx_t0[1]] -= (fdt_nb[0] * delta_v[idx_t0[0] - order_nb[0], idx_t0[1]]
                                          + fdt_nb[1] * delta_v[idx_t0[0] + order_nb[1], idx_t0[1]]
                                          + fdt_nb[2] * delta_v[idx_t0[0], idx_t0[1] - order_nb[2]]
                                          + fdt_nb[3] * delta_v[idx_t0[0], idx_t0[1] + order_nb[3]]
                                          - 0.5 * delta_tt[idx_t0[0], idx_t0[1]]) / fdt0
    
    # Scaling the velocity perturbation
    delta_v[:] = 2.0 * delta_v / (vv * vv * vv)
    
    return


def _solve_linearized_fwd(vel0: np.ndarray, delta_v: np.ndarray, ishot: int, dx: float, dz: float, tt0: np.ndarray, tt_idx: np.ndarray, RecPos: np.ndarray):
    """Function to solve linearized problem"""
    # Sorting traveltime in ascending order
    tt_idx = _sorting2D(tt0, tt_idx)
    delta_tt = np.zeros_like(vel0)
    _fmm_tt_lin_fwd(delta_v, delta_tt, vel0, tt0, tt_idx, dx, dz)
    return delta_tt


def _solve_linearized_adj(vel0: np.ndarray, data: np.ndarray, ishot: int, dx: float, dz: float, tt0: np.ndarray, tt_idx: np.ndarray, RecPos: np.ndarray):
    delta_tt = np.zeros_like(vel0)
    delta_v = np.zeros_like(vel0)
    # Sorting traveltime in ascending order
    tt_idx = _sorting2D(tt0, tt_idx, ordering="d")
    # Injecting traveltime to correct grid positions
    for iRec in range(RecPos.shape[0]):
        delta_tt[RecPos[iRec, 0], RecPos[iRec, 1]] = data[ishot, iRec]
    _fmm_tt_lin_adj(delta_v, delta_tt, vel0, tt0, tt_idx, dx, dz)
    return delta_v


class EikonalTT_2D(o.Operator):
    
    def __init__(self, vel: o.Vector, tt_data: o.Vector, SouPos: np.ndarray, RecPos: np.ndarray, dx: float = None, dz: float = None):
        """2D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        super(EikonalTT_2D, self).__init__(domain=vel, range=tt_data)
        # Setting acquisition geometry
        self.ns = SouPos.shape[0]
        self.nr = RecPos.shape[0]
        self.SouPos = SouPos.copy()
        self.RecPos = RecPos.copy()
        dataShape = tt_data.shape
        if dataShape[0] != self.ns:
            raise ValueError("Number of sources inconsistent with traveltime vector (shape[0])")
        if dataShape[1] != self.nr:
            raise ValueError("Number of receivers inconsistent with traveltime vector (shape[1])")
        self.dx = dx if dx is not None else vel.ax_info[0].d
        self.dz = dz if dz is not None else vel.ax_info[1].d
        self.nx = vel.shape[0]
        self.nz = vel.shape[1]
        # General unsorted traveltime indices
        idx_1d = np.arange(vel.size)
        idx, idz = np.unravel_index(idx_1d, vel.shape)
        self.tt_idx = np.array([idx, idz]).T
        # Traveltime maps
        self.tt_maps = []
        for _ in range(self.ns):
            self.tt_maps.append(np.zeros(vel.shape))
    
    def forward(self, add, model, data):
        """Forward non-linear traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()

        for iShot in range(self.ns):
            data[iShot], self.tt_maps[iShot] = _compute_traveltime(model[:], self.SouPos, self.RecPos, self.dx, self.dz, data[iShot], iShot)
        return


class EikonalTT_lin_2D(o.Operator):
    
    def __init__(self, vel: o.Vector, tt_data: o.Vector, SouPos: np.ndarray, RecPos: np.ndarray, dx: float = None, dz: float = None, tt_maps=None):
        """2D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        super(EikonalTT_lin_2D, self).__init__(domain=vel, range=tt_data)
        # Setting acquisition geometry
        self.ns = SouPos.shape[0]
        self.nr = RecPos.shape[0]
        self.SouPos = SouPos.copy()
        self.RecPos = RecPos.copy()
        dataShape = tt_data.shape
        if dataShape[0] != self.ns:
            raise ValueError("Number of sources inconsistent with traveltime vector (shape[0])")
        if dataShape[1] != self.nr:
            raise ValueError("Number of receivers inconsistent with traveltime vector (shape[1])")
        self.dx = dx if dx is not None else vel.ax_info[0].d
        self.dz = dz if dz is not None else vel.ax_info[1].d
        self.nx = vel.shape[0]
        self.nz = vel.shape[1]
        # Background model
        self.vel = vel.clone()
        # General unsorted traveltime indices
        idx_1d = np.arange(vel.size)
        idx, idz = np.unravel_index(idx_1d, vel.shape)
        self.tt_idx = np.array([idx, idz]).T
        # Traveltime maps
        if tt_maps is None:
            self.tt_maps = []
            for _ in range(self.ns):
                self.tt_maps.append(np.zeros(self.vel.shape))
        else:
            self.tt_maps = tt_maps
    
    def forward(self, add, model, data):
        """Forward linearized traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        dummyData = np.zeros_like(data[0])
        ###################################
        # Computing background traveltime #
        ###################################
        if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.ns)]):
            for shot in range(self.ns):
                _, self.tt_maps[shot] = _compute_traveltime(self.vel[:], self.SouPos, self.RecPos, self.dx, self.dz, dummyData, shot)
        ###################################
        # Computing linearized traveltime #
        ###################################
        for shot in range(self.ns):
            delta_tt = _solve_linearized_fwd(self.vel[:], model[:], shot, self.dx, self.dz, self.tt_maps[shot], self.tt_idx, self.RecPos)
            for iRec in range(self.nr):
                data[shot, iRec] += delta_tt[self.RecPos[iRec, 0], self.RecPos[iRec, 1]]
        return
    
    def adjoint(self, add, model, data):
        """Adjoint linearized traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        dummyData = np.zeros_like(data[0])
        ###################################
        # Computing background traveltime #
        ###################################
        if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.ns)]):
            for shot in range(self.ns):
                _, self.tt_maps[shot] = _compute_traveltime(self.vel[:], self.SouPos, self.RecPos, self.dx, self.dz, dummyData, shot)
        ###################################
        # Computing velocity perturbation #
        ###################################
        for shot in range(self.ns):
            model[:] += _solve_linearized_adj(self.vel[:], data[:], shot, self.dx, self.dz, self.tt_maps[shot], self.tt_idx, self.RecPos)
        return
    
    def set_vel(self, vel):
        """Function to set background velocity model"""
        self.vel.copy(vel)
