import numpy as np
import occamypy as o
from typing import Tuple, List
from devitoseismic import AcquisitionGeometry, demo_model, SeismicModel
from devitoseismic.acoustic import AcousticWaveSolver
import devito

devito.configuration['log-level'] = 'ERROR'


def create_models(args: dict) -> Tuple[SeismicModel, SeismicModel, SeismicModel]:
    hard = demo_model('layers-isotropic', origin=(0., 0.),
                      shape=args["shape"], spacing=args["spacing"],
                      nbl=args["nbl"], grid=None, nlayers=2)
    smooth = demo_model('layers-isotropic', origin=(0., 0.),
                        shape=args["shape"], spacing=args["spacing"],
                        nbl=args["nbl"], grid=hard.grid, nlayers=2)
    
    devito.gaussian_smooth(smooth.vp, sigma=args["filter_sigma"])
    
    water = demo_model('layers-isotropic', origin=(0., 0.),
                       shape=args["shape"], spacing=args["spacing"],
                       nbl=args["nbl"], grid=hard.grid, nlayers=1)
    
    return hard, smooth, water


def build_src_coordinates(x: float, z: float) -> np.ndarray:
    src = np.empty((1, 2), dtype=np.float32)
    src[0, :] = x
    src[0, -1] = z
    return src


def build_rec_coordinates(model: SeismicModel, args: dict) -> np.ndarray:
    """Receivers equispaced on the whole domain"""
    rec = np.empty((args["nreceivers"], 2))
    rec[:, 0] = np.linspace(0, model.domain_size[0], num=args["nreceivers"])
    rec[:, 1] = args["rec_depth"]
    
    return rec


def direct_arrival_mask(data: o.Vector, rec_pos: np.ndarray, src_pos: np.ndarray,
                        vel_sep: float = 1500., offset: float = 0.) -> o.Vector:
    dt = data.ax_info[0].d
    
    direct = np.sqrt(np.sum((src_pos - rec_pos) ** 2, axis=1)) / vel_sep
    direct += offset
    
    mask = data.clone().zero()
    
    iwin = np.round(direct / dt).astype(int)
    for i in range(rec_pos.shape[0]):
        mask[iwin[i]:, i] = 1.
    
    return mask


def _propagate_shot(model: SeismicModel, rec_pos: np.ndarray, src_pos: np.ndarray, param: dict) -> o.VectorNumpy:
    geometry = AcquisitionGeometry(model, rec_pos, src_pos, **param)
    solver = AcousticWaveSolver(model, geometry, **param)
    
    devito.clear_cache()
    
    # propagate (source -> receiver data)
    data = o.VectorNumpy(solver.forward()[0].data.__array__())
    
    data.ax_info = [o.AxInfo(geometry.nt, geometry.t0, geometry.dt / 1000, "time [s]"),
                    o.AxInfo(geometry.nrec, float(rec_pos[0][0]), float(rec_pos[1][0] - rec_pos[0][0]), "rec pos x [m]")]
    
    devito.clear_cache()
    return data


def propagate_shots(model: SeismicModel, rec_pos: np.ndarray, src_pos: List[np.ndarray], param: dict):
    if len(src_pos) == 1:
        return _propagate_shot(model=model, rec_pos=rec_pos, src_pos=src_pos[0], param=param)
    else:
        return o.superVector([_propagate_shot(model=model, rec_pos=rec_pos, src_pos=s, param=param) for s in src_pos])


class BornSingleSource(o.Operator):
    
    def __init__(self, velocity: SeismicModel, src_pos: np.ndarray, rec_pos: np.ndarray, args: dict):
        
        # store params
        self.src_pos = src_pos
        self.rec_pos = rec_pos
        self.nbl = args["nbl"]
        
        # build geometry and acoustic solver
        self.geometry = AcquisitionGeometry(velocity, rec_pos, src_pos, **args)
        self.solver = AcousticWaveSolver(velocity, self.geometry, **args)
        
        # allocate vectors
        self.velocity = o.VectorNumpy(velocity.vp.data.__array__())
        self.velocity.ax_info = [
            o.AxInfo(velocity.vp.shape[0], velocity.origin[0] - self.nbl * velocity.spacing[0], velocity.spacing[0],
                     "x [m]"),
            o.AxInfo(velocity.vp.shape[1], velocity.origin[1] - self.nbl * velocity.spacing[1], velocity.spacing[1],
                     "z [m]")]
        
        csg = o.VectorNumpy((self.geometry.nt, self.geometry.nrec))
        csg.ax_info = [o.AxInfo(self.geometry.nt, self.geometry.t0, self.geometry.dt / 1000, "time [s]"),
                       o.AxInfo(self.geometry.nrec, float(rec_pos[0][0]), float(rec_pos[1][0] - rec_pos[0][0]),
                                "rec pos x [m]")]
        
        super(BornSingleSource, self).__init__(self.velocity, csg)
        
        # store source wavefield
        self.src_wfld = self.solver.forward(save=True)[1]
    
    def __str__(self):
        return "DeviBorn"
    
    def forward(self, add, model, data):
        """Modeling function: image -> residual data"""
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        
        recs = self.solver.jacobian(dmin=model[:])[0]
        data[:] += recs.data.__array__()
        
        return
    
    def adjoint(self, add, model, data):
        """Adjoint function: data -> image"""
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        
        recs = self.geometry.rec.copy()
        recs.data[:] = data[:]
        
        img = self.solver.gradient(rec=recs, u=self.src_wfld)[0]
        model[:] += img.data.__array__()
        
        return
