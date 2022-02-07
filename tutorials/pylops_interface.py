import numpy as np
import occamypy as o
import matplotlib.pyplot as plt
import pylops


if __name__ == '__main__':
    
    shape = (3, 5)
    x = o.VectorNumpy(shape).set(1.)
    
    # test FromPylops
    d = np.arange(x.size)
    D = o.pylops_interface.FromPylops(x, x, pylops.Diagonal(d))
    D.dotTest(True)
    
    # test ToPylops
    S = o.pylops_interface.ToPylops(o.Scaling(x, 2.))
    pylops.utils.dottest(S, x.size, x.size, tol=1e-6, complexflag=0, verb=True)
    print("\t", np.isclose((S * x.arr.ravel()).reshape(x.shape), 2. * x.arr).all())
    
    ####################################################################################################################
    # interesting pylops example
    # https://pylops.readthedocs.io/en/latest/tutorials/ctscan.html#sphx-glr-tutorials-ctscan-py
    def radoncurve(x, r, theta):
        return (r - ny // 2) / (np.sin(np.deg2rad(theta)) + 1e-15) + np.tan(np.deg2rad(90 - theta)) * x + ny // 2
    
    
    x = np.load('data/shepp_logan_phantom.npy').T
    x = x / x.max()
    nx, ny = x.shape
    
    ntheta = 150
    theta = np.linspace(0., 180., ntheta, endpoint=False)
    
    RLop = pylops.signalprocessing.Radon2D(np.arange(ny), np.arange(nx),
                                           theta, kind=radoncurve,
                                           centeredh=True, interp=False,
                                           engine='numpy', dtype='float64')
    
    y = RLop.H * x.ravel()
    y = y.reshape(ntheta, ny)
    
    xrec = RLop * y.ravel()
    xrec = xrec.reshape(nx, ny)
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(x.T, vmin=0, vmax=1, cmap='gray')
    axs[0].set_title('Model')
    axs[0].axis('tight')
    axs[1].imshow(y.T, cmap='gray')
    axs[1].set_title('Data')
    axs[1].axis('tight')
    axs[2].imshow(xrec.T, cmap='gray')
    axs[2].set_title('Adjoint model')
    axs[2].axis('tight')
    fig.tight_layout()
    plt.show()
    
    x_ = o.VectorNumpy(x)
    y_ = o.VectorNumpy((ntheta, ny))
    R_ = o.pylops_interface.FromPylops(x_, y_, RLop.H)
    y_ = R_ * x_
    xrec_ = R_.H * y_
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(x_.plot().T, vmin=0, vmax=1, cmap='gray')
    axs[0].set_title('Model')
    axs[0].axis('tight')
    axs[1].imshow(y_.plot().T, cmap='gray')
    axs[1].set_title('Data')
    axs[1].axis('tight')
    axs[2].imshow(xrec_.plot().T, cmap='gray')
    axs[2].set_title('Adjoint model')
    axs[2].axis('tight')
    fig.tight_layout()
    plt.show()
    
    # TV inversion
    G = o.Gradient(x_, stencil="backward")
    
    # same of PyLops tutorial
    mu = 1.5
    lamda = [1.0, 1.0]
    niter = 3
    niterinner = 4
    
    problemTV = o.GeneralizedLasso(x_.clone().zero(), y_, R_, eps=lamda[0] / mu, reg=G)
    
    SB = o.SplitBregman(o.BasicStopper(niter=niter), niter_inner=niterinner, niter_solver=20,
                        linear_solver="LSQR", warm_start=True)
    
    SB.setDefaults(save_obj=True)
    SB.run(problemTV, verbose=True, inner_verbose=False)
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(x_.plot().T, vmin=0, vmax=1, cmap='gray')
    axs[0].set_title('Model')
    axs[0].axis('tight')
    axs[1].imshow(problemTV.model.plot().T, vmin=0, vmax=1, cmap='gray')
    axs[1].set_title('Inverted model')
    axs[1].axis('tight')
    fig.tight_layout()
    plt.show()
    
    ####################################################################################################################
    # now with dask
    client = o.dask.DaskClient(local_params={"processes": True}, n_wrks=4)
    print("Number of workers = %d" % client.getNworkers())
    print("Workers Ids = %s" % client.getWorkerIds())
    
    vec = o.VectorNumpy(np.load('data/shepp_logan_phantom.npy').T * 1.).scale(1 / 255)
    
    # # method 1: instantiate a vector template and spread it using the chunk parameter
    # this creates one copy of the vector on each worker, for each chunk.
    # e.g., here we have 4 workers, with 2,3,5,10 copies each.
    # verify that vec_.vecDask[0].result().checkSame(vec_.vecDask[1].result())
    v1 = o.dask.DaskVector(client,
                           vector_template=vec,
                           chunks=(2, 3, 5, 10))
    
    # # method 2: instantiate multiple vectors and spreading them to the given workers
    # this is useful for distibuting large vectors
    v2 = o.dask.DaskVector(client,
                           vectors=[o.VectorNumpy(vec[:50]),
                                    o.VectorNumpy(vec[50:100]),
                                    o.VectorNumpy(vec[100:150]),
                                    o.VectorNumpy(vec[150:])],
                           chunks=(1, 1, 1, 1))
    
    # Collect such a dask vector to a in-core vector
    Collect = o.dask.DaskCollect(v2, vec)
    v2c = Collect * v2
    vec.checkSame(v2c)
    
    # method 3: use a Spread operator to distribute a vector...
    Spread = o.dask.DaskSpread(client, vec, [1] * client.getNworkers())
    v3 = Spread * vec
    # ...and to get it back
    pippo = Spread.T * v3
    pippo.checkSame(vec)
    
    # Pylops To DaskOperator, leveraging FromPylops.
    D1 = o.dask.DaskOperator(
        client,
        o.pylops_interface.FromPylops,
        [(v, v, pylops.FirstDerivative(np.prod(shape))) for v, shape in zip(v1.vecDask, v1.shape)],
        v1.chunks)
    D1.dotTest(True)
    
    D2 = o.dask.DaskOperator(
        client,
        o.pylops_interface.FromPylops,
        [(v, v, pylops.FirstDerivative(np.prod(shape))) for v, shape in zip(v2.vecDask, v2.shape)],
        v2.chunks)
    D2.dotTest(True)
    D3 = o.dask.DaskOperator(
        client,
        o.pylops_interface.FromPylops,
        [(v, v, pylops.FirstDerivative(np.prod(shape))) for v, shape in zip(v3.vecDask, v3.shape)],
        Spread.chunks)
    D3.dotTest(True)
    
    print("end of script")
