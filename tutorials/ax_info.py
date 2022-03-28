import numpy as np
import occamypy as o
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # load a 20Hz ricker sampled at 500Hz
    ot = 0.
    dt = 0.002  # fs = 500Hz
    wav = o.VectorNumpy(np.load("./data/ricker20.npy"))
    wav.ax_info = [o.AxInfo(wav.size, ot, dt, "time [s]")]
    
    plt.figure()
    plt.plot(wav.ax_info[0].plot(), wav.plot())
    plt.xlim(wav.ax_info[0].o, wav.ax_info[0].last)
    plt.title("Wavelet"), plt.xlabel(wav.ax_info[0].l)
    plt.tight_layout()
    plt.show()
    
    # build a two-layer velocity model sampled at 10m, with a Gaussian anomaly
    vp = o.VectorNumpy((101, 101)).zero()
    vp.ax_info = [o.AxInfo(101, 0., 10., "z [m]"),
                  o.AxInfo(101, 0., 10., "x [m]")]
    
    vp[35, 50] = 1.
    G = o.GaussianFilter(vp, (3, 10))
    vp = G * vp
    vp.scale(1e5)
    vp.addbias(1500.)
    vp[50:] = 3500.
    
    plt.figure()
    plt.imshow(vp.plot(), cmap="jet", aspect="auto", vmin=vp.min(), vmax=vp.max(),
               extent=[vp.ax_info[1].o, vp.ax_info[1].last, vp.ax_info[0].last, vp.ax_info[0].o])
    plt.colorbar(label="[m/s]")
    plt.xlabel(vp.ax_info[1].l)
    plt.ylabel(vp.ax_info[0].l)
    plt.title("Velocity model")
    plt.tight_layout()
    plt.show()
    