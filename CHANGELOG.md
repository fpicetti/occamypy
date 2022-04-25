# 0.1.5
* Moved `set_seed_everywhere` in the main utils submodule
* Signal processing operators are called directly from occamypy init (no cupy/torch submodule needed)
* Removed `snr` logic from `rand` and `randn`;  now they accept distribution descriptors
* A lot of cleanings in the documentation, names, and methods
* Added [PyLops](tutorials/PyLops%20and%20OccamyPy%20together.ipynb) tutorial
* Added Traveltime tomography tutorials in [1D](tutorials/1D%20Travel-time%20tomography%20using%20the%20Eikonal%20equation.ipynb), [2D](tutorials/2D%20Travel-time%20tomography%20using%20the%20Eikonal%20equation.ipynb), and [Marmousi](tutorials/Traveltime%20tomography%20for%20Seismic%20Exploration.ipynb)
* Added [Devito-based LS-RTM](tutorials/2D%20LS-RTM%20with%20devito,%20dask,%20and%20regularizers.ipynb) tutorial
* Added [autograd](occamypy/torch/autograd.py) torch submodule to cast linear operators to torch automatic differentiation engine (see the [tutorial](tutorials/2D%20LS-RTM%20with%20devito%20and%20Automatic%20Differentiation.ipynb))
* Added a future work tutorial on [automatically differentiated operators](tutorials/Automatic%20Differentiation%20for%20nonlinear%20operators.ipynb)

# 0.1.4
* Added support for F-contiguous arrays
* Added [PyLops](https://pylops.readthedocs.io/en/stable/) interface [operators](ea8505947c926e376a6def40b1fccfbadf3940d2)
* Added plot utilities
* Added [`AxInfo`](tutorials/AxInfo%20-%20exploit%20physical%20vectors.ipynb) class for handling physical vectors
* Added Padding operators different from ZeroPad
* Improvements on VectorTorch methods and attributes
* Added FISTA solver wrapper

# 0.1.3
* Fix circular imports

# 0.1.2
* Added `__getitem__()` method to vector class
* Added PyTorch FFT operators
* Fix convolution in PyTorch
* Added a number of utilities

# 0.1.1
* Derivative operators are now agnostic to the computation engine
* Added Dask Blocky Operator
* fixed `rand()` in VectorNumpy
* added kwargs for Dask Operators
 
# 0.1.0
* First official release.
