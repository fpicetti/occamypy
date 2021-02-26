# OccamyPy: an object-oriented optimization library for small- and large-scale problems

@Authors: [Ettore Biondi](mailto:ettore88@stanford.edu),
Guillame Barnier,
Robert Clapp,
[Francesco Picetti](mailto:francesco.picetti@polimi.it),
Stuart Farris

### Abstract
We present an object-oriented optimization framework that can be employed to solve
small- and large-scale problems based on the concept of vectors and operators.
By using such a strategy, we implement different iterative optimization algorithms
that can be used in combination with architecture-independent vectors and operators,
allowing the minimization of single-machine or cluster-based problems with a unique codebase.
We implement a Python library following the described structure with a user-friendly interface.
We demonstrate its flexibility and scalability on multiple inverse problems,
where convex and non-convex objective functions are optimized with different iterative algorithms.

### Installation
Preferred way is through Python Package Index:
```bash
pip install occamypy
```
In a python3 environment, clone this repo and then simply run `pip install -e .`;
the library is set up in order to install its requirements.


### History
This library was initially developed at
[Stanford Exploration Project](http://zapad.stanford.edu/ettore88/python-solver)
for solving large scale seismic problems.
Inspired by Equinor's [PyLops](https://github.com/equinor/pylops)
we publish this library as our contribution to scientific community.

## How it works
This framework allows for the definition of linear and non-linear mapping functions that
operate on abstract vector objects that can be defined to use
heterogeneous computational resources, from personal laptops to HPC environments.

- **vector** class: this is the building block for handling data. It contains the required
mathematical operations such as norm, scaling, dot-product, sum, point-wise multiplication.
These methods can be implemented using existing libraries (e.g., Numpy, Cupy, PyTorch) or
user-defined ones (e.g., [SEPLib](http://sepwww.stanford.edu/doku.php?id=sep:software:seplib)).
See the [`vector`](./occamypy/vector) subpackage for details and implementations.

- **operator** class: a mapping function between a `domain` vector and a `range` vector.
It  can be linear and non-linear.
Linear operators require the definition of both the forward and adjoint functions;
non-linear operators require the forward mapping and its Jacobian operator.
See the [`operator`](./occamypy/operator) subpackage for details and implementations.

- **problem** class: it represents the objective function related to  an optimization problem.
Defined upon operators (e.g., modeling and regularization) and vectors (observed data, priors).
It contains the methods for objective function and gradient computation, as our solvers are mainly gradient based.
See the [`problem`](./occamypy/problem) subpackage for details and implementations.

- **solver** class: it aims at finding the solution to a problem by employing methods
defined within the vector, operator and problem classes.
Additionally, it allows to restart an optimization method from an intermetdiate result
written as serialized objects on permanent computer memory.
See the [`solver`](./occamypy/solver) subpackage for details and implementations.

### Scalability
The main objective of the described framework and implemented library is to solve large-scale inverse problems.
Any vector and operator can be split into blocks to be distributed to multiple nodes.
This is achieved via custom [Dask](https://dask.org/) vector and operator classes.
See the [`dask`](./occamypy/dask) subpackage for details and implementations.

### Tutorials
We provide some [tutorials](./tutorials) that demonstrate the flexibility of occamypy.
Please refer to them as a good starting point for developing your own code.

### Contributing
Follow the following instructions and read carefully the [CONTRIBUTING](CONTRIBUTING.md) file before getting started.


