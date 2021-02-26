import os
from setuptools import setup, find_packages

NAME = "occamypy"
DESCRIPTION = "OccamyPy. An object-oriented optimization library for small- and large-scale problems."
URL = "https://github.com/fpicetti/occamypy"
EMAIL = "francesco.picetti@polimi.it"
AUTHOR = "Ettore Biondi, Guillame Barnier, Robert Clapp, Francesco Picetti, Stuart Farris"
REQUIRES_PYTHON = ">=3.6.0"
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_readme():
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with open(readme_path, encoding="utf-8") as f:
        return f"\n{f.read()}"


def load_version():
    context = {}
    with open(os.path.join(PROJECT_ROOT, "occamypy", "__version__.py")) as f:
        exec(f.read(), context)
    return context["__version__"]


setup(name='occamypy',
      version=load_version(),
      url="https://github.com/fpicetti/occamypy",
      description='An Object-Oriented Optimization Framework for Large-Scale Inverse Problems',
      long_description=load_readme(),
      long_description_content_type='text/markdown',
      keywords=['algebra', 'inverse problems', 'large-scale optimization'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Operating System :: Unix'
      ],
      author=AUTHOR,
      author_email=EMAIL,
      install_requires=['numpy',
                        'scipy',
                        'h5py',
                        'numba',
                        'torch>=1.7.0',
                        'dask',
                        'dask-jobqueue',
                        'dask-kubernetes',
                        'matplotlib',
                        'gputil',
                        ],
      packages=find_packages(),
      zip_safe=True)
