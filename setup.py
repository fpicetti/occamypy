import os
from setuptools import setup, find_packages


def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)


setup(name='occamypy',
      version='0.1',
      description='An Object-Oriented Optimization Framework for Large-Scale Inverse Problems',
      long_description=open(src('README.md')).read(),
      long_description_content_type='text/markdown',
      keywords=['algebra', 'inverse problems', 'large-scale optimiziation'],
      classifiers=[
          # 'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          # 'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
          'Natural Language :: English',
          'Programming Language :: Python :: >3.6',
          'Topic :: Scientific/Engineering :: Mathematics'
      ],
      license='GNU',

      # url='http://ictshore.com/',
      author='Ettore Biondi, Guillame Barnier, Robert Clapp, Francesco Picetti, Stuart Farris',  # TODO they should be our github nicknames
      author_email='ettore88@stanford.edu',  # TODO it should be a definitive email, maybe a gmail one?
      
      install_requires=['numpy >= 1.15.0', 'scipy', 'cupy'],  # todo add hdf5
      extras_require={'advanced': ['numba', 'pyfftw', 'PyWavelets']},
      packages=find_packages(),
      zip_safe=True)
