"""
bubblebuster
A Python Library for detecting water box bubbles in structural 
files used in molecular simulations. 
"""

import sys
import os
from setuptools import setup, find_packages
from setuptools import setup, Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("bubblebuster.bubblebuster", ["bubblebuster/bubblebuster.pyx"]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("bubblebuster.bubblebuster", ["bubblebuster/bubblebuster.c"]),
    ]

long_description = str("Bubblebuster is a Python Library for detecting water box bubbles " 
    + "in structural files used in molecular simulations, by partitioning the system's "
    + "periodic box into a cubic grid. The atom density of each cube is compared against "
    + "the periodic box's atom density. If any cube has a density less than the periodic "
    + "box atom density multiplied by the cutoff parameter (default is 0.5), "
    + "the water box is deemed to likely have a bubble. The size of the cubes, along with "
    + "the cutoff value, can be customized according to the system. However, this must be "
    + "done with caution as a cutoff value that is to large/small can yield false "
    + "positives/negatives."
    )

setup(
    name='bubblebuster',
    author='Andy Stokely',
    author_email='amstokely@ucsd.edu',
    license='MIT',
    description="A Python Library for detecting water box bubbles in structural "
        + "files used in molecular simulations.",
    keywords='molecular dynamics water box periodic',
    url='https://github.com/astokely/bubblebuster',
    long_description=long_description,
    packages=find_packages(),
    install_requires=["numpy", "pytest", "nptyping", "mdtraj", "cython"],              
    platforms=['Linux',
                'Unix',],
    python_requires=">=3.6",          
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
