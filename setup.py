"""
bubblebuster
A Python Library for detecting water box bubbles in structural 
files used in molecular simulations. 
"""

import sys
import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

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
    python_requires=">=3.8",          
    ext_modules = cythonize("bubblebuster/bubblebuster.pyx")
)
