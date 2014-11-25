# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:53:53 2014

@author: Rafael
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'CAMP_C',
  ext_modules = cythonize(["CAMP_C.pyx"]),
  include_dirs=[numpy.get_include()]
)