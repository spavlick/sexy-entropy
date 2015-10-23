#!/usr/bin/env python

import glob
import os
try:
    from setuptools import setup
    have_setuptools = True
except ImportError:
    from distutils.core import setup
    have_setuptools = False

kwargs = {'name': 'uLSIF',
          'version': '0.7.0',
          'packages': ['uLSIF'],

          }

if have_setuptools:
    kwargs.update({
        # Required dependencies
        'install_requires': ['numpy', 'h5py', 'matplotlib'],

        })

setup(**kwargs)
