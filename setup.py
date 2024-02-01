#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 April 06, 17:33:28
@last modified : 2023 April 24, 15:59:09
@last modified : 2023 April 24, 15:59:09
"""

import os
import logging
from distutils.core import setup, Extension

logging.basicConfig(level=logging.INFO)

os.environ["CC"] = "g++"

fast_filters_module = Extension(
    "fast_filters",
    sources=["atoms_detection/fast_filters.cpp"],
)

setup(
    name="atoms_detection",
    version="0.0.1a0",
    description="",
    ext_modules=[fast_filters_module],
)
