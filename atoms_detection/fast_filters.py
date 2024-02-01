import os
import math
import ctypes
import numpy as np
from glob import glob
from functools import partial
from multiprocessing import Pool
from utils.paths import LIB_PATH

# Load the shared library
try:
    lib_path = glob(os.path.join(LIB_PATH, "fast_filters*.so"))[0]
    lib = ctypes.cdll.LoadLibrary(lib_path)
except OSError as e:
    raise OSError(
            "Did you compile the shared library? Please run `python setup.py build_ext`"
            ) from e

# Define the functions arguments and return type
lib.reflect_borders.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ]
lib.reflect_borders.restype = None
lib.median_filter.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ]
lib.median_filter.restype = None


# Create the median filter function in Python
def median_filter(data, window_size, width, height):
    out = np.zeros((height, width), dtype=np.float32)
    lib.median_filter(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            width,
            height,
            window_size,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
    return out


# Create the reflecting borders function in Python
def reflecting_borders(data, span):
    out = np.zeros(
            (data.shape[0] + 2 * span, data.shape[1] + 2 * span), dtype=np.float32
            )
    lib.reflect_borders(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            data.shape[1],
            data.shape[0],
            span,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
    return out

def relecting_borders_py(data, span):
    data = np.pad(data, span, mode="reflect")
    return data


def median_filter_parallel(data, window_size, splits=None, inplace=False):
    if splits is None:
        splits = (
                int(math.sqrt(os.cpu_count())) + 1
                )  # at least 1 split per core (n_splits = splits**2 because 2 dimensions)
    out = data if inplace else data.copy()
    height, width = data.shape
    span = window_size // 2
    padded = reflecting_borders(data, span)
    # TODO: split the data evenly and take into account the remaining subarray shape if not all height/width not divisble by the number of splits
    subarrays = []
    for i in range(splits):
        for j in range(splits):
            istart = i * height // splits
            jstart = j * width // splits
            iend = istart + height // splits + 2 * span
            jend = jstart + width // splits + 2 * span
            subarrays.append(padded[istart:iend, jstart:jend])
    f = partial(
            median_filter,
            window_size=window_size,
            width=width // splits,
            height=height // splits,
            )
    with Pool(processes=splits * splits) as pool:
        filtered_subs = list(pool.map(f, subarrays))
    # TODO: merge subarrays faster (maybe with cython)
    for i in range(splits):
        for j in range(splits):
            out[
                    i * height // splits : (i + 1) * height // splits,
                    j * width // splits : (j + 1) * width // splits,
                    ] = filtered_subs[i * splits + j]
    return out
