"""segyml: SEG-Y seismic data library with ML framework integration.

Usage:
    >>> import segyml
    >>> data, headers = segyml.load("survey.segy")
    >>> tensor, headers = segyml.load("survey.segy", backend="torch")
    >>> segyml.save("output.segy", data, dt=4000)
    >>> n = segyml.asc2segy("raw_asc/", "output.segy")
    >>> segyml.wiggle(data[:, :100], dt=0.004)
"""

from .api import load, save, asc2segy
from .visualize import wiggle, image
from ._headers import (
    FORMAT_IBM_FLOAT,
    FORMAT_INT32,
    FORMAT_INT16,
    FORMAT_IEEE_FLOAT32,
    FORMAT_INT8,
)

__version__ = "0.1.0"
__all__ = [
    "load", "save", "asc2segy",
    "wiggle", "image",
    "FORMAT_IBM_FLOAT", "FORMAT_INT32", "FORMAT_INT16",
    "FORMAT_IEEE_FLOAT32", "FORMAT_INT8",
]
