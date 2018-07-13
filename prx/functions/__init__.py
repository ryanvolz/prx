"""Useful function classes.

.. currentmodule:: prx.functions

Common Functions
----------------

.. autosummary::
    :toctree:

    norms
    indicators
    polynomials
    covariance

Customizable Functions
----------------------

.. autosummary::
    :toctree:

    Function
    SeparableFunction
    base

"""

from .covariance import *
from .indicators import *
from .norms import *
from . import base
from .base import Function, SeparableFunction
from .polynomials import *
