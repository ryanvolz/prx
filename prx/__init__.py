"""Optimization algorithms based on the proximal operator.

.. currentmodule:: prx


Standard Problems
-----------------

.. autosummary::
   :toctree:

    problems


Specifying Linear Operators
---------------------------

.. autosummary::
    :toctree:

    operator_class
    operator_utils


Solving Custom Problems
-----------------------

.. autosummary::
    :toctree:

    algorithms
    objectives
    separable_array


Functions, Gradients, and Prox Operators
----------------------------------------

.. autosummary::
    :toctree:

    fun
    grad
    prox

"""
from . import algorithms
from . import fun
from . import functions
from . import grad
from . import objectives
from . import problems
from .problems import *
from . import prox
from .operator_class import *
from .operator_utils import *
from .separable_array import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
