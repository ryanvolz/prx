"""Optimization algorithms based on the proximal operator.

.. currentmodule:: prx


Standard Problems
-----------------

.. autosummary::
   :toctree:

   objectives.covariance_estimation
   objectives.l1_minimization
   objectives.least_squares


Custom Problems
---------------

Objectives
**********

.. autosummary::
   :toctree:

   objectives
   objectives.split_objectives
   ~objectives.split_objectives.SplitObjective

Functions
*********

.. autosummary::
   :toctree:

   functions
   functions.covariance
   functions.norms
   functions.indicators
   functions.polynomials
   functions.SeparableFunction
   separable_array.separray

Algorithms
**********

.. autosummary::
   :toctree:

   algorithms
   algorithms.admm_algos
   algorithms.proxgrad_algos


Specifying Linear Operators
---------------------------

.. autosummary::
   :toctree:

   operator_class
   operator_utils


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
from .objectives import *
from . import prox
from .operator_class import *
from .operator_utils import *
from .separable_array import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
