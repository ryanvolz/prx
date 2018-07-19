"""Optimization objectives involving the prox operator.

.. currentmodule:: prx.objectives


Common Objectives
-----------------

.. autosummary::
   :toctree:

   covariance_estimation
   l1_minimization
   least_squares


General Objectives
------------------

.. autosummary::
   :toctree:

   split_objectives


Customization
-------------

.. autosummary::
   :toctree:

   base

"""

from . import base
from .covariance_estimation import *
from .l1_minimization import *
from .least_squares import *
from .split_objectives import *

__all__ = (
    covariance_estimation.__all__ + l1_minimization.__all__ +
    least_squares.__all__ + split_objectives.__all__
)
