"""Common optimization problems involving the prox operator.

.. currentmodule:: prx.problems

.. autosummary::
    :toctree:

    l1_minimization
    least_squares

"""

from .l1_minimization import *
from .least_squares import *

__all__ = l1_minimization.__all__ + least_squares.__all__
