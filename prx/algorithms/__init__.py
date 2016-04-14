"""Optimization algorithms involving the prox operator.

.. currentmodule:: prx.algorithms

Algorithms
----------

.. autosummary::
    :toctree:

    admm_algos
    proxgrad_algos


In Development
--------------

.. autosummary::
    :toctree:

    primaldual_algos

"""

from .admm_algos import *
from .primaldual_algos import *
from .proxgrad_algos import *
