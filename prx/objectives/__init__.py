"""Useful objective functions.

.. currentmodule:: prx.objectives

Common Objectives
-----------------

.. autosummary::
    :toctree:

    norms
    indicators
    polynomials
    covariance

Customizable Objectives
-----------------------

.. autosummary::
    :toctree:

    Objective
    SeparableObjective
    objective_classes

"""

from .covariance import *
from .indicators import *
from .norms import *
from . import objective_classes
from .objective_classes import Objective, SeparableObjective
from .polynomials import *
