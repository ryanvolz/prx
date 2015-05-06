"""Optimization algorithms based on the proximal operator.

.. currentmodule:: prx

Standard Problems
-----------------

.. autosummary::
    :toctree:

    standard_probs

"""
from .amp_algos import *
from .grad_funcs import *
from .ist_algos import *
from .func_classes import *
from .norms import *
from .operator_class import *
from .operator_utils import *
from .prox_algos import *
from .prox_funcs import *
from .separable_array import *
from .standard_funcs import *
from .standard_probs import *
from .thresholding import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
