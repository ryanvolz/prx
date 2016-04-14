#-----------------------------------------------------------------------------
# Copyright (c) 2014-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------
"""Objective classes for indicator functions.

.. currentmodule:: prx.objectives.indicators

General Indicators
------------------

.. autosummary::
    :toctree:

    NNegInd
    NPosInd
    ZerosInd
    PointInd

Norm Ball Indicators
--------------------

.. autosummary::
    :toctree:

    L1BallInd
    L2BallInd
    LInfBallInd

"""

from __future__ import division
import numpy as np

from .objective_classes import (
    _class_docstring_wrapper, _init_docstring_wrapper,
    IndicatorObjective,
)
from ..prox.projections import (
    proj_nneg, proj_npos, proj_zeros,
)
from .norms import L1BallInd, L2BallInd, LInfBallInd
from .polynomials import PointInd

__all__ = [
    'NNegInd', 'NPosInd', 'ZerosInd',
    'L1BallInd', 'L2BallInd', 'LInfBallInd', 'PointInd'
]

class NNegInd(IndicatorObjective):
    __doc__ = _class_docstring_wrapper(
    """Objective class for the non-negative indicator function.

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.


    Notes
    -----

    The indicator function is zero for vectors with only non-negative entries
    and infinity if any of the entries are negative.

    The prox operator is Euclidean projection onto the non-negative halfspace,
    i.e. the negative entries are set to zero.

    """)
    @property
    def _conjugate_class(self):
        return NPosInd

    def fun(self, x):
        """Indicator function for non-negative values."""
        if np.any(x < 0):
            return np.inf
        else:
            return 0

    def prox(self, x, lmbda=1):
        """Projection onto the non-negative reals (negatives set to zero)."""
        return proj_nneg(x)

class NPosInd(IndicatorObjective):
    __doc__ = _class_docstring_wrapper(
    """Objective class for the non-positive indicator function.

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.


    Notes
    -----

    The indicator function is zero for vectors with only non-positive entries
    and infinity if any of the entries are positive.

    The prox operator is Euclidean projection onto the non-positive halfspace,
    i.e. the positive entries are set to zero.

    """)
    @property
    def _conjugate_class(self):
        return NNegInd

    def fun(self, x):
        """Indicator function for non-positive values."""
        if np.any(x > 0):
            return np.inf
        else:
            return 0

    prox = staticmethod(proj_npos)
    def prox(self, x, lmbda=1):
        """Projection onto the non-positive reals (positives set to zero)."""
        return proj_npos(x)

class ZerosInd(IndicatorObjective):
    __doc__ = _class_docstring_wrapper(
    """Objective class for the indicator function of prescribed zero elements.

    {common_summary}


    Attributes
    ----------

    z : boolean array
        Array specifying the zero locations, requiring x[z] == 0.

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.


    Notes
    -----

    The indicator function is zero for vectors with only zeros in the
    specified places, infinity if any of the required zero entries are nonzero.

    The prox operator is Euclidean projection onto the set with
    specified zeros (x[z] is set to 0).

    """)
    @_init_docstring_wrapper
    def __init__(self, z, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """Create Objective that defines an indicator function.

        {common_summary}

        Since this objective is an indicator, `scale` can be eliminated::

            s*f(a*x + b) => f(a*x + b)


        Parameters
        ----------

        z : boolean array
            Array specifying the zero locations, requiring x[z] == 0.

        {common_params}

        """
        # stretch can be eliminated by bringing into shift
        # (since multiplying does not change zero locations)
        if stretch is not None and shift is not None:
            shift = shift/stretch
        stretch = None

        self._z = z

        # we can also eliminate scaling,
        # but this is taken care of by parent class
        super(ZerosInd, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const,
        )

    @property
    def _conjugate_class(self):
        return ZerosInd

    @property
    def _conjugate_args(self):
        # The conjugate of the zero indicator is the indicator for the
        # complimentary set of zeros.
        z = ~self._z
        kwargs = super(ZerosInd, self)._conjugate_args
        kwargs.update(z=z)
        return kwargs

    @property
    def z(self):
        return self._z

    def fun(self, x):
        """Indicator function for zero elements z."""
        if np.any(x[self._z] != 0):
            return np.inf
        else:
            return 0

    def prox(self, x, lmbda=1):
        """Projection onto the set with specified zeros (x[z] is set to 0)"""
        return proj_zeros(x, z=self._z)
