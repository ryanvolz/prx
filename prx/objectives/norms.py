# ----------------------------------------------------------------------------
# Copyright (c) 2014-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Objective classes for norm and norm ball indicator functions.

.. currentmodule:: prx.objectives.norms

Norms
-----

.. autosummary::
    :toctree:

    L1Norm
    L1L2Norm
    L2Norm
    L2NormSqHalf
    LInfNorm

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

from ..fun.norms import l1l2norm, l1norm, l2norm, l2normsqhalf, linfnorm
from ..grad import grad_l2sqhalf
from ..prox.norms import prox_l1, prox_l1l2, prox_l2, prox_l2sqhalf, prox_linf
from ..prox.projections import proj_l1, proj_l2, proj_linf
from .objective_classes import (NormBallObjective, NormObjective,
                                NormSqObjective, _class_docstring_wrapper,
                                _init_docstring_wrapper)

__all__ = (
    'L1Norm', 'L1L2Norm', 'L2Norm', 'L2NormSqHalf', 'LInfNorm',
    'L1BallInd', 'L2BallInd', 'LInfBallInd',
)


class L1Norm(NormObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the l1-norm, ``sum(abs(x))``.

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .NormObjective : Parent class.


    Notes
    -----

    ``fun(x) = sum(abs(x))``

    The prox operator is soft thresholding::

        st(v[k], l) = {{ v[k] - l*v[k]/abs(v[k])  if abs(v[k]) > l
                      {{           0              otherwise

    """
    )

    @property
    def _conjugate_class(self):
        return LInfBallInd
    fun = staticmethod(l1norm)
    prox = staticmethod(prox_l1)


class L1L2Norm(NormObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the combined l1- and l2-norm.

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .NormObjective : Parent class.


    Notes
    -----

    ``fun(x) = l1norm(l2norm(x, axis))``

    The prox operator is block soft thresholding::

        bst(v[k, :], l) =
         {{ v[k, :] - l*v[k, :]/l2norm(v[k, :])  if l2norm(v[k, :]) > l
         {{                 0                    otherwise

    """
    )

    @_init_docstring_wrapper
    def __init__(self, axis=-1, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """Create Objective that defines a norm function.

        {common_summary}

        Since this objective is a norm, `stretch` can be eliminated by
        absorbing it into `scale` and `shift`::

            s*f(a*x + b) => a*s*f(x + b/a)


        Parameters
        ----------

        axis : int | tuple of ints, optional
            Axis or axes over which to calculate the l2-norm. The prox
            operator is applied over all remaining axes.

        {common_params}

        """
        self._axis = axis

        super(L1L2Norm, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

    @property
    def axis(self):
        return self._axis

    def fun(self, x):
        """Combined l1- and l2-norm with l2-norm taken over axis=self.axis."""
        return l1l2norm(x, self._axis)

    def prox(self, x, lmbda=1):
        """Prox op of combined l1- and l2-norm (l2 over axis=self.axis)."""
        return prox_l1l2(x, lmbda=lmbda, axis=self._axis)


class L2Norm(NormObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the l2-norm, ``sqrt(sum(abs(x)**2))``.

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .NormObjective : Parent class.


    Notes
    -----

    ``fun(x) = sqrt(sum(abs(x)**2))``

    The prox operator is block soft thresholding for block=(all of v)::

        bst(v, l) = {{ v - l*v/l2norm(v)  if l2norm(v) > l
                    {{       0            otherwise

    """
    )

    @property
    def _conjugate_class(self):
        return L2BallInd
    fun = staticmethod(l2norm)
    prox = staticmethod(prox_l2)


class L2NormSqHalf(NormSqObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for half the squared l2-norm, 0.5*sum(abs(x)**2)).

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .NormSqObjective : Parent class.


    Notes
    -----

    ``fun(x) = 0.5*sum(abs(x)**2))``

    The prox operator is the shrinkage function::

        shrink(v, l) = v/(1 + l)

    """
    )

    @property
    def _conjugate_class(self):
        return L2NormSqHalf
    fun = staticmethod(l2normsqhalf)
    grad = staticmethod(grad_l2sqhalf)
    prox = staticmethod(prox_l2sqhalf)


class LInfNorm(NormObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the linf-norm, ``max(abs(x))``.

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .NormObjective : Parent class.


    Notes
    -----

    ``fun(x) = max(abs(x))``

    The prox operator is the peak shrinkage function, which minimizes the
    maximum value for a reduction of lmbda in the l1-norm.

    """
    )

    @property
    def _conjugate_class(self):
        return L1BallInd
    fun = staticmethod(linfnorm)
    prox = staticmethod(prox_linf)


class L1BallInd(NormBallObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the indicator of the l1-ball.

    {common_summary}


    Attributes
    ----------

    radius : float | int
        Radius of the l1-ball indicator.

    {common_attributes}


    See Also
    --------

    .NormBallObjective : Parent class.


    Notes
    -----

    The indicator function is zero for vectors inside the ball, infinity
    for vectors outside the ball.

    The prox operator is Euclidean projection onto the l1-ball.

    """
    )

    @property
    def _conjugate_class(self):
        return LInfNorm

    def fun(self, x):
        """Indicator function for the l1-ball with radius=self.radius."""
        nrm = l1norm(x)
        eps = np.finfo(nrm.dtype).eps
        rad = self.radius
        if nrm <= rad*(1 + 10*eps):
            return 0
        else:
            return np.inf

    def prox(self, x, lmbda=1):
        """Projection onto the l1-ball with radius=self.radius."""
        return proj_l1(x, radius=self.radius)


class L2BallInd(NormBallObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the indicator of the l2-ball.

    {common_summary}


    Attributes
    ----------

    radius : float | int
        Radius of the l2-ball indicator.

    {common_attributes}


    See Also
    --------

    .NormBallObjective : Parent class.


    Notes
    -----

    The indicator function is zero for vectors inside the ball, infinity
    for vectors outside the ball.

    The prox operator is Euclidean projection onto the l2-ball.

    """
    )

    @property
    def conjugate_class(self):
        return L2Norm

    def fun(self, x):
        """Indicator function for the l2-ball with radius=self.radius."""
        nrm = l2norm(x)
        eps = np.finfo(nrm.dtype).eps
        rad = self.radius
        if nrm <= rad*(1 + 10*eps):
            return 0
        else:
            return np.inf

    def prox(self, x, lmbda=1):
        """Projection onto the l2-ball with radius=self.radius."""
        return proj_l2(x, radius=self.radius)


class LInfBallInd(NormBallObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the indicator of the linf-ball.

    {common_summary}


    Attributes
    ----------

    radius : float | int
        Radius of the linf-ball indicator.

    {common_attributes}


    See Also
    --------

    .NormBallObjective : Parent class.


    Notes
    -----

    The indicator function is zero for vectors inside the ball, infinity
    for vectors outside the ball.

    The prox operator is Euclidean projection onto the linf-ball.

    """
    )

    @property
    def _conjugate_class(self):
        return L1Norm

    def fun(self, x):
        """Indicator function for the linf-ball with radius=self.radius."""
        nrm = linfnorm(x)
        eps = np.finfo(nrm.dtype).eps
        rad = self.radius
        if nrm <= rad*(1 + 10*eps):
            return 0
        else:
            return np.inf

    def prox(self, x, lmbda=1):
        """Projection onto the linf-ball with radius=self.radius."""
        return proj_linf(x, radius=self.radius)
