#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np

from .func_classes import (
    prepend_docstring,
    NormFunctionWithGradProx, NormSqFunctionWithGradProx,
    NormBallWithGradProx, IndicatorWithGradProx,
)
from .grad_funcs import grad_l2sqhalf
from .prox_funcs import (
    proj_l2, proj_linf, proj_zeros, prox_l1, prox_l1l2, prox_l2,
    prox_l2sqhalf,
)
from .norms import l1norm, l1l2norm, l2norm, l2normsqhalf, linfnorm

__all__ = ['L1Norm', 'L1L2Norm', 'L2Norm', 'L2NormSqHalf', 'L2BallInd',
           'LInfBallInd', 'ZerosInd']

###***************************************************************************
### Useful function classes for function-prox objects ************************
###***************************************************************************

class L1Norm(NormFunctionWithGradProx):
    """Function and prox operator for the l1-norm, sum(abs(x)).


    See Also
    --------

    NormFunctionWithGradProx : Parent class.


    Notes
    -----

    fun(x) = sum(abs(x))

    The prox operator is soft thresholding:

    .. math::

        st(x[k], lmbda) = { (1 - lmbda/|x[k]|)*x[k]  if |x[k]| > lmbda
                          {           0              otherwise

    """
    @property
    def _conjugate_class(self):
        return LInfBallInd
    fun = staticmethod(l1norm)
    prox = staticmethod(prox_l1)

class L1L2Norm(NormFunctionWithGradProx):
    """Function and prox operator for the combined l1- and l2-norm.


    See Also
    --------

    NormFunctionWithGradProx : Parent class.


    Notes
    -----

    fun(x) = l1norm(l2norm(x, axis))

    The prox operator is block soft thresholding (for 'k' NOT along axis):

    .. math::

      bst(x[k, :], lmbda) = {(1 - lmbda/||x[k]||_2)*x[k]  if ||x[k]||_2 > lmbda
                            {          0                  otherwise

    """
    @prepend_docstring(NormFunctionWithGradProx.__init__)
    def __init__(self, axis=-1, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """axis : int or None
            The axes over which to take the l2-norm
            (axis=None specifies all axes and is equivalent to L2Norm).

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
        """Prox operator of combined l1- and l2-norm (l2 over axis=self.axis)."""
        return prox_l1l2(x, lmbda=lmbda, axis=self._axis)

class L2Norm(NormFunctionWithGradProx):
    """Function and prox operator for the l2-norm, sqrt(sum(abs(x)**2)).


    See Also
    --------

    NormFunctionWithGradProx : Parent class.


    Notes
    -----

    fun(x) = sqrt(sum(abs(x)**2))

    The prox operator is the block thresholding function for block=(all of x):

    .. math::

        bst(x, lmbda) = { (1 - lmbda/||x||_2)*x  if ||x||_2 > lmbda
                        {           0            otherwise

    """
    @property
    def _conjugate_class(self):
        return L2BallInd
    fun = staticmethod(l2norm)
    prox = staticmethod(prox_l2)

class L2NormSqHalf(NormSqFunctionWithGradProx):
    """Function with gradient and prox operator for half the squared l2-norm.


    See Also
    --------

    NormSqFunctionWithGradProx : Parent class.


    Notes
    -----

    fun(x) = 0.5*sum(abs(x)**2))

    The prox operator is the shrinkage function:
        shrink(x, lmbda) = x/(1 + lmbda)

    """
    @property
    def _conjugate_class(self):
        return L2NormSqHalf
    fun = staticmethod(l2normsqhalf)
    grad = staticmethod(grad_l2sqhalf)
    prox = staticmethod(prox_l2sqhalf)

class L2BallInd(NormBallWithGradProx):
    """Function and prox operator for the indicator of the l2-ball.


    See Also
    --------

    NormBallWithGradProx : Parent class.


    Notes
    -----

    The indicator function is zero for vectors inside the ball, infinity for
    vectors outside the ball.

    The prox operator is Euclidean projection onto the l2-ball.

    """
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

class LInfBallInd(NormBallWithGradProx):
    """Function and prox operator for the indicator of the linf-ball.


    See Also
    --------

    NormBallWithGradProx : Parent class.


    Notes
    -----

    The indicator function is zero for vectors inside the ball, infinity for
    vectors outside the ball.

    The prox operator is Euclidean projection onto the linf-ball.

    """
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

class ZerosInd(IndicatorWithGradProx):
    """Function and prox operator for the indicator of zero elements.


    See Also
    --------

    IndicatorWithGradProx : Parent class.


    Notes
    -----

    The indicator function is zero for vectors with only zeros in the
    specified places, infinity if any of the required zero entries are nonzero.

    The prox operator is Euclidean projection onto the set with
    specified zeros (x[z] is set to 0).

    """
    @prepend_docstring(IndicatorWithGradProx.__init__)
    def __init__(self, z=None, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """The parameter 'z' must be a boolean array giving the zero locations.

        """
        if z is None:
            raise ValueError('Must specify zero locations (z)!')

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
            linear=linear, const=const)

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
