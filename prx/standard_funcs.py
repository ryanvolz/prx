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
    prepend_docstring, FunctionWithGradProx,
    NormFunctionWithGradProx, NormSqFunctionWithGradProx,
    NormBallWithGradProx, IndicatorWithGradProx,
)
from .grad_funcs import grad_l2sqhalf
from .prox_funcs import (
    proj_l1, proj_l2, proj_linf, proj_zeros,
    prox_l1, prox_l1l2, prox_l2, prox_l2sqhalf, prox_linf,
)
from .norms import l1norm, l1l2norm, l2norm, l2normsqhalf, linfnorm
from .operator_class import DiagLinop

__all__ = [
    'L1Norm', 'L1L2Norm', 'L2Norm', 'L2NormSqHalf', 'LInfNorm',
    'L1BallInd', 'L2BallInd', 'LInfBallInd',
    'Quadratic', 'Affine', 'Const', 'PointInd',
    'ZerosInd'
]

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

class LInfNorm(NormFunctionWithGradProx):
    """Function and prox operator for the linf-norm, max(abs(x)).


    See Also
    --------

    NormFunctionWithGradProx : Parent class.


    Notes
    -----

    fun(x) = max(abs(x))

    The prox operator is the peak shrinkage function, which minimizes the
    maximum value for a reduction of lmbda in the l1-norm.

    """
    @property
    def _conjugate_class(self):
        return L1BallInd
    fun = staticmethod(linfnorm)
    prox = staticmethod(prox_linf)

class L1BallInd(NormBallWithGradProx):
    """Function and prox operator for the indicator of the l1-ball.


    See Also
    --------

    NormBallWithGradProx : Parent class.


    Notes
    -----

    The indicator function is zero for vectors inside the ball, infinity for
    vectors outside the ball.

    The prox operator is Euclidean projection onto the l1-ball.

    """
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
        """Projection onto the linf-ball with radius=self.radius."""
        return proj_l1(x, radius=self.radius)

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

class Quadratic(FunctionWithGradProx):
    """Function and prox operator for a quadratic function.

    This function is defined as

    .. math::

        f(x) = 0.5*s*(||Ax||_2)^2 + Re(<b, x>) + c,

    where `A` is a linear operator or matrix, `b` is an array specifying the
    linear term, `c` is a scalar constant, and `s` is a scaling constant.

    .. automethod:: __init__


    Attributes
    ----------

    A : matrix or function
        Linear operator defining the quadratic term.

    b : ndarray
        Linear term.

    c : float or int
        Constant term.

    s : float or int
        Scaling of quadratic term.


    See Also
    --------

    Affine : Subtype with a zero quadratic term.
    Const : Subtype with zero quadratic and linear terms.
    LinearFunctionWithGradProx : Parent class.

    """
    def __new__(cls, A=None, b=None, c=None, s=1, scale=None, stretch=None,
                shift=None, linear=None, const=None):
        # if quadratic term is None, want to define an Affine or Const
        if A is None:
            # if linear terms are None, want to define a Const function
            if b is None and linear is None:
                obj = super(Quadratic, cls).__new__(
                    Const, scale=scale, stretch=stretch, shift=shift,
                    linear=linear, const=const,
                )
                return obj
            # otherwise we do want an Affine function
            # must specify explicitly since this is also called from Const
            else:
                obj = super(Quadratic, cls).__new__(
                    Affine, scale=scale, stretch=stretch, shift=shift,
                    linear=linear, const=const,
                )
                return obj
        # otherwise we do want a Quadratic function
        # must specify explicitly since this is also called from subtypes
        else:
            obj = super(Quadratic, cls).__new__(
                Quadratic, scale=scale, stretch=stretch, shift=shift,
                linear=linear, const=const,
            )
            return obj

    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, A=None, b=None, c=None, s=1, scale=None, stretch=None,
                 shift=None, linear=None, const=None):
        """A : LinearOperator
            Linear operator defining the quadratic term: 0.5*(||Ax||_2)^2.

        b : ndarray
            Vector defining the linear term: Re(<b, x>).

        c : float or int
            Constant term.

        s : float or int
            Scaling of quadratic term.

        """
        # change 'None's to identities
        if A is None:
            A = DiagLinop(0)
        if b is None:
            b = 0
        if c is None:
            c = 0
        if scale is None:
            scale = 1
        if stretch is None:
            stretch = 1
        if shift is None:
            shift = 0
        if linear is None:
            linear = 0
        if const is None:
            const = 0

        # NOTE: can't do anything to simplify or combine shift term
        # since we don't know what size input the LinearOperator accepts

        # combine constant terms
        # TODO replace multiply/sum with inner1d when it is available in numpy
        c = scale*c + const
        const = None

        # combine linear terms
        b = scale*np.conj(stretch)*b + linear
        linear = None

        # combine scale and stretch into quadratic coefficient
        s = s*scale*np.abs(stretch)**2
        scale = None
        stretch = None

        self._A = A
        self._b = b
        self._bconj = np.conj(b)
        self._c = c
        self._s = s

        super(Quadratic, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const,
        )

    #@property
    #def _conjugate_class(self):
        #return Quadratic

    #@property
    #def _conjugate_args(self):
        ## The conjugate of the point indicator is the affine function with
        ## the point as the linear term.
        #kwargs = super(PointInd, self)._conjugate_args
        #kwargs.update(linear, self._p)
        #return kwargs

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def s(self):
        return self._s

    def fun(self, x):
        """Quadratic function."""
        q = self._s*l2normsqhalf(self._A(x))
        # TODO replace multiply/sum with inner1d when it is available in numpy
        l = np.multiply(self._bconj, x).sum().real
        return q + l + self._c

    def grad(self, x):
        """Gradient of quadratic function."""
        return self._s*self._A.H(self._A(x)) + self._b.real

    #def prox(self, x, lmbda=1):
        #"""Prox of an quadratic function."""
        #return self._A.ideninv(x - lmbda*self._b, lmbda*self._s)

class Affine(Quadratic):
    #@property
    #def _conjugate_class(self):
        #return PointInd

    #@property
    #def _conjugate_args(self):
        ## The conjugate of the point indicator is the affine function with
        ## the point as the linear term.
        #kwargs = super(PointInd, self)._conjugate_args
        #kwargs.update(linear, self._p)
        #return kwargs

    def fun(self, x):
        """Affine function."""
        # TODO replace multiply/sum with inner1d when it is available in numpy
        return np.multiply(self._bconj, x).sum().real + self._c

    def grad(self, x):
        """Gradient of affine function."""
        return self._b.real

    def prox(self, x, lmbda=1):
        """Prox of an affine function."""
        return x - lmbda*self._b

class Const(Affine):
    #@property
    #def _conjugate_class(self):
        #return Infinity

    def fun(self, x):
        """Constant function."""
        return self._c

    @staticmethod
    def grad(x):
        """Zero vector, the gradient of a constant."""
        return np.zeros_like(x)

    @staticmethod
    def prox(x, lmbda=1):
        """Identity function, the prox operator of a constant function."""
        return x

class PointInd(IndicatorWithGradProx):
    """Function and prox operator for the indicator of a point.


    See Also
    --------

    IndicatorWithGradProx : Parent class.


    Attributes
    ----------

    p : ndarray
        The point at which this function is defined.


    Notes
    -----

    The indicator function is zero at the given point p and infinity
    everywhere else. Its gradient is undefined.

    The prox operator returns the defining point.

    """
    @prepend_docstring(IndicatorWithGradProx.__init__)
    def __init__(self, p, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """p : ndarray
            The point at which this function is defined.

        """
        # linear can be eliminated by evaluating at point and bringing into
        # const
        if linear is not None:
            linconst = np.vdot(linear, p)
            if const is not None:
                const = const + linconst
            else:
                const = linconst
        linear = None

        # stretch and shift can be eliminated by absorbing into point
        if shift is not None:
            p = p - shift
        shift = None
        if stretch is not None:
            p = p/stretch
        stretch = None

        self._p = p

        # we can also eliminate scaling,
        # but this is taken care of by parent class
        super(PointInd, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const,
        )

    @property
    def _conjugate_class(self):
        return Affine

    @property
    def _conjugate_args(self):
        # The conjugate of the point indicator is the affine function with
        # the point as the linear term.
        kwargs = super(PointInd, self)._conjugate_args
        kwargs.update(b, self._p)
        return kwargs

    @property
    def p(self):
        return self._p

    def fun(self, x):
        """Indicator function for the point p."""
        if np.allclose(x, self._p):
            return 0
        else:
            return np.inf

    def prox(self, x, lmbda=1):
        """Projection onto the point p. (Always returns p.)"""
        return self._p

class ZerosInd(IndicatorWithGradProx):
    """Function and prox operator for the indicator of zero elements.


    Attributes
    ----------

    z : boolean ndarray
        Array specifying the zero locations at True entries.


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
    def __init__(self, z, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """z : boolean ndarray
            Array specifying the zero locations at True entries.

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
