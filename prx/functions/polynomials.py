# ----------------------------------------------------------------------------
# Copyright (c) 2014-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Function classes for polynomial functions and related indicators."""

from __future__ import division

import numpy as np

from . import base as _base
from ..fun.norms import l2normsqhalf
from ..operator_class import DiagLinop

__all__ = (
    'Quadratic', 'Affine', 'Const', 'PointInd',
)


class Quadratic(_base.BaseFunction):
    """Function class for a quadratic function.

    This function is defined as::

        f(x) = s*l2normsqhalf(A(x)) + Re(<b, x>) + c,

    where `A` is a linear operator or matrix, `b` is an array specifying
    the linear term, `c` is a scalar constant, and `s` is a scaling
    constant.

    {function_summary}


    Attributes
    ----------

    {quadratic_params}

    {function_attributes}


    See Also
    --------

    Affine : Subtype with a zero quadratic term.
    Const : Subtype with zero quadratic and linear terms.
    .BaseFunction : Parent class.


    Notes
    -----

    {function_notes}

    """

    _doc_quadratic_params = """
    A : :class:`.LinearOperator`
        Linear operator defining the quadratic term.

    b : array
        Linear term.

    c : float | int
        Constant term.

    s : float | int
        Scaling of quadratic term.

    """

    def __new__(
        cls, A=None, b=None, c=None, s=1, scale=None, stretch=None, shift=None,
        linear=None, const=None,
    ):
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

    def __init__(
        self, A=None, b=None, c=None, s=1, scale=None, stretch=None,
        shift=None, linear=None, const=None,
    ):
        """Create Function object for a quadratic function.

        {init_summary}


        Parameters
        ----------

        {quadratic_params}

        {init_params}

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

    # @property
    # def _conjugate_class(self):
    #     return Quadratic
    #
    # @property
    # def _conjugate_args(self):
    #     # The conjugate of the point indicator is the affine function with
    #     # the point as the linear term.
    #     kwargs = super(PointInd, self)._conjugate_args
    #     kwargs.update(b=self._p)
    #     return kwargs

    @property
    def A(self):
        """Linear operator defining the quadratic term."""
        return self._A

    @property
    def b(self):
        """Vector defining the linear term: Re(<b, x>)."""
        return self._b

    @property
    def c(self):
        """Constant term."""
        return self._c

    @property
    def s(self):
        """Scaling of quadratic term."""
        return self._s

    def fun(self, x):
        """Quadratic function."""
        quad = self._s*l2normsqhalf(self._A(x))
        # TODO replace multiply/sum with inner1d when it is available in numpy
        lin = np.multiply(self._bconj, x).sum().real
        return quad + lin + self._c

    def grad(self, x):
        """Gradient of quadratic function."""
        return self._s*self._A.H(self._A(x)) + self._b.real

    # def prox(self, x, lmbda=1):
    #     """Prox of an quadratic function."""
    #     return self._A.ideninv(x - lmbda*self._b, lmbda*self._s)


class Affine(Quadratic):
    # @property
    # def _conjugate_class(self):
    #     return PointInd
    #
    # @property
    # def _conjugate_args(self):
    #     # The conjugate of the point indicator is the affine function with
    #     # the point as the linear term.
    #     kwargs = super(PointInd, self)._conjugate_args
    #     kwargs.update(b=self._p)
    #     return kwargs

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
    # @property
    # def _conjugate_class(self):
    #     return Infinity

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


class PointInd(_base.IndicatorFunction):
    """Function class for the point indicator function.

    {function_summary}


    Attributes
    ----------

    {point_params}

    {function_attributes}


    See Also
    --------

    .IndicatorFunction : Parent class.


    Notes
    -----

    The indicator function is zero at the given point p and infinity
    everywhere else. Its gradient is undefined.

    {function_notes}

    The prox operator returns the defining point.

    """

    _doc_point_params = """
    p : array
        The point at which this function is defined.

    """

    def __init__(
        self, p, scale=None, stretch=None, shift=None, linear=None, const=None,
    ):
        """Create Function that defines an indicator function.

        {init_summary}


        Parameters
        ----------

        {point_params}

        {init_params}

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
        kwargs.update(b=self._p)
        return kwargs

    @property
    def p(self):
        """The point at which this function is defined."""
        return self._p

    def fun(self, x):
        """Indicator function for the point p."""
        if np.allclose(x, self._p):
            return 0
        else:
            return np.inf

    def prox(self, x, lmbda=1):
        """Projection onto the point p (always returns p)."""
        return self._p
