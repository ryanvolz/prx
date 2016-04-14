#-----------------------------------------------------------------------------
# Copyright (c) 2014-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------
"""Objective classes for polynomial functions.

.. currentmodule:: prx.objectives.polynomials

Polynomials
-----------

.. autosummary::
    :toctree:

    Quadratic
    Affine
    Const

Related Indicators
------------------

.. autosummary::
    :toctree:

    PointInd

"""

from __future__ import division
import numpy as np

from .objective_classes import (
    _class_docstring_wrapper, _init_docstring_wrapper,
    BaseObjective, IndicatorObjective,
)
from ..fun.norms import l2normsqhalf
from ..operator_class import DiagLinop

__all__ = [
    'Quadratic', 'Affine', 'Const', 'PointInd',
]

class Quadratic(BaseObjective):
    __doc__ = _class_docstring_wrapper(
    """Objective class for a quadratic function.

    This function is defined as::

        f(x) = s*l2normsqhalf(A(x)) + Re(<b, x>) + c,

    where `A` is a linear operator or matrix, `b` is an array specifying the
    linear term, `c` is a scalar constant, and `s` is a scaling constant.

    {common_summary}


    Attributes
    ----------

    A : LinearOperator
        Linear operator defining the quadratic term.

    b : array
        Linear term.

    c : float | int
        Constant term.

    s : float | int
        Scaling of quadratic term.

    {common_attributes}


    See Also
    --------

    Affine : Subtype with a zero quadratic term.
    Const : Subtype with zero quadratic and linear terms.
    .BaseObjective : Parent class.


    Notes
    -----

    {common_notes}

    """)
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

    @_init_docstring_wrapper
    def __init__(self, A=None, b=None, c=None, s=1, scale=None, stretch=None,
                 shift=None, linear=None, const=None):
        """Create Objective object for a quadratic function.

        {common_summary}


        Parameters
        ----------

        A : LinearOperator, optional
            Linear operator defining the quadratic term: l2normsqhalf(A(x)).

        b : array, optional
            Vector defining the linear term: Re(<b, x>).

        c : float | int, optional
            Constant term.

        s : float | int, optional
            Scaling of quadratic term.

        {common_params}

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

class PointInd(IndicatorObjective):
    __doc__ = _class_docstring_wrapper(
    """Objective class for the point indicator function.

    {common_summary}


    Attributes
    ----------

    p : array
        The point at which this function is defined.

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.


    Notes
    -----

    The indicator function is zero at the given point p and infinity
    everywhere else. Its gradient is undefined.

    The prox operator returns the defining point.

    """)
    @_init_docstring_wrapper
    def __init__(self, p, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """Create Objective that defines an indicator function.

        {common_summary}

        Since this objective is an indicator, `scale` can be eliminated::

            s*f(a*x + b) => f(a*x + b)


        Parameters
        ----------

        p : array
            The point at which this function is defined.

        {common_params}

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
        return _polynomial_classes.Affine

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
