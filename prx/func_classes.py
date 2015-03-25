#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np

from .func_ops import (
    shift_fun, shift_grad, shift_prox,
    stretch_fun, stretch_grad, stretch_prox,
    scale_fun, scale_grad, scale_prox,
    addlinear_fun, addlinear_grad, addlinear_prox,
    addconst_fun
)

def prepend_docstring(parent):
    def decorator(f):
        if f.__doc__ is None:
            f.__doc__ = parent.__doc__
        else:
            f.__doc__ = parent.__doc__ + f.__doc__
        return f

    return decorator

###***************************************************************************
### Base classes for function-prox objects ***********************************
###***************************************************************************

class FunctionWithGradProx(object):
    """Function with gradient and prox operator.

    This class holds a function and its associated gradient and prox operators,
    with optional scaling, stretching, shifting, and linear or constant
    terms added for initialized objects.

    The function is evaluated by calling obj(x) or obj.fun(x).
    Its gradient is evaluated by calling obj.grad(x).
    Its prox operator is evaluated by calling obj.prox(x, lmbda).

    The :meth:`fun`, :meth:`grad`, and :meth:`prox` methods must be
    defined for a particular operator by inheriting from this class. It is
    also necessary to override the attribute :attr:`_conjugate_class` to
    give the class of the conjugate function, if applicable.


    Attributes
    ----------

    conjugate : :class:`FunctionWithGradProx` object
        The corresponding conjugate function/grad/prox.

    const : float or int
        Added constant.

    linear : float or int
        Added inner product term applied to function input.

    scale : float or int
        Function scaling.

    shift : float or int
        Input shifting.

    stretch : float or int
        Input stretching.


    See Also
    --------

    FunctionWithGradProx : Basic function/grad/prox class.
    LinearFunctionWithGradProx : Special case for linear functions.
    NormFunctionWithGradProx : Special case for norms.
    NormSqFunctionWithGradProx : Special case for squared norms.
    IndicatorWithGradProx : Special case for indicator functions.
    NormBallWithGradProx : Special case for norm ball indicator functions.


    Notes
    -----

    The prox operator gives the solution to the problem:

    .. math::

        prox(v, lmbda) = argmin_x ( g(x) + 1/(2*lmbda)*(||x - v||_2)**2 ).

    """
    def __init__(self, scale=None, stretch=None, shift=None, linear=None,
                 const=None):
        """Create function with gradient and prox operator.

        With the class defined for the function f(x), this returns an object
        for evaluating the value, gradient, and prox operator of the function

        .. math::

            g(x) = s*f(a*x + b) + <c, x> + d

        for `scale` :math:`s`, `stretch` :math:`a`, `shift` :math:`b`,
        `linear` :math:`c`, and `const` :math:`d`.


        Parameters
        ----------

        scale : float or int
            Function scaling.

        stretch : float or int
            Input stretching.

        shift : float or int
            Input shifting.

        linear : float or int
            Added inner product term applied to function input.

        const : float or int
            Added constant.

        """
        # change 'None's to identities
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

        self._scale = scale
        self._stretch = stretch
        self._shift = shift
        self._linear = linear
        self._const = const
        # modify functions to apply parameters
        # (order of operations is important!)
        if shift is not None and np.any(shift != 0):
            self.fun = shift_fun(self.fun, shift)
            self.grad = shift_grad(self.grad, shift)
            self.prox = shift_prox(self.prox, shift)
        if stretch is not None and stretch != 1:
            self.fun = stretch_fun(self.fun, stretch)
            self.grad = stretch_grad(self.grad, stretch)
            self.prox = stretch_prox(self.prox, stretch)
        if scale is not None and scale != 1:
            self.fun = scale_fun(self.fun, scale)
            self.grad = scale_grad(self.grad, scale)
            self.prox = scale_prox(self.prox, scale)
        if linear is not None and np.any(linear != 0):
            self.fun = addlinear_fun(self.fun, linear)
            self.grad = addlinear_grad(self.grad, linear)
            self.prox = addlinear_prox(self.prox, linear)
        if const is not None and np.any(const != 0):
            self.fun = addconst_fun(self.fun, const)
            # added constant does not change grad or prox functions

    def __call__(self, x):
        return self.fun(x)

    @property
    def conjugate(self):
        """Object for the conjugate function.


        Notes
        -----

        The convex conjugate of f(x) is defined as

        .. math::
        
            f^*(y) = sup_x ( <y, x> - f(x) ).

        Additionally, if :math:`g(x) = s*f(a*x + b) + <c, x> + d`, then

        .. math::

            g^*(y) = s*f^*(y/(a*s) - c/(a*s)) - <b/a, y> - d.

        """
        Conjugate = self._conjugate_class
        return Conjugate(**self._conjugate_args)

    @property
    def _conjugate_class(self):
        """Return the class for the conjugate function."""
        raise NotImplementedError

    @property
    def _conjugate_args(self):
        """Return the keyword arguments for the conjugate function in a dict."""
        scale = self._scale
        stretch = 1/(self._scale*self._stretch)
        shift = -stretch*self._linear
        linear = -self._shift/self._stretch
        const = -self._const

        return dict(scale=scale, stretch=stretch, shift=shift, linear=linear,
                    const=const)

    def fun(self, x):
        raise NotImplementedError

    def grad(self, x):
        raise NotImplementedError

    def prox(self, x, lmbda=1):
        raise NotImplementedError

    @property
    def const(self):
        return self._const

    @property
    def linear(self):
        return self._linear

    @property
    def scale(self):
        return self._scale

    @property
    def shift(self):
        return self._shift

    @property
    def stretch(self):
        return self._stretch

class LinearFunctionWithGradProx(FunctionWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, scale=None, stretch=None, shift=None, linear=None,
                 const=None):
        # we can eliminate stretching and shifting:
        # s*f(a*x + b) + <c, x> + d ==> a*s*f(x) + <c, x> + (s*f(b) + d)

        # change 'None's to identities
        if scale is None:
            scale = 1
        if stretch is None:
            stretch = 1
        if shift is None:
            shift = 0
        if const is None:
            const = 0
        # absorb stretch into scale
        scale = scale*stretch
        stretch = None
        # absorb shift into const
        const = const + scale*self.fun(shift)
        shift = None
        # turn identities into Nones
        if scale == 1:
            scale = None
        if const == 0:
            const = None

        super(LinearFunctionWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

class NormFunctionWithGradProx(FunctionWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, scale=None, stretch=None, shift=None, linear=None,
                 const=None):
        # we can eliminate stretching:
        # s*f(a*x + b) + <c, x> + d ==> a*s*f(x + b/a) + <c, x> + d

        # absorb stretch into scale and shift
        if stretch is not None:
            if shift is not None:
                shift = shift/stretch
            if scale is not None:
                scale = scale*stretch
            else:
                scale = stretch
            stretch = None

        super(NormFunctionWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

class NormSqFunctionWithGradProx(FunctionWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, scale=None, stretch=None, shift=None, linear=None,
                 const=None):
        # we can eliminate stretching:
        # s*f(a*x + b) + <c, x> + d ==> (a**2)*s*f(x + b/a) + <c, x> + d

        # absorb stretch into scale and shift
        if stretch is not None:
            if shift is not None:
                shift = shift/stretch
            if scale is not None:
                scale = scale*stretch**2
            else:
                scale = stretch**2
            stretch = None

        super(NormSqFunctionWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

class IndicatorWithGradProx(FunctionWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, radius=1, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        # we can eliminate scaling:
        # s*f(a*x + b) + <c, x> + d ==> f(a*x + b) + <c, x> + d

        # eliminate scale
        if scale is not None:
            scale = None

        super(IndicatorWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

class NormBallWithGradProx(IndicatorWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(IndicatorWithGradProx.__init__)
    def __init__(self, radius=1, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """
        radius : float or int
            Radius of the norm ball indicator.

        """
        # we can eliminate stretching:
        # s*f_r(a*x + b) + <c, x> + d ==> s*f_(r/a)(x + b/a) + <c, x> + d

        # absorb stretch into radius and shift
        if stretch is not None:
            radius = radius/stretch
            if shift is not None:
                shift = shift/stretch
            stretch = None

        # set radius parameter
        self._radius = radius

        # we can also eliminate scaling,
        # but this is taken care of by parent class
        super(NormBallWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

    @property
    def radius(self):
        return self._radius
