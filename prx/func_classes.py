# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import division
import numpy as np

from .func_ops import (
    shift_fun, shift_grad, shift_prox,
    stretch_fun, stretch_grad, stretch_prox,
    scale_fun, scale_grad, scale_prox,
    addlinear_fun, addlinear_grad, addlinear_prox,
    addconst_fun
)
from .separable_array import separray
from .prox_algos import admm

__all__ = [
    'FunctionWithGradProx', 'LinearFunctionWithGradProx',
    'NormFunctionWithGradProx', 'NormSqFunctionWithGradProx',
    'IndicatorWithGradProx', 'NormBallWithGradProx',
    'CustomFunction', 'SeparableFunction',
]

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

    .. automethod:: __init__


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

            g(x) = s*f(a*x + b) + Re(<c, x>) + d

        for `scale` :math:`s`, `stretch` :math:`a`, `shift` :math:`b`,
        `linear` :math:`c`, and `const` :math:`d`.


        Parameters
        ----------

        scale : float | int, optional
            Function scaling.

        stretch : float | int | array, optional
            Input stretching.

        shift : float | int | array, optional
            Input shifting.

        linear : float | int | array, optional
            Added inner product term applied to function input.

        const : float | int, optional
            Added constant.

        """
        # modify functions to apply parameters
        # (order of operations is important!)
        if shift is not None and np.any(shift != 0):
            self.fun = shift_fun(self.fun, shift)
            self.grad = shift_grad(self.grad, shift)
            self.prox = shift_prox(self.prox, shift)
            self._shift = shift
        else:
            self._shift = 0
        if stretch is not None and np.any(stretch != 1):
            self.fun = stretch_fun(self.fun, stretch)
            self.grad = stretch_grad(self.grad, stretch)
            self.prox = stretch_prox(self.prox, stretch)
            self._stretch = stretch
        else:
            self._stretch = 1
        if scale is not None and scale != 1:
            self.fun = scale_fun(self.fun, scale)
            self.grad = scale_grad(self.grad, scale)
            self.prox = scale_prox(self.prox, scale)
            self._scale = scale
        else:
            self._scale = 1
        if linear is not None and np.any(linear != 0):
            self.fun = addlinear_fun(self.fun, linear)
            self.grad = addlinear_grad(self.grad, linear)
            self.prox = addlinear_prox(self.prox, linear)
            self._linear = linear
        else:
            self._linear = 0
        if const is not None and const != 0:
            self.fun = addconst_fun(self.fun, const)
            # added constant does not change grad or prox functions
            self._const = const
        else:
            self._const = 0

    def __call__(self, x):
        return self.fun(x)

    def __add__(self, other):
        """Return the function object for the sum of two functions.

        The summed function's value, gradient, and prox are defined when both
        of the component functions have a value, gradient, or prox,
        respectively.

        Since the prox operator requires the solution of an optimization
        problem in general, the summed prox operator is evaluated using
        :func:`.admm`. Accordingly, the order of the addition may effect
        the speed and accuracy of the prox operator solution.

        """
        if not isinstance(other, FunctionWithGradProx):
            return NotImplemented

        summed = FunctionWithGradProx()

        def summed_fun(x):
            return self.fun(x) + other.fun(x)

        def summed_grad(x):
            return self.grad(x) + other.grad(x)

        # summed prox(v, lmbda) solves:
        # argmin_x ( self(x) + other(x) + ( ||x - v||_2 )^2 / (2*lmbda) )
        # so use ADMM to get general solution with
        #   F = other + ( ||x - v||_2 )^2 / (2*lmbda) )
        #   G = self
        # and then the prox of G is defined in terms of other.prox as below
        def summed_prox(x, lmbda=1):
            def proxF(u, s):
                return other.prox((s*x + lmbda*u)/(s + lmbda),
                                  lmbda*s/(s + lmbda))
            F = CustomFunction(fun=other.fun, prox=proxF)()
            return admm(F, self, x0=x, pen=lmbda, maxits=100, printrate=None)

        summed.fun = summed_fun
        summed.grad = summed_grad
        summed.prox = summed_prox

        return summed

    __radd__ = __add__

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


###***************************************************************************
### User-facing classes/factories for function-prox objects ******************
###***************************************************************************

def CustomFunction(cls=FunctionWithGradProx, fun=None, grad=None, prox=None,
                   Conjugate=None):
    """Custom function with gradient and prox operator.

    Create a class for representing a function, gradient, and prox defined
    by the supplied functions.


    Parameters
    ----------

    cls : class
        FunctionWithGradProx class or a subclass that the custom function
        belongs to.

    fun : function
        ``f(x)``, which returns the value at a given point `x`.

    grad : function
        ``grad_f(x)``, which returns the gradient at a given point `x`.

    prox : function
        ``prox_f(x, lmbda=1)``, which evaluates the prox operator at `x` with
        scaling `lmbda`.

    Conjugate : class
        Class for the corresponding conjugate function, if any.


    Returns
    -------

    Custom : class
        Class inheriting from cls that contains the custom function/grad/prox.

    """
    f = fun
    g = grad
    p = prox
    del fun, grad, prox
    class Custom(cls):
        __doc__ = cls.__doc__
        if Conjugate is not None:
            @property
            def _conjugate_class(self):
                return Conjugate

        if f is not None:
            fun = staticmethod(f)
        if g is not None:
            grad = staticmethod(g)
        if p is not None:
            prox = staticmethod(p)

    return Custom

class SeparableFunction(FunctionWithGradProx):
    """Separable function with gradient and prox operator.

    A separable function is one that can be divided into summed component
    functions that each operate on mutually exclusive partitions of the input.
    For a separable function F(X), we assume that the input X is a separated
    array that when iterated over gives the individual input components,
    ``X = (x1, x2, ..., xn)``. Then F is desribed by its components
    ``(f1, f2, ..., fn)`` so that ``F(X) = f1(x1) + f2(x2) + ... + fn(xn)``.

    .. automethod:: __init__


    Attributes
    ----------

    components : tuple
        The component function/grad/prox objects.

    conjugate : :class:`SeparableFunction` object
        The corresponding separable conjugate function/grad/prox.


    See Also
    --------

    FunctionWithGradProx : Base function/grad/prox class.


    Notes
    -----

    The gradient and prox operators of a separable function are simply the
    combination of the gradients and proxes of the component functions:
    ``grad_F(X) = ( grad_f1(x1), grad_f2(x2), ..., grad_fn(xn) )`` and
    similarly for the prox operator.

    """
    def __init__(self, *components, **kwargs):
        """Create separable function as combination of component functions.


        Parameters
        ----------

        *components : FunctionWithGradProx objects
            Non-keyword arguments are the component functions that comprise
            the separable function.


        Other Parameters
        ----------------

        scale, stretch, shift, linear, const : float or int
            Transformation parameters for the entire separable function,
            applied in addition to and after any possible transformations
            included in the component functions.

        """
        self._components = components
        super(SeparableFunction, self).__init__(**kwargs)

    def _component_arg_gen(self, X):
        for comp, x in zip(self._components, X):
            yield comp, x

    @property
    def components(self):
        """Component function/grad/prox objects of the separable function."""
        return self._components

    @property
    def conjugate(self):
        """Object for the conjugate function.

        The conjugate of a separable function is the separable function of the
        conjugate of the parts.


        See Also
        --------

        FunctionWithGradProx.conjugate

        """
        conj_comps = tuple(comp.conjugate for comp in self._components)
        return SeparableFunction(*conj_comps)

    def fun(self, X):
        """Evaluate the separable function at X.


        Parameters
        ----------

        X : iterable of ndarrays with ``len(X) == len(self.components)``
            Partition of the input into separate values, typically provided by
            a :func:`.separray`. Each component function operates on only its
            portion of the input.


        Returns
        -------

        F(X) : value
            Function value at X, which is the sum of the values of the
            individual components: ``F(X) = f1(x1) + f2(x2) + ... + fn(xn)``.

        """
        compx = self._component_arg_gen(X)

        comp, x = compx.next()
        y = comp.fun(x)

        for comp, x in compx:
            y += comp.fun(x)

        return y

    def grad(self, X):
        """Evaluate gradient of the separable function at X.


        Parameters
        ----------

        X : iterable of ndarrays with ``len(X) == len(self.components)``
            Partition of the input into separate values, typically provided by
            a :func:`.separray`. Each component function operates on only its
            portion of the input.


        Returns
        -------

        grad_F(X) : :func:`.separray` with parts like the items of X
            Gradient at X, which is the combination of the gradients of the
            individual components:
            ``grad_F(X) = ( grad_f1(x1), grad_f2(x2), ..., grad_fn(xn) )``.

        """
        compx = self._component_arg_gen(X)

        return separray(*(comp.grad(x) for comp, x in compx))

    def prox(self, X, lmbda=1):
        """Evaluate prox operator of the separable function at X.


        Parameters
        ----------

        X : iterable of ndarrays with ``len(X) == len(self.components)``
            Partition of the input into separate values, typically provided by
            a :func:`.separray`. Each component function operates on only its
            portion of the input.


        Returns
        -------

        prox_F(X) : :func:`.separray` with parts like the items of X
            Prox operator at X, which is the combination of the prox operators
            of the individual components:
            ``prox_F(X) = ( prox_f1(x1), prox_f2(x2), ..., prox_fn(xn) )``.

        """
        compx = self._component_arg_gen(X)

        return separray(*(comp.prox(x, lmbda) for comp, x in compx))

    __call__ = fun
