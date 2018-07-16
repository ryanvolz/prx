# ----------------------------------------------------------------------------
# Copyright (c) 2015-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Base classes for function-grad-prox Function objects."""

from __future__ import division

import numpy as np

from six import with_metaclass

from ..algorithms import _admm
from ..docstring_helpers import DocstringSubstituteMeta
from ..separable_array import separray
from .transform_operations import (addconst_fun, addlinear_fun, addlinear_grad,
                                   addlinear_prox, scale_fun, scale_grad,
                                   scale_prox, shift_fun, shift_grad,
                                   shift_prox, stretch_fun, stretch_grad,
                                   stretch_prox)

__all__ = (
    'BaseFunction', 'LinearFunction',
    'NormFunction', 'NormSqFunction',
    'IndicatorFunction', 'NormBallFunction',
    'Function', 'SeparableFunction',
)


class BaseFunction(with_metaclass(DocstringSubstituteMeta, object)):
    """Function class, a function with gradient and/or prox operator.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    LinearFunction : Special case for linear functions.
    NormFunction : Special case for norms.
    NormSqFunction : Special case for squared norms.
    IndicatorFunction : Special case for indicator functions.
    NormBallFunction : Special case for norm ball indicator functions.


    Notes
    -----

    {function_notes}

    """

    _doc_function_summary = """
    This class holds a function and its associated gradient and
    prox operators, with optional scaling, stretching, shifting, and linear or
    constant terms added for initialized objects.

    The function is evaluated by calling ``obj(x)`` or ``obj.fun(x)``.
    Its gradient is evaluated by calling ``obj.grad(x)``.
    Its prox operator is evaluated by calling ``obj.prox(x, lmbda)``.

    The :meth:`fun`, :meth:`grad`, and :meth:`prox` methods must be
    defined for a particular operator by inheriting from this class. It is
    also necessary to override the attribute :attr:`_conjugate_class` to
    give the class of the conjugate function, if applicable.

    """

    _doc_function_attributes = """
    conjugate : :class:`.BaseFunction` object
        The corresponding conjugate Function.

    const : float | int
        Added constant.

    linear : float | int | array
        Added inner product term applied to function input.

    scale : float | int
        Function scaling.

    shift : float | int | array
        Input shifting.

    stretch : float | int | array
        Input stretching.

    """

    _doc_function_notes = """
    The prox operator of a function ``g(x)`` gives the solution ``x`` to the
    problem::

        minimize    g(x) + l2normsqhalf(x - v)/lmbda

    for a given `v` and `lmbda`.

    """

    _doc_init_summary = """
    With the class defined for the function f(x), this creates an object for
    evaluating the value, gradient, and prox operator of the function::

        g(x) = s*f(a*x + b) + Re(<c, x>) + d

    for ``s=scale``, ``a=stretch``, ``b=shift``, ``c=linear``, and ``d=const``.

    """

    _doc_init_params = """
    scale : float | int, optional
        Function scaling.

    stretch : float | int | (array, for separable functions), optional
        Input stretching.

    shift : float | int | array, optional
        Input shifting.

    linear : float | int | array, optional
        Added inner product term applied to function input.

    const : float | int, optional
        Added constant.

    """

    def __init__(
        self, scale=None, stretch=None, shift=None, linear=None, const=None,
    ):
        """Create Function with gradient and/or prox operator.

        {init_summary}


        Parameters
        ----------

        {init_params}

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
        """Evaluate function."""
        return self.fun(x)

    def __add__(self, other):
        """Return the Function object for the sum of two functions.

        The summed Function's value, gradient, and prox are defined when both
        of the component Functions have a value, gradient, or prox,
        respectively.

        Since the prox operator requires the solution of an optimization
        problem in general, the summed prox operator is evaluated using
        :class:`.ADMM`. Accordingly, the order of the addition may effect
        the speed and accuracy of the prox operator solution.

        """
        if not isinstance(other, BaseFunction):
            return NotImplemented

        summed = BaseFunction()

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
            F = Function(fun=other.fun, prox=proxF)
            return _admm(F, self, x0=x, pen=lmbda, maxits=100, printrate=None)

        summed.fun = summed_fun
        summed.grad = summed_grad
        summed.prox = summed_prox

        return summed

    __radd__ = __add__

    @property
    def conjugate(self):
        """Function object for the conjugate function.

        Notes
        -----

        The convex conjugate of :math:`f(x)` is defined as

        .. math::

            f^*(y) = sup_x ( <y, x> - f(x) ).

        Additionally, if :math:`g(x) = s*f(a*x + b) + Re(<c, x>) + d`, then

        .. math::

            g^*(y) = s*f^*(y/(a*s) - c/(a*s)) - Re(<b/a, y>) - d.

        """
        Conjugate = self._conjugate_class
        return Conjugate(**self._conjugate_args)

    @property
    def _conjugate_class(self):
        """Return the class for the conjugate Function."""
        raise NotImplementedError

    @property
    def _conjugate_args(self):
        """Return the keyword args for the conjugate Function in a dict."""
        scale = self._scale
        stretch = 1/(self._scale*self._stretch)
        shift = -stretch*self._linear
        linear = -self._shift/self._stretch
        const = -self._const

        return dict(scale=scale, stretch=stretch, shift=shift, linear=linear,
                    const=const)

    def fun(self, x):
        """Evaluate function."""
        raise NotImplementedError

    def grad(self, x):
        """Evaluate gradient."""
        raise NotImplementedError

    def prox(self, x, lmbda=1):
        """Evaluate proximal operator."""
        raise NotImplementedError

    @property
    def const(self):
        """Additive constant."""
        return self._const

    @property
    def linear(self):
        """Additive linear component."""
        return self._linear

    @property
    def scale(self):
        """Scaling constant."""
        return self._scale

    @property
    def shift(self):
        """Shift constant."""
        return self._shift

    @property
    def stretch(self):
        """Stretch constant."""
        return self._stretch


class LinearFunction(BaseFunction):
    """Function class for a linear function.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    BaseFunction : Base Function class.
    NormFunction : Special case for norms.
    NormSqFunction : Special case for squared norms.
    IndicatorFunction : Special case for indicator functions.
    NormBallFunction : Special case for norm ball indicator functions.


    Notes
    -----

    {function_notes}

    """

    _doc_init_summary = """
    {init_summary}

    Since this function is linear, `scale` and `shift` can be
    eliminated by absorbing them into `stretch` and `const`::

        s*f(a*x + b) + d => f((s*a)*x) + (s*f(b) + d)

    """

    def __init__(
        self, scale=None, stretch=None, shift=None, linear=None, const=None,
    ):
        """Create Function that is known to be linear.

        {init_summary}


        Parameters
        ----------

        {init_params}

        """
        # change 'None's to identities
        if scale is None:
            scale = 1
        if stretch is None:
            stretch = 1
        if shift is None:
            shift = 0
        if const is None:
            const = 0
        # absorb scale and shift into const
        const = const + scale*self.fun(shift)
        shift = None
        # absorb scale into stretch
        stretch = scale*stretch
        scale = None

        # turn identities into Nones
        if stretch == 1:
            stretch = None
        if const == 0:
            const = None

        super(LinearFunction, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)


class NormFunction(BaseFunction):
    """Function class for a norm function.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    BaseFunction : Base Function class.
    LinearFunction : Special case for linear functions.
    NormSqFunction : Special case for squared norms.
    IndicatorFunction : Special case for indicator functions.
    NormBallFunction : Special case for norm ball indicator functions.


    Notes
    -----

    {function_notes}

    """

    _doc_init_summary = """
    {init_summary}

    Since this function is a norm, `scale` can be eliminated by
    absorbing it into `stretch` and `shift`::

        s*f(a*x + b) => f((s*a)*x + (s*b))

    """

    def __init__(
        self, scale=None, stretch=None, shift=None, linear=None, const=None,
    ):
        """Create Function that defines a norm function.

        {init_summary}


        Parameters
        ----------

        {init_params}

        """
        # absorb scale into stretch and shift
        if scale is not None:
            if shift is not None:
                shift = scale*shift
            if stretch is not None:
                stretch = scale*stretch
            else:
                stretch = scale
            scale = None

        super(NormFunction, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)


class NormSqFunction(BaseFunction):
    """Function class for a squared norm function.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    BaseFunction : Base Function class.
    LinearFunction : Special case for linear functions.
    NormFunction : Special case for norms.
    IndicatorFunction : Special case for indicator functions.
    NormBallFunction : Special case for norm ball indicator functions.


    Notes
    -----

    {function_notes}

    """

    _doc_init_summary = """
    {init_summary}

    Since this function is a squared norm, `scale` can be eliminated by
    absorbing it into `stretch` and `shift`::

        s*f(a*x + b) => f((sqrt(s)*a)*x + (sqrt(s)*b))

    """

    def __init__(
        self, scale=None, stretch=None, shift=None, linear=None, const=None,
    ):
        """Create Function that defines a squared norm function.

        {init_summary}


        Parameters
        ----------

        {init_params}

        """
        # absorb scale into stretch and shift
        if scale is not None:
            if shift is not None:
                shift = np.sqrt(scale)*shift
            if stretch is not None:
                stretch = np.sqrt(scale)*stretch
            else:
                stretch = np.sqrt(scale)
            scale = None

        super(NormSqFunction, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)


class IndicatorFunction(BaseFunction):
    """Function class for an indicator function.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    BaseFunction : Base Function class.
    LinearFunction : Special case for linear functions.
    NormFunction : Special case for norms.
    NormSqFunction : Special case for squared norms.
    NormBallFunction : Special case for norm ball indicator functions.


    Notes
    -----

    {function_notes}

    """

    _doc_init_summary = """
    {init_summary}

    Since this function is an indicator, `scale` can be eliminated::

        s*f(a*x + b) => f(a*x + b)

    """

    def __init__(
        self, scale=None, stretch=None, shift=None, linear=None, const=None,
    ):
        """Create Function that defines an indicator function.

        {init_summary}


        Parameters
        ----------

        {init_params}

        """
        # eliminate scale
        if scale is not None:
            scale = None

        super(IndicatorFunction, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)


class NormBallFunction(IndicatorFunction):
    """Function class for a norm ball indicator function.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    BaseFunction : Base Function class.
    LinearFunction : Special case for linear functions.
    NormFunction : Special case for norms.
    NormSqFunction : Special case for squared norms.
    IndicatorFunction : Special case for indicator functions.


    Notes
    -----

    {function_notes}

    """

    _doc_function_attributes = """
    radius : float | int
        Radius of the norm ball indicator.

    {function_attributes}

    """

    _doc_init_params = """
    radius : float | int, optional
        Radius of the norm ball indicator.

    {init_params}

    """

    def __init__(
        self, radius=1, scale=None, stretch=None, shift=None, linear=None,
        const=None,
    ):
        """Create Function that defines a norm ball indicator function.

        {init_summary}


        Parameters
        ----------

        {init_params}

        """
        # set radius parameter
        self._radius = radius

        # we can eliminate scaling,
        # but this is taken care of by parent class
        super(NormBallFunction, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

    @property
    def radius(self):
        """Norm ball radius."""
        return self._radius


# ****************************************************************************
# * User-facing classes/factories for function-prox objects ******************
# ****************************************************************************

def Function(
    fun=None, grad=None, prox=None, cls=BaseFunction, Conjugate=None, **kwargs
):
    """Create custom Function object, a function with grad and/or prox.

    This function provides a convenient way to create and Function object
    directly from function, gradient, and prox functions. Keyword arguments
    are passed to the __init__ method of the defined Function class to allow
    for initialization with transformation parameters.


    Parameters
    ----------

    fun : function, optional
        ``f(x)``, which returns the value at a given point x.

    grad : function, optional
        ``grad_f(x)``, which returns the gradient at a given point x.

    prox : function, optional
        ``prox_f(x, lmbda=1)``, which evaluates the prox operator at x with
        scaling lmbda.

    cls : class, optional
        Function class or a subclass that the custom function belongs to.

    Conjugate : class, optional
        Class for the corresponding conjugate Function, if any.


    Returns
    -------

    f : `cls` object
        Object of type `cls` that contains the custom function.


    Other Parameters
    ----------------

    kwargs : dict
        Keyword arguments (e.g. transformations) passed to the __init__ method
        of `cls`.

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

    return Custom(**kwargs)


class SeparableFunction(BaseFunction):
    """Function class for a separable function.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    :class:`.BaseFunction` : Base Function class.


    Notes
    -----

    {function_notes}

    """

    _doc_function_summary = """
    A separable function is one that can be divided into summed component
    functions that each operate on mutually exclusive partitions of the
    input. For a separable function F(X), we assume that the input X is a
    separated array that when iterated over gives the individual input
    components, ``X = (x1, x2, ..., xn)``. Then F is described by its
    components ``(f1, f2, ..., fn)`` so that
    ``F(X) = f1(x1) + f2(x2) + ... + fn(xn)``.

    {function_summary}

    """

    _doc_function_attributes = """
    components : tuple
        The component Function objects.

    {function_attributes}

    """

    _doc_function_notes = """
    The gradient and prox operators of a separable function are simply the
    combination of the gradients and proxes of the component functions:
    ``grad_F(X) = ( grad_f1(x1), grad_f2(x2), ..., grad_fn(xn) )`` and
    similarly for the prox operator.

    {function_notes}

    """

    _doc_init_summary = """
    The Function objects the comprise the separable function are passed
    as positional arguments. Transformation parameters for the entire
    separable function, applied in addition to and after any possible
    transformations included in the component functions, can be supplied as
    keyword arguments.

    {init_summary}

    """

    _doc_init_params = """
    components : tuple of :class:`.BaseFunction` objects
        Non-keyword arguments are the component Functions that comprise
        the separable function.

    {init_params}

    """

    def __init__(self, *components, **kwargs):
        """Create separable Function as combination of component Functions.

        {init_summary}


        Parameters
        ----------

        {init_params}


        """
        self._components = components
        super(SeparableFunction, self).__init__(**kwargs)

    def _component_arg_gen(self, X):
        for comp, x in zip(self._components, X):
            yield comp, x

    @property
    def components(self):
        """Component Function objects of the separable function."""
        return self._components

    @property
    def conjugate(self):
        """Object for the conjugate Function.

        The conjugate of a separable function is the separable function of
        the conjugate of the parts.


        See Also
        --------

        BaseFunction.conjugate

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
