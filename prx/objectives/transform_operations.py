# ----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Function factories for simple operations on functions, grads, and proxes.

.. currentmodule:: prx.objectives.transform_operations


Apply function transform to function
------------------------------------

.. autosummary::
    :toctree:

    addconst_fun
    addlinear_fun
    scale_fun
    shift_fun
    stretch_fun


Apply function transform to gradient
------------------------------------

.. autosummary::
    :toctree:

    addlinear_grad
    scale_grad
    shift_grad
    stretch_grad


Apply function transform to prox operator
-----------------------------------------

.. autosummary::
    :toctree:

    addlinear_prox
    scale_prox
    shift_prox
    stretch_prox

"""

from __future__ import division

from functools import wraps

import numpy as np

__all__ = (
    'addconst_fun',
    'addlinear_fun', 'addlinear_grad', 'addlinear_prox',
    'scale_fun', 'scale_grad', 'scale_prox',
    'shift_fun', 'shift_grad', 'shift_prox',
    'stretch_fun', 'stretch_grad', 'stretch_prox',
)


def addconst_fun(f, c):
    """For given function f(x), returns g(x) = f(x) + c."""
    @wraps(f)
    def addedconst(x):
        return f(x) + c

    return addedconst


def addlinear_fun(f, a):
    """For given function f(x), returns g(x) = f(x) + Re(<a, x>)."""
    aconj = np.conj(a)

    @wraps(f)
    def addedlinear(x):
        # TODO replace with inner1d when it becomes available in numpy
        return f(x) + np.multiply(aconj, x).sum().real

    return addedlinear


def addlinear_grad(grad, a):
    """For given gradient of f(x), returns gradient of f(x) + Re(<a, x>)."""
    areal = a.real

    @wraps(grad)
    def addedlinear(x):
        return grad(x) + areal

    return addedlinear


def addlinear_prox(prox, a):
    """For given prox operator of f(x), returns prox for f(x) + Re(<a, x>)."""
    @wraps(prox)
    def addedlinear(x, lmbda=1):
        return prox(x - lmbda*a, lmbda)

    return addedlinear


def scale_fun(f, scale):
    """For given function f(x), returns g(x) = scale*f(x)."""
    @wraps(f)
    def scaled(x):
        return scale*f(x)

    return scaled


def scale_grad(grad, scale):
    """For given gradient of f(x), returns gradient of scale*f(x)."""
    @wraps(grad)
    def scaled(x):
        return scale*grad(x)

    return scaled


def scale_prox(prox, scale):
    """For given prox operator of f(x), returns prox for scale*f(x)."""
    @wraps(prox)
    def scaled(x, lmbda=1):
        return prox(x, lmbda*scale)

    return scaled


def shift_fun(f, shift):
    """For given function f(x), returns g(x) = f(x + shift)."""
    @wraps(f)
    def shifted(x):
        return f(x + shift)

    return shifted


def shift_grad(grad, shift):
    """For given gradient of f(x), returns gradient of f(x + shift)."""
    @wraps(grad)
    def shifted(x):
        return grad(x + shift)

    return shifted


def shift_prox(prox, shift):
    """For given prox operator of f(x), returns prox for f(x + shift)."""
    @wraps(prox)
    def shifted(x, lmbda=1):
        return prox(x + shift, lmbda) - shift

    return shifted


def stretch_fun(f, stretch):
    """For given function f(x), returns g(x) = f(stretch*x)."""
    @wraps(f)
    def stretched(x):
        return f(stretch*x)

    return stretched


def stretch_grad(grad, stretch):
    """For given gradient of f(x), returns gradient of f(stretch*x)."""
    @wraps(grad)
    def stretched(x):
        return stretch*grad(stretch*x)

    return stretched


def stretch_prox(prox, stretch):
    """For given prox operator of f(x), returns prox for f(stretch*x)."""
    @wraps(prox)
    def stretched(x, lmbda=1):
        return prox(stretch*x, lmbda*stretch**2)/stretch

    return stretched
