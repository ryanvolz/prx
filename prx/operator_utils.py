# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Operator utilities."""

import numpy as np

from .fun.norms import l2norm

__all__ = ('adjointness_error', 'opnorm')


def get_random_normal(shape, dtype):
    x = np.empty(shape, dtype)
    x.real = np.random.randn(*shape)
    try:
        x.imag = np.random.randn(*shape)
    except TypeError:
        pass
    return x


def adjointness_error(A, Astar, inshape, indtype, its=100):
    """Check adjointness of `A` and `Astar` for `its` instances of random data.

    For random unit-normed x and y, this finds the error in the adjoint
    identity :math:`<Ax, y> == <x, A*y>`::

        err = abs( vdot(A(x), y) - vdot(x, Astar(y)) )


    Parameters
    ----------

    A : function
        Forward operator.

    Astar : function
        Adjoint operator.

    inshape : tuple
        Shape of the input to A.

    indtype : tuple
        Array dtype of the input to A.

    its : int
        Number of random instances for which to check the adjointness.


    Returns
    -------

    out : ndarray of length `its`
        A vector of the adjointness error magnitudes for each random instance.

    """
    x = get_random_normal(inshape, indtype)
    y = A(x)
    outshape = y.shape
    outdtype = y.dtype

    errs = np.zeros(its)
    for k in range(its):
        x = get_random_normal(inshape, indtype)
        x = x/l2norm(x)
        y = get_random_normal(outshape, outdtype)
        y = y/l2norm(y)
        ip_A = np.vdot(A(x), y)
        ip_Astar = np.vdot(x, Astar(y))
        errs[k] = np.abs(ip_A - ip_Astar)

    return errs


def opnorm(A, Astar, inshape, indtype, reltol=1e-8, abstol=1e-6, maxits=100,
           printrate=None):
    """Estimate the l2-induced operator norm: sup_v ||A(v)||/||v|| for v != 0.

    Uses the power iteration method to estimate the operator norm of
    `A` and `Astar`.

    Parameters
    ----------

    A : function
        Forward operator.

    Astar : function
        Adjoint operator.

    inshape : tuple
        Shape of the input to `A`.

    indtype : tuple
        Array dtype of the input to `A`.

    reltol : float
        Relative tolerance for judging convergence of the operator norms.

    abstol : float
        Absolute tolerance for judging convergence of the operator norms.

    maxits : int
        Maximum number of iterations to try to reach convergence to the
        operator norms.

    printrate : int | None
        Printing interval for displaying the current iteration count and
        operator norms. If None, do not print.


    Returns
    -------

    out : tuple
        Tuple containing: (norm of `A`, norm of `Astar`, vector inducing
        maximum scaling).

    """
    v0 = get_random_normal(inshape, indtype)
    v = v0/l2norm(v0)
    norm_f0 = 1
    norm_a0 = 1
    for k in range(maxits):
        Av = A(v)
        norm_f = l2norm(Av)
        w = Av/norm_f
        Asw = Astar(w)
        norm_a = l2norm(Asw)
        v = Asw/norm_a

        delta_f = abs(norm_f - norm_f0)
        delta_a = abs(norm_a - norm_a0)

        if printrate is not None and (k % printrate) == 0:
            s = 'Iteration {0}, forward norm: {1}, adjoint norm: {2}'
            s = s.format(k, norm_f, norm_a)
            print(s)
        if (delta_f < abstol + reltol*max(norm_f, norm_f0)
                and delta_a < abstol + reltol*max(norm_a, norm_a0)):
            break

        norm_f0 = norm_f
        norm_a0 = norm_a

    return norm_f, norm_a, v
