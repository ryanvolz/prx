# ----------------------------------------------------------------------------
# Copyright (c) 2015-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Standard optimization problems.

.. currentmodule:: prx.standard_probs

l-1 minimization
----------------

.. autosummary::
    :toctree:

    bpdn
    dantzig
    l1rls
    lasso
    srlasso


Other
-----

.. autosummary::
    :toctree:

    nnls
    zcls

"""

from __future__ import division
from functools import wraps
import numpy as np

from .standard_funcs import (
    L1Norm, L1L2Norm, L2Norm, L2NormSqHalf,
    L2BallInd, LInfBallInd, NNegInd, ZerosInd,
)
from .prox_algos import proxgrad, proxgradaccel, admm, admmlin, pdhg

__all__ = ['bpdn', 'dantzig', 'l1rls', 'lasso', 'nnls', 'srlasso', 'zcls']

def backends(*algorithms):
    algnames = [a.__name__ for a in algorithms]
    algos = dict(zip(algnames, algorithms))
    def problem_decorator(f):
        @wraps(f)
        def algorithm(*args, **kwargs):
            # match solver kwarg to available algorithms
            solvername = kwargs.pop('solver', None)
            if solvername is None:
                solvername = algorithms[0].__name__
            try:
                solver = algos[solvername]
            except KeyError:
                s = '{0} is not an available solver for {1}'
                s.format(solvername, f.__name__)
                raise ValueError(s)

            # get algorithm arguments from problem function
            algargs, algkwargs = f(*args, **kwargs)

            return solver(*algargs, **algkwargs)

        algnames[0] = algnames[0] + ' (default)'
        algorithm.__doc__ = algorithm.__doc__.format(', '.join(algnames))
        algorithm.algorithms = algos
        return algorithm
    return problem_decorator


@backends(admmlin, pdhg)
def bpdn(A, Astar, b, eps, x0, **kwargs):
    """Solves the basis pursuit denoising problem.

    argmin_x ||x||_1
      s.t.   ||A(x) - b||_2 <= eps

    If the keyword argument 'axis' is given, the l1-norm is replaced by the
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.

    Additional keyword arguments are passed to the solver.

    Available algorithms: {0}.

    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm()
    else:
        F = L1L2Norm(axis=axis)
    G = L2BallInd(radius=eps)

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(admmlin, pdhg)
def dantzig(A, Astar, b, delta, x0, **kwargs):
    """Solves the Dantzig selector problem..

    argmin_x ||x||_1
      s.t.   ||Astar(A(x) - b)||_inf <= delta

    If the keyword argument 'axis' is given, the l1-norm is replaced by the
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.

    Additional keyword arguments are passed to the solver.

    Available algorithms: {0}.

    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm()
    else:
        F = L1L2Norm(axis=axis)
    G = LInfBallInd(radius=delta)

    # "A" and "Astar" in admmlin notation
    AsA = lambda x: Astar(A(x))
    # "b" in admmlin notation
    Asb = Astar(b)

    args = (F, G, AsA, AsA, Asb, x0)

    return args, kwargs

@backends(proxgradaccel, admmlin, pdhg, proxgrad)
def l1rls(A, Astar, b, lmbda, x0, **kwargs):
    """Solves the l1-regularized least squares problem.

    argmin_x 0.5*(||A(x) - b||_2)**2 + lmbda*||x||_1

    This problem is sometimes called the LASSO since it is equivalent
    to the original LASSO formulation
        argmin_x 0.5*(||A(x) - b||_2)**2
          s.t.   ||x||_1 <= tau
    for appropriate lmbda(tau).

    This function uses the linearized alternating direction method of
    multipliers (LADMM).

    If the keyword argument 'axis' is given, the l1-norm is replaced by the
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.

    Additional keyword arguments are passed to the solver.

    Available algorithms: {0}.

    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm(scale=lmbda)
    else:
        F = L1L2Norm(axis=axis, scale=lmbda)
    G = L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(proxgradaccel, admmlin, pdhg, proxgrad)
def lasso(A, Astar, b, tau, x0, **kwargs):
    """Solves the LASSO problem.

    argmin_x 0.5*(||A(x) - b||_2)**2
      s.t.   ||x||_1 <= tau

    This definition follows Tibshirani's original formulation.
    The l1-regularized least squares problem
        argmin_x 0.5*(||A(x) - b||_2)**2 + lmbda*||x||_1
    is sometimes called the LASSO since they are equivalent for appropriate
    selection of lmbda(tau).

    Additional keyword arguments are passed to the solver.

    Available algorithms: {0}.

    """
    F = L1BallInd(radius=tau)
    G = L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(proxgradaccel, admmlin, pdhg, proxgrad)
def nnls(A, Astar, b, x0, **kwargs):
    """Solves the non-negative least squares problem.

    argmin_x 0.5*(||A(x) - b||_2)**2
      s.t.   x >= 0

    Additional keyword arguments are passed to the solver.

    Available algorithms: {0}.

    """
    F = NNegInd()
    G = L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(admmlin, pdhg)
def srlasso(A, Astar, b, lmbda, x0, **kwargs):
    """Solves the square root LASSO problem.

    argmin_x ||A(x) - b||_2 + lmbda*||x||_1

    The square root LASSO may be preferred to l1-regularized least squares
    since the selection of the parameter 'lmbda' need not depend on the noise
    level of the measurements 'b' (i.e. b = A(x) + n for some noise n). See
    Belloni, Chernozhukov, and Wang in Biometrika (2011) for more details.

    If the keyword argument 'axis' is given, the l1-norm is replaced by the
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.

    Additional keyword arguments are passed to the solver.

    Available algorithms: {0}.

    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm(scale=lmbda)
    else:
        F = L1L2Norm(axis=axis, scale=lmbda)
    G = L2Norm()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(proxgradaccel, admmlin, pdhg, proxgrad)
def zcls(A, Astar, b, zeros, x0, **kwargs):
    """Solves the zero-constrained least squares problem..

    argmin_x 0.5*(||A(x) - b||_2)**2
      s.t.   x[zeros] == 0

    Additional keyword arguments are passed to the solver.

    Available algorithms: {0}.

    """
    F = ZerosInd(z=zeros)
    G = L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs
