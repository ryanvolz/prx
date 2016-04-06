# ----------------------------------------------------------------------------
# Copyright (c) 2015-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
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

        solvers = '``\'' + '\'`` | ``\''.join(algnames) + '\'``'
        seealso = '.' + ', .'.join(algnames)
        algorithm.__doc__ = algorithm.__doc__.format(
            solvers=solvers, seealso=seealso,
        )
        algorithm.algorithms = algos
        return algorithm
    return problem_decorator


@backends(admmlin, pdhg)
def bpdn(A, Astar, b, eps, x0, **kwargs):
    """Solves the basis pursuit denoising problem.

    Given A, b, and eps, solve for x::

        minimize    l1norm(x)
        subject to  l2norm(A(x) - b) <= eps


    Parameters
    ----------

    A : callable
        Linear operator that would accept `x0` as input.

    Astar : callable
        Linear operator that is the adjoint of `A`, would accept `b` as input.

    b : array
        Constant vector used in l2-norm expression.

    eps : float
        A positive number describing the l2-norm constraint.

    x0 : array
        Initial iterate for x.


    Returns
    -------

    x : array
        Solution to the optimization problem.


    Other Parameters
    ----------------

    axis : None | int | tuple of ints, optional
        If this argument is given, the l1-norm is replaced by the combined l1-
        and l2-norm with the l2-norm taken over the specified axes:
        ``l1norm(l2norm(x, axis))``. This can be used to solve the "block" or
        "group" sparsity problem.

    solver : {solvers}, optional
        Algorithm to use.

    **kwargs
        Additional keyword arguments passed to the solver.


    See Also
    --------

    dantzig, l1rls, lasso, srlasso, {seealso}

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
    """Solves the Dantzig selector problem.

    Given A, b, and delta, solve for x::

        minimize    l1norm(x)
        subject to  linfnorm(Astar(A(x) - b)) <= delta


    Parameters
    ----------

    A : callable
        Linear operator that would accept `x0` as input.

    Astar : callable
        Linear operator that is the adjoint of `A`, would accept `b` as input.

    b : array
        Constant vector used in linf-norm expression.

    delta : float
        A positive number describing the linf-norm constraint.

    x0 : array
        Initial iterate for x.


    Returns
    -------

    x : array
        Solution to the optimization problem.


    Other Parameters
    ----------------

    axis : None | int | tuple of ints, optional
        If this argument is given, the l1-norm is replaced by the combined l1-
        and l2-norm with the l2-norm taken over the specified axes:
        ``l1norm(l2norm(x, axis))``. This can be used to solve the "block" or
        "group" sparsity problem.

    solver : {solvers}, optional
        Algorithm to use.

    **kwargs
        Additional keyword arguments passed to the solver.


    See Also
    --------

    bpdn, l1rls, lasso, srlasso, {seealso}


    References
    ----------

    .. [1] E. Candes and T. Tao, "The Dantzig selector: Statistical estimation
       when p is much larger than n," Ann. Statist., vol. 35, no. 6, pp.
       2313-2351, Dec. 2007.


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

    Given A, b, and lmbda, solve for x::

        minimize  0.5*l2norm(A(x) - b)**2 + lmbda*l1norm(x)

    This problem is sometimes called the LASSO since it is equivalent to the
    original :func:`lasso` formulation for appropriate lmbda as a function of
    tau.


    Parameters
    ----------

    A : callable
        Linear operator that would accept `x0` as input.

    Astar : callable
        Linear operator that is the adjoint of `A`, would accept `b` as input.

    b : array
        Constant vector used in l2-norm expression.

    lmbda : float
        A positive number giving the l1-norm weighting.

    x0 : array
        Initial iterate for x.


    Returns
    -------

    x : array
        Solution to the optimization problem.


    Other Parameters
    ----------------

    axis : None | int | tuple of ints, optional
        If this argument is given, the l1-norm is replaced by the combined l1-
        and l2-norm with the l2-norm taken over the specified axes:
        ``l1norm(l2norm(x, axis))``. This can be used to solve the "block" or
        "group" sparsity problem.

    solver : {solvers}, optional
        Algorithm to use.

    **kwargs
        Additional keyword arguments passed to the solver.


    See Also
    --------

    bpdn, dantzig, lasso, srlasso, {seealso}


    References
    ----------

    .. [1] A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems," SIAM Journal on Imaging
       Sciences, vol. 2, no. 1, pp. 183-202, Jan. 2009.

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

    Given A, b, and tau, solve for x::

        minimize    0.5*l2norm(A(x) - b)**2
        subject to  l1norm(x) <= tau

    This definition follows Tibshirani's original formulation from [1]_.
    The :func:`l1rls` problem is sometimes called the LASSO since they are
    equivalent for appropriate selection of `lmbda` as a function of `tau`.


    Parameters
    ----------

    A : callable
        Linear operator that would accept `x0` as input.

    Astar : callable
        Linear operator that is the adjoint of `A`, would accept `b` as input.

    b : array
        Constant vector used in l2-norm expression.

    tau : float
        A positive number describing the l1-norm constraint.

    x0 : array
        Initial iterate for x.


    Returns
    -------

    x : array
        Solution to the optimization problem.


    Other Parameters
    ----------------

    solver : {{{solvers}}}, optional
        Algorithm to use.

    **kwargs
        Additional keyword arguments passed to the solver.


    See Also
    --------

    bpdn, dantzig, l1rls, srlasso, {seealso}


    References
    ----------

    .. [1] R. Tibshirani, "Regression Shrinkage and Selection via the Lasso,"
       Journal of the Royal Statistical Society. Series B (Methodological),
       vol. 58, no. 1, pp. 267-288, Jan. 1996.


    """
    F = L1BallInd(radius=tau)
    G = L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(proxgradaccel, admmlin, pdhg, proxgrad)
def nnls(A, Astar, b, x0, **kwargs):
    """Solves the non-negative least squares problem.

    Given A and b, solve for x::

        minimize    l2norm(A(x) - b)
        subject to  x >= 0


    Parameters
    ----------

    A : callable
        Linear operator that would accept `x0` as input.

    Astar : callable
        Linear operator that is the adjoint of `A`, would accept `b` as input.

    b : array
        Constant vector used in l2-norm expression.

    x0 : array
        Initial iterate for x.


    Returns
    -------

    x : array
        Solution to the optimization problem.


    Other Parameters
    ----------------

    solver : {solvers}, optional
        Algorithm to use.

    **kwargs
        Additional keyword arguments passed to the solver.


    See Also
    --------

    zcls, {seealso}

    """
    F = NNegInd()
    G = L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(admmlin, pdhg)
def srlasso(A, Astar, b, lmbda, x0, **kwargs):
    """Solves the square root LASSO problem.

    Given A, b, and lmbda, solve for x::

        minimize  l2norm(A(x) - b) + lmbda*l1norm(x)

    The square root LASSO may be preferred to :func:`l1rls` since the
    selection of the parameter `lmbda` need not depend on the noise level of
    the measurements `b` (under the model where ``b = A(x) + n`` for some
    noise n). See [1]_ for more details.


    Parameters
    ----------

    A : callable
        Linear operator that would accept `x0` as input.

    Astar : callable
        Linear operator that is the adjoint of `A`, would accept `b` as input.

    b : array
        Constant vector used in l2-norm expression.

    lmbda : float
        A positive number giving the l1-norm weighting.

    x0 : array
        Initial iterate for x.


    Returns
    -------

    array or dict
        Solution to the optimization problem (see solver function for details).


    Other Parameters
    ----------------

    axis : None | int | tuple of ints, optional
        If this argument is given, the l1-norm is replaced by the combined l1-
        and l2-norm with the l2-norm taken over the specified axes:
        ``l1norm(l2norm(x, axis))``. This can be used to solve the "block" or
        "group" sparsity problem.

    solver : {solvers}, optional
        Algorithm to use.

    **kwargs
        Additional keyword arguments passed to the solver.


    See Also
    --------

    bpdn, dantzig, l1rls, lasso, {seealso}


    References
    ----------

    .. [1] A. Belloni, V. Chernozhukov, and L. Wang, "Square-root lasso:
       pivotal recovery of sparse signals via conic programming," Biometrika,
       vol. 98, no. 4, pp. 791-806, Dec. 2011.


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
    """Solves the zero-constrained least squares problem.

    Given A, b, and zeros, solve for x::

        minimize    l2norm(A(x) - b)
        subject to  x[zeros] == 0


    Parameters
    ----------

    A : callable
        Linear operator that would accept `x0` as input.

    Astar : callable
        Linear operator that is the adjoint of `A`, would accept `b` as input.

    b : array
        Constant vector used in l2-norm expression.

    zeros : boolean array
        Where True, requires that corresponding entries in x be zero.

    x0 : array
        Initial iterate for x.


    Returns
    -------

    x : array
        Solution to the optimization problem.


    Other Parameters
    ----------------

    solver : {solvers}, optional
        Algorithm to use.

    **kwargs
        Additional keyword arguments passed to the solver.


    See Also
    --------

    nnls, {seealso}

    """
    F = ZerosInd(z=zeros)
    G = L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs
