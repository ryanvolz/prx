# ----------------------------------------------------------------------------
# Copyright (c) 2015-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Optimization objectives for minimizing the l2-norm."""

from .. import algorithms as _alg
from .. import functions as _fun
from ._common import backends

__all__ = ('lasso', 'nnls', 'zcls')


@backends(_alg._proxgradaccel, _alg.admmlin, _alg.pdhg, _alg.proxgrad)
def lasso(A, Astar, b, tau, x0, **kwargs):
    """Solve the LASSO problem.

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

    .bpdn, .dantzig, .l1rls, .srlasso, {seealso}


    References
    ----------

    .. [1] R. Tibshirani, "Regression Shrinkage and Selection via the Lasso,"
       Journal of the Royal Statistical Society. Series B (Methodological),
       vol. 58, no. 1, pp. 267-288, Jan. 1996.


    """
    F = _fun.L1BallInd(radius=tau)
    G = _fun.L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs


@backends(_alg._proxgradaccel, _alg.admmlin, _alg.pdhg, _alg.proxgrad)
def nnls(A, Astar, b, x0, **kwargs):
    """Solve the non-negative least squares problem.

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
    F = _fun.NNegInd()
    G = _fun.L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs


@backends(_alg._proxgradaccel, _alg.admmlin, _alg.pdhg, _alg.proxgrad)
def zcls(A, Astar, b, zeros, x0, **kwargs):
    """Solve the zero-constrained least squares problem.

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
    F = _fun.ZerosInd(z=zeros)
    G = _fun.L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs
