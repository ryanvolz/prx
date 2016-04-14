# ----------------------------------------------------------------------------
# Copyright (c) 2015-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Optimization problems minimizing the l1-norm.

.. currentmodule:: prx.problems.l1_minimization

.. autosummary::
    :toctree:

    bpdn
    dantzig
    l1rls
    srlasso

"""

from .. import objectives as _obj
from .. import algorithms as _alg
from ._common import backends

__all__ = ['bpdn', 'dantzig', 'l1rls', 'srlasso']


@backends(_alg.admmlin, _alg.pdhg)
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
        F = _obj.L1Norm()
    else:
        F = _obj.L1L2Norm(axis=axis)
    G = _obj.L2BallInd(radius=eps)

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(_alg.admmlin, _alg.pdhg)
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
        F = _obj.L1Norm()
    else:
        F = _obj.L1L2Norm(axis=axis)
    G = _obj.LInfBallInd(radius=delta)

    # "A" and "Astar" in admmlin notation
    AsA = lambda x: Astar(A(x))
    # "b" in admmlin notation
    Asb = Astar(b)

    args = (F, G, AsA, AsA, Asb, x0)

    return args, kwargs

@backends(_alg.proxgradaccel, _alg.admmlin, _alg.pdhg, _alg.proxgrad)
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
        F = _obj.L1Norm(scale=lmbda)
    else:
        F = _obj.L1L2Norm(axis=axis, scale=lmbda)
    G = _obj.L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs

@backends(_alg.admmlin, _alg.pdhg)
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
        F = _obj.L1Norm(scale=lmbda)
    else:
        F = _obj.L1L2Norm(axis=axis, scale=lmbda)
    G = _obj.L2Norm()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs
