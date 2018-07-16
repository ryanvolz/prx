# ----------------------------------------------------------------------------
# Copyright (c) 2015-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Optimization objectives for minimizing the l1-norm."""

import numpy as np

from . import split_objectives as _split_objectives
from .. import algorithms as _alg
from .. import functions as _fun
from ._common import backends

__all__ = ('bpdn', 'dantzig', 'L1RLS', '_l1rls', 'srlasso')


@backends(_alg.admmlin, _alg.pdhg)
def bpdn(A, Astar, b, eps, x0, **kwargs):
    """Solve the basis pursuit denoising problem.

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

    dantzig, l1rls, .lasso, srlasso, {seealso}

    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = _fun.L1Norm()
    else:
        F = _fun.L1L2Norm(axis=axis)
    G = _fun.L2BallInd(radius=eps)

    args = (F, G, A, Astar, b, x0)

    return args, kwargs


@backends(_alg.admmlin, _alg.pdhg)
def dantzig(A, Astar, b, delta, x0, **kwargs):
    """Solve the Dantzig selector problem.

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

    bpdn, l1rls, .lasso, srlasso, {seealso}


    References
    ----------

    .. [1] E. Candes and T. Tao, "The Dantzig selector: Statistical estimation
       when p is much larger than n," Ann. Statist., vol. 35, no. 6, pp.
       2313-2351, Dec. 2007.


    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = _fun.L1Norm()
    else:
        F = _fun.L1L2Norm(axis=axis)
    G = _fun.LInfBallInd(radius=delta)

    # "A" and "Astar" in admmlin notation
    def AsA(x):
        return Astar(A(x))

    # "b" in admmlin notation
    Asb = Astar(b)

    args = (F, G, AsA, AsA, Asb, x0)

    return args, kwargs


class L1RLS(_split_objectives.SplitObjectiveAffine):
    """Solve the l1-regularized least squares problem.

    Given `A`, `b`, `lmbda`, and `w`, solve for ``x``::

        minimize  0.5*l2norm(A(x) - b)**2 + lmbda*l1norm(w * x)

    This problem is sometimes called the LASSO since it is equivalent to the
    original `Lasso` formulation for appropriate lmbda as a function of tau.

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    BPDN, Dantzig, .Lasso, SRLasso, {algorithm_self}


    Notes
    -----

    {algorithm_notes}

    The accelerated proximal gradient algorithm applied specifically to the
    l1-regularized least squares problem is discussed in [#BT09]_.


    References
    ----------

    {algorithm_references}

    .. [#BT09] {BT09}

    """

    _doc_objective_self = ':class:`.L1RLS`'

    _doc_objective_parameters = """
    lmbda : float
        A positive number giving the l1-norm scaling.

    w : float | array, optional
        A scalar or array giving the weighting on ``x`` inside the l1-norm.

    axis : False | None | int | tuple of ints, optional
        If this argument is given, the l1-norm is replaced by the combined
        l1- and l2-norm with the l2-norm taken over the specified axes:
        ``l1norm(l2norm(x, axis))``. This can be used to solve the "block"
        or "group" sparsity problem.

    """

    def __init__(self, lmbda=1, w=1, axis=False, **kwargs):
        """Initialize l1-regularized least squares problem.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}

        """
        self.lmbda = lmbda
        self.w = w
        self.axis = axis
        super(L1RLS, self).__init__(**kwargs)

    def validate_params(self):
        """."""
        # check lmbda > 0
        if self.lmbda <= 0:
            raise ValueError('lmbda must be positive')

        self.w = np.asarray(self.w)
        if np.any(self.w == 0):
            raise ValueError('w must be nonzero')

        # set split objective parameters based on lmbda and axis
        if self.axis is False:
            F = _fun.L1Norm(scale=self.lmbda, stretch=self.w)
        else:
            F = _fun.L1L2Norm(axis=self.axis, scale=self.lmbda, stretch=self.w)
        G = _fun.L2NormSqHalf()
        self.F = F.fun
        self.proxF = F.prox
        self.G = G.fun
        self.gradG = G.grad
        self.proxG = G.prox

        return super(L1RLS, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(L1RLS, self).get_params(deep=deep)
        params.update(lmbda=self.lmbda, w=self.w, axis=self.axis)
        return params

    def set_params(self, lmbda=None, w=None, axis=None, **alg_params):
        """."""
        self._assign_params(lmbda=lmbda, w=w, axis=axis)
        return super(L1RLS, self).set_params(**alg_params)


@backends(_alg._proxgradaccel, _alg.admmlin, _alg.pdhg, _alg._proxgrad)
def _l1rls(A, Astar, b, lmbda, x0, **kwargs):
    """Solve the l1-regularized least squares problem.

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

    bpdn, dantzig, .lasso, srlasso, {seealso}


    References
    ----------

    .. [1] A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems," SIAM Journal on Imaging
       Sciences, vol. 2, no. 1, pp. 183-202, Jan. 2009.

    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = _fun.L1Norm(scale=lmbda)
    else:
        F = _fun.L1L2Norm(axis=axis, scale=lmbda)
    G = _fun.L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs


@backends(_alg.admmlin, _alg.pdhg)
def srlasso(A, Astar, b, lmbda, x0, **kwargs):
    """Solve the square root LASSO problem.

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

    bpdn, dantzig, l1rls, .lasso, {seealso}


    References
    ----------

    .. [1] A. Belloni, V. Chernozhukov, and L. Wang, "Square-root lasso:
       pivotal recovery of sparse signals via conic programming," Biometrika,
       vol. 98, no. 4, pp. 791-806, Dec. 2011.


    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = _fun.L1Norm(scale=lmbda)
    else:
        F = _fun.L1L2Norm(axis=axis, scale=lmbda)
    G = _fun.L2Norm()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs
