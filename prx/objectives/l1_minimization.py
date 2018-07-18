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

__all__ = ('_bpdn', '_dantzig', '_l1rls', '_srlasso',
           'BPDN', 'Dantzig')


class BPDN(_split_objectives.SplitObjectiveAffineProx):
    """Solve the basis pursuit denoising problem.

    Given `A`, `b`, `eps`, and `w`, solve for ``x``::

        minimize    l1norm(w * x)
        subject to  l2norm(A(x) - b) <= eps

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    Dantzig, .L1RLS, .L1RootLS, .Lasso, {algorithm_self}


    Notes
    -----

    {algorithm_notes}


    References
    ----------

    {algorithm_references}

    """

    _doc_objective_self = ':class:`.BPDN`'

    _doc_objective_parameters = """
    eps : float
        A positive number describing the l2-norm constraint.

    w : float | array, optional
        A scalar or array giving the weighting on ``x`` inside the l1-norm.

    axis : False | None | int | tuple of ints, optional
        If this argument is given, the l1-norm is replaced by the combined
        l1- and l2-norm with the l2-norm taken over the specified axes:
        ``l1norm(l2norm(x, axis))``. This can be used to solve the "block"
        or "group" sparsity problem.

    """

    def __init__(self, eps=1, w=1, axis=False, **kwargs):
        """Initialize basis pursuit denoising problem.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}

        """
        self.eps = eps
        self.w = w
        self.axis = axis
        super(BPDN, self).__init__(**kwargs)

    def validate_params(self):
        """."""
        if self.eps <= 0:
            raise ValueError('eps must be positive')

        self.w = np.asarray(self.w)
        if np.any(self.w == 0):
            raise ValueError('w must be nonzero')

        # set split objective parameters
        if self.axis is False:
            F = _fun.L1Norm(stretch=self.w)
        else:
            F = _fun.L1L2Norm(axis=self.axis, stretch=self.w)
        G = _fun.L2BallInd(radius=self.eps)
        self.F = F.fun
        self.proxF = F.prox
        self.G = G.fun
        self.proxG = G.prox

        return super(BPDN, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(BPDN, self).get_params(deep=deep)
        params.update(eps=self.eps, w=self.w, axis=self.axis)
        return params

    def set_params(self, eps=None, w=None, axis=None, **alg_params):
        """."""
        self._assign_params(eps=eps, w=w, axis=axis)
        return super(BPDN, self).set_params(**alg_params)


@backends(_alg._admmlin, _alg._pdhg)
def _bpdn(A, Astar, b, eps, x0, **kwargs):
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

    _dantzig, _l1rls, ._lasso, _srlasso, {seealso}

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


class Dantzig(_split_objectives.SplitObjectiveAffineProx):
    """Solve the Dantzig selector problem.

    Given `A`, `b`, `delta`, and `w`, solve for ``x``::

        minimize    l1norm(w * x)
        subject to  linfnorm(A(x) - b) <= delta

    For the classic formulation of the Dantzig selector problem, `A` and `b`
    should be composed such that ``A(x) = Bstar(B(x))`` and ``b = Bstar(c)``
    for `B` and `c` describing the underlying measurement equation
    ``B(x) = c + noise`` and `Bstar` giving the adjoint of `B`. In that case,
    `A` will be self-adjoint with ``A == Astar``.

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    BPDN, .L1RLS, .L1RootLS, .Lasso, {algorithm_self}


    Notes
    -----

    {algorithm_notes}


    References
    ----------

    {algorithm_references}

    """

    _doc_objective_self = ':class:`.Dantzig`'

    _doc_objective_parameters = """
    delta : float
        A positive number describing the linf-norm constraint.

    w : float | array, optional
        A scalar or array giving the weighting on ``x`` inside the l1-norm.

    axis : False | None | int | tuple of ints, optional
        If this argument is given, the l1-norm is replaced by the combined
        l1- and l2-norm with the l2-norm taken over the specified axes:
        ``l1norm(l2norm(x, axis))``. This can be used to solve the "block"
        or "group" sparsity problem.

    """

    def __init__(self, delta=1, w=1, axis=False, **kwargs):
        """Initialize Dantzig selector problem.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}

        """
        self.delta = delta
        self.w = w
        self.axis = axis
        super(Dantzig, self).__init__(**kwargs)

    def validate_params(self):
        """."""
        if self.delta <= 0:
            raise ValueError('delta must be positive')

        self.w = np.asarray(self.w)
        if np.any(self.w == 0):
            raise ValueError('w must be nonzero')

        # set split objective parameters
        if self.axis is False:
            F = _fun.L1Norm(stretch=self.w)
        else:
            F = _fun.L1L2Norm(axis=self.axis, stretch=self.w)
        G = _fun.LInfBallInd(radius=self.delta)
        self.F = F.fun
        self.proxF = F.prox
        self.G = G.fun
        self.proxG = G.prox

        return super(Dantzig, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(Dantzig, self).get_params(deep=deep)
        params.update(delta=self.delta, w=self.w, axis=self.axis)
        return params

    def set_params(self, delta=None, w=None, axis=None, **alg_params):
        """."""
        self._assign_params(delta=delta, w=w, axis=axis)
        return super(Dantzig, self).set_params(**alg_params)


@backends(_alg._admmlin, _alg._pdhg)
def _dantzig(A, Astar, b, delta, x0, **kwargs):
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

    _bpdn, _l1rls, ._lasso, _srlasso, {seealso}


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


@backends(_alg._proxgradaccel, _alg._admmlin, _alg._pdhg, _alg._proxgrad)
def _l1rls(A, Astar, b, lmbda, x0, **kwargs):
    """Solve the l1-regularized least squares problem.

    Given A, b, and lmbda, solve for x::

        minimize  0.5*l2norm(A(x) - b)**2 + lmbda*l1norm(x)

    This problem is sometimes called the LASSO since it is equivalent to the
    original :func:`_lasso` formulation for appropriate lmbda as a function of
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

    _bpdn, _dantzig, ._lasso, _srlasso, {seealso}


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


@backends(_alg._admmlin, _alg._pdhg)
def _srlasso(A, Astar, b, lmbda, x0, **kwargs):
    """Solve the square root LASSO problem.

    Given A, b, and lmbda, solve for x::

        minimize  l2norm(A(x) - b) + lmbda*l1norm(x)

    The square root LASSO may be preferred to :func:`_l1rls` since the
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

    _bpdn, _dantzig, _l1rls, ._lasso, {seealso}


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
