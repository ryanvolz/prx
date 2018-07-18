# ----------------------------------------------------------------------------
# Copyright (c) 2015-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Optimization objectives for minimizing the l2-norm."""

import numpy as np

from . import split_objectives as _split_objectives
from .. import algorithms as _alg
from .. import functions as _fun
from ._common import backends

__all__ = ('_lasso', '_nnls', '_zcls',
           'L1RLS', 'L1RootLS', 'Lasso', 'NNLS', 'ZCLS')


class L1RLS(_split_objectives.SplitObjectiveAffine):
    """Solve the l1-regularized least squares problem.

    Given `A`, `b`, `lmbda`, and `w`, solve for ``x``::

        minimize  0.5*l2norm(A(x) - b)**2 + lmbda*l1norm(w * x)

    This problem is sometimes called the LASSO since it is equivalent to the
    original :class:`Lasso` formulation for appropriate lmbda as a function of
    tau.

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    .BPDN, .Dantzig, L1RootLS, Lasso, {algorithm_self}


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

        # set split objective parameters
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


class L1RootLS(_split_objectives.SplitObjectiveAffineProx):
    """Solve the l1-regularized root least squares problem.

    Given `A`, `b`, `lmbda`, and `w`, solve for ``x``::

        minimize  l2norm(A(x) - b) + lmbda*l1norm(w * x)

    This problem is was introduced in [#BCW11]_ as the "square-root LASSO", but
    it is more similar to the l1-regularized least squares problem of
    :class:`L1RLS` than Tibshirani's original LASSO formulation [#Tib96]_ since
    it employs l1 minimization instead of an l1-ball constraint.

    :class:`L1RootLS` may be preferred to :class:`L1RLS` since the selection
    of the parameter `lmbda` need not depend on the noise level of the
    measurements `b` (under the model where ``b = A(x) + n`` for some noise n).
    See [#BCW11]_ for more details.

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    .BPDN, .Dantzig, L1RLS, Lasso, {algorithm_self}


    Notes
    -----

    {algorithm_notes}

    The l1-regularized root least squares problem specifically is discussed in
    [#BCW11]_.


    References
    ----------

    {algorithm_references}

    .. [#BCW11] {BCW11}

    .. [#Tib96] {Tib96}

    """

    _doc_objective_self = ':class:`.L1RootLS`'

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
        """Initialize l1-regularized root least squares problem.

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
        super(L1RootLS, self).__init__(**kwargs)

    def validate_params(self):
        """."""
        # check lmbda > 0
        if self.lmbda <= 0:
            raise ValueError('lmbda must be positive')

        self.w = np.asarray(self.w)
        if np.any(self.w == 0):
            raise ValueError('w must be nonzero')

        # set split objective parameters
        if self.axis is False:
            F = _fun.L1Norm(scale=self.lmbda, stretch=self.w)
        else:
            F = _fun.L1L2Norm(axis=self.axis, scale=self.lmbda, stretch=self.w)
        G = _fun.L2Norm()
        self.F = F.fun
        self.proxF = F.prox
        self.G = G.fun
        self.proxG = G.prox

        return super(L1RootLS, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(L1RootLS, self).get_params(deep=deep)
        params.update(lmbda=self.lmbda, w=self.w, axis=self.axis)
        return params

    def set_params(self, lmbda=None, w=None, axis=None, **alg_params):
        """."""
        self._assign_params(lmbda=lmbda, w=w, axis=axis)
        return super(L1RootLS, self).set_params(**alg_params)


class Lasso(_split_objectives.SplitObjectiveAffine):
    """Solve the LASSO problem.

    LASSO stands for "least absolute shrinkage and selection operator".

    Given `A`, `b`, `tau`, and `w`, solve for ``x``::

        minimize    0.5*l2norm(A(x) - b)**2
        subject to  l1norm(w * x) <= tau

    This definition follows Tibshirani's original formulation from [#Tib96]_.
    The :class:`L1RLS` problem is sometimes called the LASSO since they are
    equivalent for appropriate selection of `lmbda` as a function of `tau`.

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    .BPDN, .Dantzig, L1RLS, L1RootLS, {algorithm_self}


    Notes
    -----

    {algorithm_notes}


    References
    ----------

    {algorithm_references}

    .. [#Tib96] {Tib96}

    """

    _doc_objective_self = ':class:`.Lasso`'

    _doc_objective_parameters = """
    tau : float
        A positive number describing the l1-norm constraint.

    w : float | array, optional
        A scalar or array giving the weighting on ``x`` inside the l1-norm.

    """

    def __init__(self, tau=1, w=1, **kwargs):
        """Initialize the LASSO problem.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}

        """
        self.tau = tau
        self.w = w
        super(Lasso, self).__init__(**kwargs)

    def validate_params(self):
        """."""
        if self.tau <= 0:
            raise ValueError('tau must be positive')

        self.w = np.asarray(self.w)
        if np.any(self.w == 0):
            raise ValueError('w must be nonzero')

        # set split objective parameters
        F = _fun.L1BallInd(radius=self.tau, stretch=self.w)
        G = _fun.L2NormSqHalf()
        self.F = F.fun
        self.proxF = F.prox
        self.G = G.fun
        self.gradG = G.grad
        self.proxG = G.prox

        return super(Lasso, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(Lasso, self).get_params(deep=deep)
        params.update(tau=self.tau, w=self.w)
        return params

    def set_params(self, tau=None, w=None, **alg_params):
        """."""
        self._assign_params(tau=tau, w=w)
        return super(Lasso, self).set_params(**alg_params)


@backends(_alg._proxgradaccel, _alg._admmlin, _alg._pdhg, _alg._proxgrad)
def _lasso(A, Astar, b, tau, x0, **kwargs):
    """Solve the LASSO problem.

    Given A, b, and tau, solve for x::

        minimize    0.5*l2norm(A(x) - b)**2
        subject to  l1norm(x) <= tau

    This definition follows Tibshirani's original formulation from [1]_.
    The :func:`_l1rls` problem is sometimes called the LASSO since they are
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

    ._bpdn, ._dantzig, ._l1rls, ._srlasso, {seealso}


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


class NNLS(_split_objectives.SplitObjectiveAffine):
    """Solve the non-negative least squares problem.

    Given `A` and `b`, solve for ``x``::

        minimize    0.5*l2norm(A(x) - b)**2
        subject to  x >= 0

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    ZCLS, {algorithm_self}


    Notes
    -----

    {algorithm_notes}


    References
    ----------

    {algorithm_references}

    """

    _doc_objective_self = ':class:`.NNLS`'

    _doc_objective_parameters = ''

    def __init__(self, **kwargs):
        """Initialize non-negative least squares problem.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}

        """
        super(NNLS, self).__init__(**kwargs)

    def validate_params(self):
        """."""
        # set split objective parameters
        F = _fun.NNegInd()
        G = _fun.L2NormSqHalf()
        self.F = F.fun
        self.proxF = F.prox
        self.G = G.fun
        self.gradG = G.grad
        self.proxG = G.prox

        return super(NNLS, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(NNLS, self).get_params(deep=deep)
        return params

    def set_params(self, **alg_params):
        """."""
        return super(NNLS, self).set_params(**alg_params)


@backends(_alg._proxgradaccel, _alg._admmlin, _alg._pdhg, _alg._proxgrad)
def _nnls(A, Astar, b, x0, **kwargs):
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

    _zcls, {seealso}

    """
    F = _fun.NNegInd()
    G = _fun.L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs


class ZCLS(_split_objectives.SplitObjectiveAffine):
    """Solve the zero-constrained least squares problem.

    Given `A`, `b`, and `zeros`, solve for ``x``::

        minimize    0.5*l2norm(A(x) - b)**2
        subject to  x[zeros] == 0

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    NNLS, {algorithm_self}


    Notes
    -----

    {algorithm_notes}


    References
    ----------

    {algorithm_references}

    """

    _doc_objective_self = ':class:`.ZCLS`'

    _doc_objective_parameters = """
    zeros : boolean array
        Where True, requires that corresponding entries in x be zero.

    """

    def __init__(self, zeros=None, **kwargs):
        """Initialize the zero-constrained least squares problem.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}

        """
        self.zeros = zeros
        super(ZCLS, self).__init__(**kwargs)

    def validate_params(self):
        """."""
        if self.zeros is None:
            raise ValueError('Must specify zeros (zero locations bool array)')

        # set split objective parameters
        F = _fun.ZerosInd(z=self.zeros)
        G = _fun.L2NormSqHalf()
        self.F = F.fun
        self.proxF = F.prox
        self.G = G.fun
        self.gradG = G.grad
        self.proxG = G.prox

        return super(ZCLS, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(ZCLS, self).get_params(deep=deep)
        params.update(zeros=self.zeros)
        return params

    def set_params(self, zeros=None, **alg_params):
        """."""
        self._assign_params(zeros=zeros)
        return super(ZCLS, self).set_params(**alg_params)


@backends(_alg._proxgradaccel, _alg._admmlin, _alg._pdhg, _alg._proxgrad)
def _zcls(A, Astar, b, zeros, x0, **kwargs):
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

    _nnls, {seealso}

    """
    F = _fun.ZerosInd(z=zeros)
    G = _fun.L2NormSqHalf()

    args = (F, G, A, Astar, b, x0)

    return args, kwargs
