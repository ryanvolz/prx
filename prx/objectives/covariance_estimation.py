# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Optimization objectives for estimating covariances."""

from . import split_objectives as _split_objectives
from .. import functions as _fun

__all__ = ('NormCovML',)


class NormCovML(_split_objectives.SplitObjectiveAffine):
    ur"""Maximum likelihood covariance estimation of a normal random variable.

    Given `A`, `b`, `s`, and `epsilon`, solve for ``x``::

        minimize    log(det( A(x) - b )) + tr( (A(x) - b) \ s )
        subject to  x ≽ epsilon * I

    The function above is the negative log-likelihood of the covariance ``x``
    given a measured sample covariance `s`, and the constraint ensures that
    the solution is positive semidefinite for ``epsilon == 0``. If `A` were the
    identity and `b` the zero matrix, then the closed-form solution to this
    problem would be ``x = s``.

    A typical case arises when the measured random variable ``m`` is an affine
    function ``m = a @ p + n`` of the variable ``p`` with unknown covariance
    ``x`` and noise ``n`` with known covariance ``-b``. Then one can solve for
    ``x`` by applying this problem with ``A(x) = a @ x @ a.H`` and
    ``s = np.cov(np.stack([m0, ..., mk]).T)``.

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    See Also
    --------

    {algorithm_self}


    Notes
    -----

    {algorithm_notes}


    References
    ----------

    {algorithm_references}

    """

    _doc_objective_self = ':class:`.NormCovML`'

    _doc_objective_parameters = """
    s : (..., M, M) array_like
        Sample covariance matrix or matrices. Must be Hermitian, with
        ``np.allclose(s, s.conj().swapaxes(-2, -1))``, and positive
        semidefinite.

    epsilon : float, optional
        Constraint semidefiniteness constant. The constraint ensures that
        ``x - epsilon * I`` is positive semidefinite, where ``I`` is the
        identity matrix.

    stokes : bool, optional
        If True, treat the variable ``x`` as an array of Stokes parameters
        representing a covariance instead of as the covariance matrix itself.
        This enables use of the indicator function :class:`.PSDIndStokes`
        instead of :class:`.PSDInd`. Note that the supplied `A` and `Astar`
        functions must then also handle ``x`` given as Stokes parameters.
        See :class:`.PSDIndStokes` for details.

    """

    def __init__(self, s=None, epsilon=0, stokes=False, **kwargs):
        """Initialize normal covariance maximum likelihood problem.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}

        """
        self.s = s
        self.epsilon = epsilon
        self.stokes = stokes
        super(NormCovML, self).__init__(**kwargs)

    def validate_params(self):
        """."""
        if self.s is None:
            raise ValueError('Must set sample covariance matrix s.')

        self.epsilon = float(self.epsilon)

        # set split objective parameters
        if self.stokes:
            F = _fun.PSDIndStokes(epsilon=self.epsilon)
        else:
            F = _fun.PSDInd(epsilon=self.epsilon)
        G = _fun.NormCovNegLogLik(S=self.s)
        self.F = F.fun
        self.proxF = F.prox
        self.G = G.fun
        self.gradG = G.grad
        self.proxG = G.prox

        return super(NormCovML, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(NormCovML, self).get_params(deep=deep)
        params.update(s=self.s, epsilon=self.epsilon, stokes=self.stokes)
        return params

    def set_params(self, s=None, epsilon=None, stokes=None, **alg_params):
        """."""
        self._assign_params(s=s, epsilon=epsilon, stokes=stokes)
        return super(NormCovML, self).set_params(**alg_params)
