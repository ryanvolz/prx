# -----------------------------------------------------------------------------
# Copyright (c) 2014-2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------
"""Function classes for covariance likelihood functions."""

from __future__ import division

import numpy as np

from . import base as _base
from ..fun.covariance import normcov_negloglik
from ..grad.covariance import grad_normcov_negloglik

__all__ = (
    'NormCovNegLogLik',
)


class NormCovNegLogLik(_base.BaseFunction):
    # TODO, consider building this up from more basic logdet and trace objs
    r"""Function class for the normal covariance negative log-likelihood.

    For a multivariate normal random variable with a measured sample
    covariance, this function represents the (shifted and scaled) negative
    log-likelihood of the variable's unknown covariance matrix
    parameter.

    {function_summary}


    Attributes
    ----------

    S : (..., M, M) array_like
        Sample covariance matrix or matrices. Must be Hermitian, with
        ``S == S.conj().swapaxes(-2, -1)``, and positive semidefinite.

    {function_attributes}


    See Also
    --------

    .IndicatorFunction : Parent class.


    Notes
    -----

    ``fun(X) = log(det(X)) + tr(X \ S)``

    ``grad(X) = X \ (X - S) / S``

    This function is convex when restricted to the domain where X <= 2*S or
    equivalently where (2*S - X) is positive semidefinite.

    """

    def __init__(
        self, S, scale=None, stretch=None, shift=None, linear=None, const=None,
    ):
        """Create Function defining normal covariance negative log-likelihood.

        {init_summary}


        Parameters
        ----------

        S : (..., M, M) array_like
            Sample covariance matrix or matrices. Must be Hermitian, with
            ``np.allclose(S, S.conj().swapaxes(-2, -1))``, and positive
            semidefinite.

        {init_params}

        """
        if not np.allclose(S, S.conj().swapaxes(-2, -1)):
            raise ValueError('Sample covariance S must be Hermitian.')
        w = np.linalg.eigvalsh(S)
        if np.any(
            np.min(w, axis=-1) < -8*np.finfo(w.dtype).eps*np.max(w, axis=-1)
        ):
            errstr = 'Sample covariance S must be positive semidefinite.'
            raise ValueError(errstr)
        self._S = S

        super(NormCovNegLogLik, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

    @property
    def _conjugate_class(self):
        raise NotImplementedError

    @property
    def S(self):
        """Sample covariance matrix associated with the likelihood function."""
        return self._S

    def fun(self, X):
        """Evaluate normal covariance negative log-likelihood function."""
        return np.sum(normcov_negloglik(X, self._S))

    def grad(self, X):
        """Evaluate normal covariance negative log-likelihood gradient."""
        return grad_normcov_negloglik(X, self._S)
