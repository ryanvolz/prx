# ----------------------------------------------------------------------------
# Copyright (c) 2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Covariance likelihood functions."""

import numpy as np

__all__ = ('normcov_negloglik',)


def normcov_negloglik(X, S):
    r"""Normal covariance negative log-likelihood: ``log(det(X)) + tr(X \ S)``.

    Parameters
    ----------

    X : (..., M, M) array_like
        Covariance matrix parameter. Must be Hermitian, with
        ``np.allclose(X, X.conj().swapaxes(-2, -1))``, and positive definite.

    S : (..., M, M) array_like
        Sample covariance matrix or matrices. Must be Hermitian, with
        ``np.allclose(S, S.conj().swapaxes(-2, -1))``, and positive
        semidefinite.


    Returns
    -------

    value : float | array_like
        Negative log-likelihood value or array of values.


    See Also
    --------

    .grad_normcov_negloglik

    """
    sign, logdet = np.linalg.slogdet(X)
    val = np.real(
        np.log(sign) + logdet
        + np.trace(np.linalg.solve(X, S), axis1=-2, axis2=-1)
    )

    return val
