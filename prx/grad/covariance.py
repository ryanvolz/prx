# ----------------------------------------------------------------------------
# Copyright (c) 2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Gradients of covariance likelihood functions.

.. currentmodule:: prx.grad.covariance

.. autosummary::
    :toctree:

    grad_normcov_negloglik


"""

import numpy as np

__all__ = ('grad_normcov_negloglik',)


def grad_normcov_negloglik(X, S):
    r"""Normal covariance negative log-likelihood grad: ``X \ (X - S) / S``.

    The corresponding function is: ``log(det(X)) + tr(X \ S)``.


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

    gradient : array_like
        Gradient array.


    See Also
    --------

    .normcov_negloglik

    """
    # ( X \ (X - S) / X ) with Hermitian X and S
    G = np.linalg.solve(
        X, np.linalg.solve(X, X - S).conj().swapaxes(-2, -1),
    )
    # ensure result is exactly Hermitian for numerical stability
    G = (G + G.conj().swapaxes(-2, -1)) / 2

    return G
