# -----------------------------------------------------------------------------
# Copyright (c) 2014-2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------
"""Objective classes for covariance likelihood functions.

.. currentmodule:: prx.objectives.covariance

Covariance
-----------

.. autosummary::
    :toctree:

    NegNormCovarianceLogLikelihood

Related Indicators
------------------

.. autosummary::
    :toctree:


"""

from __future__ import division

import numpy as np

from .objective_classes import (BaseObjective, _class_docstring_wrapper,
                                _init_docstring_wrapper)

__all__ = (
    'NegNormCovarianceLogLikelihood',
)


class NegNormCovarianceLogLikelihood(BaseObjective):
    # TODO, consider building this up from more basic logdet and trace fcns
    # TODO, consider adding getter and setter functions for S
    __doc__ = _class_docstring_wrapper(
        """Objective class for the negative log-likelihood normal covariance.

        {common_summary}


        Attributes
        ----------

        S : array with shape (N, N) or (M, N, N)
            Sample covariance matrix or matrices. Must be Hermitian, with
            ``S == S.conj().transpose()``.

        {common_attributes}


        See Also
        --------

        .IndicatorObjective : Parent class.


        Notes
        -----


        """
    )

    @_init_docstring_wrapper
    def __init__(self, S, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """Create Objective that defines an indicator function.

        {common_summary}

        Since this objective is an indicator, `scale` can be eliminated::

            s*f(a*x + b) => f(a*x + b)


        Parameters
        ----------

        S : array with shape (N, N) or (M, N, N)
            Sample covariance matrix or matrices. Must be Hermitian, with
            ``S == S.conj().transpose()``.

        {common_params}

        """
        self._S = S
        assert len(self._S.shape) in {2, 3}, \
            "Error, S must be an N x N matrix or a stack of M N x N matrices"
        # Reshape S to be compatible with for loop over matrix stack if S only
        # has 1 matrix
        if len(self._S) == 2:
            self._S = self._S.reshape((1, -1))
        # eliminate scale
        if scale is not None:
            scale = None

        super(NegNormCovarianceLogLikelihood, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const)

    @property
    def _conjugate_class(self):
        return NegNormCovarianceLogLikelihood

    def fun(self, x):
        # TODO docstring
        assert len(x.shape) in {2, 3}, (
            "Error, input matrix x must be an N x N matrix or a stack of M"
            " N x N matrices"
        )
        # Reshape x to be compatible with for loop over matrix stack if x only
        # has 1 matrix
        if len(x.shape) == 2:
            x = x.reshape((1, -1))
        assert x.shape == self._S.shape, (
            "Error, dimensionality of x and sample covariance matrix S must"
            " match"
        )
        m = self._S.shape[0]
        val = 0
        for k in range(m):
            s_k = self._S[k]
            x_k = x[k]
            val = val - np.real(
                np.log(np.linalg.det(x_k))
                + np.trace(np.linalg.solve(x_k, s_k))
            )

        return -val

    def grad(self, x):
        # TODO docstring
        length = self._S.shape[1]
        m = self._S.shape[0]
        g = np.zeros((m, length, length), dtype=np.complex)

        assert len(x.shape) in {2, 3}, (
            "Error, input matrix x must be an N x N matrix or a stack of M"
            " N x N matrices"
        )
        # Reshape x to be compatible with for loop over matrix stack if x only
        # has 1 matrix
        if len(x.shape) == 2:
            x = x.reshape((1, -1))

        for i in range(m):
            s_i = self._S[i]
            x_i = x[i]
            g_i = np.linalg.solve(x_i.T, np.linalg.solve(x_i, (x_i - s_i)).T).T
            g_i = (g_i + np.conj(g_i.T)) / 2
            g[i] = g_i

        return g
