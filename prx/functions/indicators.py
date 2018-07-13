# ----------------------------------------------------------------------------
# Copyright (c) 2014-2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Function classes for indicator functions."""

from __future__ import division

import numpy as np

from . import base as _base
from ..prox.projections import (proj_nneg, proj_npos, proj_psd,
                                proj_psd_stokes, proj_zeros)

__all__ = (
    'NNegInd', 'NPosInd', 'ZerosInd',
    'PSDInd', 'PSDIndStokes',
)


class NNegInd(_base.IndicatorFunction):
    """Function class for the non-negative indicator function.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    .IndicatorFunction : Parent class.
    NPosInd : Conjugate function.
    .proj_nneg : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for vectors with only non-negative
    entries and infinity if any of the entries are negative.

    {function_notes}

    The prox operator is Euclidean projection onto the non-negative
    halfspace, i.e. the negative entries are set to zero.

    """

    @property
    def _conjugate_class(self):
        return NPosInd

    def fun(self, x):
        """Indicator function for non-negative values."""
        if np.any(x < 0):
            return np.inf
        else:
            return 0

    def prox(self, x, lmbda=1):
        """Projection onto the non-negative reals (negatives set to zero)."""
        return proj_nneg(x)


class NPosInd(_base.IndicatorFunction):
    """Function class for the non-positive indicator function.

    {function_summary}


    Attributes
    ----------

    {function_attributes}


    See Also
    --------

    .IndicatorFunction : Parent class.
    NNegInd : Conjugate function.
    .proj_npos : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for vectors with only non-positive
    entries and infinity if any of the entries are positive.

    {function_notes}

    The prox operator is Euclidean projection onto the non-positive
    halfspace, i.e. the positive entries are set to zero.

    """

    @property
    def _conjugate_class(self):
        return NNegInd

    def fun(self, x):
        """Indicator function for non-positive values."""
        if np.any(x > 0):
            return np.inf
        else:
            return 0

    def prox(self, x, lmbda=1):
        """Projection onto the non-positive reals (positives set to zero)."""
        return proj_npos(x)


class ZerosInd(_base.IndicatorFunction):
    """Function class for the indicator of prescribed zero elements.

    {function_summary}


    Attributes
    ----------

    z : boolean array
        Array specifying the zero locations, requiring x[z] == 0.

    {function_attributes}


    See Also
    --------

    .IndicatorFunction : Parent class.
    .proj_zeros : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for vectors with only zeros in the
    specified places, infinity if any of the required zero entries are
    nonzero.

    {function_notes}

    The prox operator is Euclidean projection onto the set with
    specified zeros (x[z] is set to 0).

    """

    def __init__(
        self, z, scale=None, stretch=None, shift=None, linear=None, const=None,
    ):
        """Create Function that defines an indicator function.

        {init_summary}


        Parameters
        ----------

        z : boolean array
            Array specifying the zero locations, requiring x[z] == 0.

        {init_params}

        """
        # stretch can be eliminated by bringing into shift
        # (since multiplying does not change zero locations)
        if stretch is not None and shift is not None:
            shift = shift/stretch
        stretch = None

        self._z = z

        # we can also eliminate scaling,
        # but this is taken care of by parent class
        super(ZerosInd, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const,
        )

    @property
    def _conjugate_class(self):
        return ZerosInd

    @property
    def _conjugate_args(self):
        # The conjugate of the zero indicator is the indicator for the
        # complimentary set of zeros.
        z = ~self._z
        kwargs = super(ZerosInd, self)._conjugate_args
        kwargs.update(z=z)
        return kwargs

    @property
    def z(self):
        """Boolean array giving the zero locations."""
        return self._z

    def fun(self, x):
        """Indicator function for zero elements z."""
        if np.any(x[self._z] != 0):
            return np.inf
        else:
            return 0

    def prox(self, x, lmbda=1):
        """Projection onto the set with specified zeros (x[z] is set to 0)."""
        return proj_zeros(x, z=self._z)


class PSDInd(_base.IndicatorFunction):
    r"""Function class for the positive semidefinite indicator function.

    {function_summary}


    Attributes
    ----------

    epsilon : float
        When `epsilon` is non-zero, the indicator is for the cone given by
        :math:`X \succeq \epsilon I` instead of the PSD cone. The prox
        operator sets eigenvalues less than `epsilon` to `epsilon`.

    {function_attributes}


    See Also
    --------

    .IndicatorFunction : Parent class.
    PSDIndStokes : Specialized PSD indicator for matrices as Stokes parameters.
    .proj_psd : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for matrices M that are positive
    semidefinite (all eigenvalues are >= 0) and infinity otherwise.

    {function_notes}

    The prox operator is Euclidean projection to the nearest positive
    semidefinite matrix (negative eigenvalues are set to zero).

    """

    def __init__(
        self, epsilon=0, scale=None, stretch=None, shift=None, linear=None,
        const=None,
    ):
        r"""Create Function that defines an indicator function.

        {init_summary}


        Parameters
        ----------

        epsilon : float, optional
            When `epsilon` is non-zero, the indicator is for the cone given by
            :math:`X \succeq \epsilon I` instead of the PSD cone. The prox
            operator sets eigenvalues less than `epsilon` to `epsilon`.

        {init_params}

        """
        self._epsilon = epsilon

        super(PSDInd, self).__init__(
            scale=scale, stretch=stretch, shift=shift,
            linear=linear, const=const,
        )

    # @property
    # def _conjugate_class(self):
    #     return PSDInd

    @property
    def epsilon(self):
        """Magnitude of identity shift to indicator cone."""
        return self._epsilon

    def fun(self, X):
        """Indicator function for positive semidefinite matrices."""
        w = np.linalg.eigvalsh(X)
        # check for negative eigenvalues, but be forgiving for very small
        # negative values relative to the maximum eignvalue
        if np.any(np.min(w, axis=-1) < -np.spacing(np.max(w, axis=-1))):
            return np.inf
        else:
            return 0

    def prox(self, X, lmbda=1):
        """Project Hermitian matrix onto positive semidefinite cone."""
        return proj_psd(X, epsilon=self._epsilon)


class PSDIndStokes(PSDInd):
    r"""Function class for the positive semidefinite indicator function.

    This indicator handles the special case of a block 2x2 Hermitian matrix
    represented by Stokes parameters where the projection can be achieved
    efficiently without an eigen-decomposition.

    {function_summary}


    Attributes
    ----------

    epsilon : float
        When `epsilon` is non-zero, the indicator is for the cone given by
        :math:`X \succeq \epsilon I` instead of the PSD cone. The prox
        operator sets eigenvalues less than `epsilon` to `epsilon`.

    {function_attributes}


    See Also
    --------

    .IndicatorFunction : Parent class.
    PSDInd : General PSD indicator.
    .proj_psd_stokes : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for matrices M that are positive
    semidefinite (all eigenvalues are >= 0) and infinity otherwise.

    {function_notes}

    The prox operator is Euclidean projection to the nearest positive
    semidefinite matrix (negative eigenvalues are set to zero).

    This indicator operates on 2x2 Hermitian blocks given in terms of Stokes
    parameters (I, Q, U, V)::

        [[  I + Q,   U - 1j*V, ],
         [ U + 1j*V,  I - Q,   ]]

    The Stokes parameters for all such blocks are passed as an array `s` such
    that::

        I = s[..., 0]
        Q = s[..., 1]
        U = s[..., 2]
        V = s[..., 3]

    The equivalent set of Hermitian matrices of shape (..., 2, 2) could be
    formed from the (...,)-shaped Stokes parameter arrays as::

        np.moveaxis(
            np.asarray([[ I + Q, U - 1j*V], [U + 1j*V, I - Q]]),
            [0, 1], [-2, -1],
        )

    """

    def fun(self, x_s):
        """Indicator for positive semidefinite matrices as Stokes params."""
        i, q, u, v = [x_s[..., k] for k in range(4)]
        if np.min(i) < -np.spacing(np.max(i)):
            # negative intensity (trace of 2x2 block), obviously not PSD
            return np.inf
        else:
            i_pol = np.sqrt(q ** 2 + u ** 2 + v ** 2)
            i_diff = i - i_pol
            if np.min(i_diff) < -np.spacing(np.max(i_diff)):
                # polarized intensity higher than total (det of 2x2 block < 0)
                return np.inf
            else:
                return 0

    def prox(self, x_s, lmbda=1):
        """Project matrix as Stokes params onto positive semidefinite cone."""
        return proj_psd_stokes(x_s, epsilon=self._epsilon)
