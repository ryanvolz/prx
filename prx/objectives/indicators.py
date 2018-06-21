# ----------------------------------------------------------------------------
# Copyright (c) 2014-2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Objective classes for indicator functions.

.. currentmodule:: prx.objectives.indicators

General Indicators
------------------

.. autosummary::
    :toctree:

    NNegInd
    NPosInd
    ZerosInd
    PSDInd
    PSDIndStokes

Norm Ball Indicators
--------------------

.. autosummary::
    :toctree:

    L1BallInd
    L2BallInd
    LInfBallInd

"""

from __future__ import division

import numpy as np

from ..prox.projections import (proj_nneg, proj_npos, proj_psd,
                                proj_psd_stokes, proj_zeros)
from .norms import L1BallInd, L2BallInd, LInfBallInd
from .objective_classes import (IndicatorObjective, _class_docstring_wrapper,
                                _init_docstring_wrapper)

__all__ = (
    'NNegInd', 'NPosInd', 'ZerosInd',
    'PSDInd', 'PSDIndStokes',
    'L1BallInd', 'L2BallInd', 'LInfBallInd',
)


class NNegInd(IndicatorObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the non-negative indicator function.

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.
    NPosInd : Conjugate objective.
    .proj_nneg : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for vectors with only non-negative
    entries and infinity if any of the entries are negative.

    The prox operator is Euclidean projection onto the non-negative
    halfspace, i.e. the negative entries are set to zero.

    """
    )

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


class NPosInd(IndicatorObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the non-positive indicator function.

    {common_summary}


    Attributes
    ----------

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.
    NNegInd : Conjugate objective.
    .proj_npos : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for vectors with only non-positive
    entries and infinity if any of the entries are positive.

    The prox operator is Euclidean projection onto the non-positive
    halfspace, i.e. the positive entries are set to zero.

    """
    )

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


class ZerosInd(IndicatorObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the indicator of prescribed zero elements.

    {common_summary}


    Attributes
    ----------

    z : boolean array
        Array specifying the zero locations, requiring x[z] == 0.

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.
    .proj_zeros : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for vectors with only zeros in the
    specified places, infinity if any of the required zero entries are
    nonzero.

    The prox operator is Euclidean projection onto the set with
    specified zeros (x[z] is set to 0).

    """
    )

    @_init_docstring_wrapper
    def __init__(self, z, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """Create Objective that defines an indicator function.

        {common_summary}

        Since this objective is an indicator, `scale` can be eliminated::

            s*f(a*x + b) => f(a*x + b)


        Parameters
        ----------

        z : boolean array
            Array specifying the zero locations, requiring x[z] == 0.

        {common_params}

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


class PSDInd(IndicatorObjective):
    __doc__ = _class_docstring_wrapper(
        """Objective class for the positive semidefinite indicator function.

    {common_summary}


    Attributes
    ----------

    epsilon : float
        When `epsilon` is non-zero, the indicator is for the cone given by
        :math:`X \succeq \epsilon I` instead of the PSD cone. The prox
        operator sets eigenvalues less than `epsilon` to `epsilon`.

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.
    PSDIndStokes : Specialized PSD indicator for matrices as Stokes parameters.
    .proj_psd : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for matrices M that are positive
    semidefinite (all eigenvalues are >= 0) and infinity otherwise.

    The prox operator is Euclidean projection to the nearest positive
    semidefinite matrix (negative eigenvalues are set to zero).

    """
    )

    @_init_docstring_wrapper
    def __init__(self, epsilon=0, scale=None, stretch=None, shift=None,
                 linear=None, const=None):
        """Create Objective that defines an indicator function.

        {common_summary}

        Since this objective is an indicator, `scale` can be eliminated::

            s*f(a*x + b) => f(a*x + b)


        Parameters
        ----------

        epsilon : float, optional
            When `epsilon` is non-zero, the indicator is for the cone given by
            :math:`X \succeq \epsilon I` instead of the PSD cone. The prox
            operator sets eigenvalues less than `epsilon` to `epsilon`.

        {common_params}

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
    __doc__ = _class_docstring_wrapper(
        """Objective class for the positive semidefinite indicator function.

    This indicator handles the special case of a block 2x2 Hermitian matrix
    represented by Stokes parameters where the projection can be achieved
    efficiently without an eigen-decomposition.

    {common_summary}


    Attributes
    ----------

    epsilon : float
        When `epsilon` is non-zero, the indicator is for the cone given by
        :math:`X \succeq \epsilon I` instead of the PSD cone. The prox
        operator sets eigenvalues less than `epsilon` to `epsilon`.

    {common_attributes}


    See Also
    --------

    .IndicatorObjective : Parent class.
    PSDInd : General PSD indicator.
    .proj_psd_stokes : Prox operator projection function.


    Notes
    -----

    The indicator function is zero for matrices M that are positive
    semidefinite (all eigenvalues are >= 0) and infinity otherwise.

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
    )

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
