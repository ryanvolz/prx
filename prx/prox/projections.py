# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2014, 2016, 2018 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Projection operators, prox operators of indicator functions."""

from __future__ import division

import numpy as np

from ..fun.norms import l2norm

__all__ = (
    'proj_l1', 'proj_l2', 'proj_linf', 'proj_nneg', 'proj_npos', 'proj_psd',
    'proj_psd_stokes', 'proj_zeros',
)


def proj_l1(v, radius=1):
    """Project v onto l1-ball with specified radius.

    Given a radius, this prox operator solves for x in::

        minimize    l2normsqhalf(x - v)
        subject to  l1norm(x) <= radius

    which is Euclidean projection onto the l1-ball.


    Parameters
    ----------

    v : array
        Input array.

    radius : float, optional
        Ball radius.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    proj_l2, proj_linf, .l1norm

    """
    vmod = np.abs(v)
    nz = vmod[vmod.nonzero()]
    s = np.sort(nz, axis=None)[::-1]  # sorted in descending order
    cs = s.cumsum()

    if cs[-1] <= radius:
        return v

    ks = 1 + np.arange(len(s))
    idx = np.searchsorted(s*ks <= cs - radius, True) - 1

    # soft thresholding threshold that will result in l1-norm of radius
    lmbda = (cs[idx] - radius) / (1 + idx)
    # now perform the soft thresholding to complete the projection
    thresh_mult = np.maximum(1 - lmbda/vmod, 0)
    return thresh_mult*v


def proj_l2(v, radius=1):
    """Project v onto l2-ball with specified radius.

    Given a radius, this prox operator solves for x in::

        minimize    l2normsqhalf(x - v)
        subject to  l2norm(x) <= radius

    which is Euclidean projection onto the l2-ball.


    Parameters
    ----------

    v : array
        Input array.

    radius : float, optional
        Ball radius.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    proj_l1, proj_linf, .l2norm

    """
    l2nrm = l2norm(v)
    projmult = radius/l2nrm
    if projmult < 1:
        return projmult*v
    else:
        return v


def proj_linf(v, radius=1):
    """Project v onto linf-ball with specified radius.

    Given a radius, this prox operator solves for x in::

        minimize    l2normsqhalf(x - v)
        subject to  linfnorm(x) <= radius

    which is Euclidean projection onto the linf-ball.


    Parameters
    ----------

    v : array
        Input array.

    radius : float, optional
        Ball radius.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    proj_l1, proj_l2, .linfnorm

    """
    vmod = np.abs(v)
    projmult = np.minimum(radius/vmod, 1)
    return projmult*v


def proj_nneg(v):
    """Project v onto the non-negative reals (negatives set to zero).

    This prox operator solves for x in::

        minimize    l2normsqhalf(x - v)
        subject to  x >= 0

    which is Euclidean projection onto the non-negative half-space.


    Parameters
    ----------

    v : array
        Input array.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    proj_npos, proj_zeros

    """
    return np.maximum(v, 0)


def proj_npos(v):
    """Project v onto the non-positive reals (positives set to zero).

    This prox operator solves for x in::

        minimize    l2normsqhalf(x - v)
        subject to  x <= 0

    which is Euclidean projection onto the non-positive half-space.


    Parameters
    ----------

    v : array
        Input array.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    proj_nneg, proj_zeros

    """
    return np.minimum(v, 0)


def proj_psd(V, epsilon=0):
    ur"""Project Hermitian matrix onto positive semidefinite cone.

    In effect, this sets negative eigenvalues set to 0.

    This prox operator solves for X in::

        minimize    l2normsqhalf(X - V, axis=(-2, -1))
        subject to  X ≽ 0 (in matrix inequality sense)

    which is Euclidean projection onto the positive semidefinite cone.


    Parameters
    ----------

    V : (..., M, M) array_like
        Hermitian matrix or matrices to project.

    epsilon : float, optional
        When `epsilon` is non-zero, project onto the cone given by
        ``X ≽ epsilon * I`` instead of the PSD cone. This sets eigenvalues less
        than `epsilon` to `epsilon`.


    Returns
    -------

    X : (..., M, M) array_like
        Nearest positive semidefinite matrix.


    See Also
    --------

    proj_psd_stokes

    """
    w, v = np.linalg.eigh(V)
    wnneg = np.maximum(w, epsilon)
    # reforming projected V from square roots guarantees that it is exactly
    # Hermitian
    Xhalf = v*np.sqrt(wnneg)[..., np.newaxis, :]
    X = np.matmul(Xhalf, Xhalf.conj().swapaxes(-2, -1))
    return X


def proj_psd_stokes(v_s, epsilon=0):
    ur"""Project Hermitian matrix onto positive semidefinite cone.

    In effect, this sets negative eigenvalues set to 0. This function handles
    the special case of a block 2x2 Hermitian matrix represented by Stokes
    parameters where the projection can be achieved efficiently without an
    eigen-decomposition.

    This prox operator solves for X in::

        minimize    l2normsqhalf(X - V, axis=(-2, -1))
        subject to  X ≽ 0

    which is Euclidean projection onto the positive semidefinite cone where V
    is the equivalent matrix corresponding to the Stokes parameters `v_s`.


    Parameters
    ----------

    v_s : (..., 4) array_like of floats
        2x2 Hermitian matrix block or blocks represented by an array of the 4
        Stokes parameters (see notes).

    epsilon : float, optional
        When `epsilon` is non-zero, project onto the cone given by
        ``X ≽ epsilon * I`` instead of the PSD cone. This sets eigenvalues less
        than `epsilon` to `epsilon`.


    Returns
    -------

    x_s : (..., 4) array_like of floats
        Nearest positive semidefinite matrix, represented by Stokes parameters.


    See Also
    --------

    proj_psd


    Notes
    -----

    This projection function operates on 2x2 Hermitian blocks given in terms of
    Stokes parameters (I, Q, U, V)::

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
    x_s = v_s.copy()
    i, q, u, v = [x_s[..., k] for k in range(4)]

    # polarized intensity
    i_pol = np.sqrt(q ** 2 + u ** 2 + v ** 2)
    # entries that need scaling, includes i < 0 since i_pol is always >= 0
    scale = i_pol > i
    # projection of intensity is average between total and polarized
    i_psd = (i[scale] + i_pol[scale]) / 2
    # but always >= epsilon
    i_psd = np.maximum(i_psd, epsilon)

    # knowing amount polarization is scaled, scale q, u, v the same
    pol_scale = i_psd / i_pol[scale]
    # make zero in case where we have 0/0
    pol_scale[np.isnan(pol_scale)] = 0

    i[scale] = i_psd
    q[scale] = q[scale] * pol_scale
    u[scale] = u[scale] * pol_scale
    v[scale] = v[scale] * pol_scale

    # since i, q, u, v are views into x_s, we can simply return x_s after
    # having written the modified values to i, q, u, v
    return x_s


def proj_zeros(v, z):
    """Project v onto set with specified zeros z (set v[z] = 0).

    This prox operator solves for x in::

        minimize    l2normsqhalf(x - v)
        subject to  x[z] == 0

    which is Euclidean projection onto the space where x[z] == 0.


    Parameters
    ----------

    v : array
        Input array.

    z : boolean array
        Array specifying the zero locations at True entries.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    proj_nneg, proj_npos

    """
    x = v.copy()
    x[z] = 0
    return x
