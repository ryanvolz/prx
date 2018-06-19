# ----------------------------------------------------------------------------
# Copyright (c) 2014, 2016 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Projection operators, prox operators of indicator functions.

.. currentmodule:: prx.prox.projections

.. autosummary::
    :toctree:

    proj_l1
    proj_l2
    proj_linf
    proj_nneg
    proj_npos
    proj_zeros

"""

from __future__ import division

import numpy as np

from ..fun.norms import l2norm

__all__ = ['proj_l1', 'proj_l2', 'proj_linf',
           'proj_nneg', 'proj_npos', 'proj_zeros', 'proj_psd_stokes']

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
    s = np.sort(nz, axis=None)[::-1] # sorted in descending order
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
    v[z] = 0
    return v


def proj_psd_stokes(x, epsilon=0):
    # TODO, docstring
    i, q, u, v = x

    i_pol = np.sqrt(q ** 2 + u ** 2 + v ** 2)
    scale = i_pol > i
    i_psd = (i[scale] + i_pol[scale]) / 2
    i_psd = np.maximum(i_psd, epsilon)
    pol_scale = i_psd / i_pol[scale]
    pol_scale[np.isnan(pol_scale)] = 0
    i[scale] = i_psd
    q[scale] = q[scale] * pol_scale
    u[scale] = u[scale] * pol_scale
    v[scale] = v[scale] * pol_scale

    # Note here x is modified in the above operations as i q u v are just views into x and affect the same arrays
    return x
