# ----------------------------------------------------------------------------
# Copyright (c) 2014-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Prox operators of norm functions."""

from __future__ import division

import numpy as np

from ..fun.norms import l2norm

__all__ = ('prox_l1', 'prox_l1l2', 'prox_l2', 'prox_l2sqhalf', 'prox_linf')


def prox_l1(v, lmbda=1):
    """l1-norm prox operator (soft thresholding).

    Given inputs v and lmbda, this prox operator solves for x in::

        minimize    l1norm(x) + l2normsqhalf(x - v)/lmbda

    In closed form, the operator is known as soft thresholding::

        st(v[k], lmbda) = { v[k] - lmbda*v[k]/abs(v[k])  if abs(v[k]) > lmbda
                          {           0                  otherwise


    Elements with magnitude less than `lmbda` are set to zero, and all other
    elements are shrunk in magnitude toward zero by `lmbda`.


    Parameters
    ----------

    v : array
        Input array.

    lmbda : float, optional
        Threshold.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    prox_l1l2, .l1norm

    """
    vmod = np.abs(v)
    with np.errstate(divide='ignore'):
        thresh_mult = np.maximum(1 - lmbda/vmod, 0)
    return thresh_mult*v


def prox_l1l2(v, lmbda=1, axis=-1):
    """Combined l1- and l2-norm prox operator (block soft thresholding).

    Given inputs v and lmbda, this prox operator solves for x in::

        minimize    l1l2norm(x) + l2normsqhalf(x - v)/lmbda

    In closed form, the operator is known as block soft thresholding::

        bst(v[k, :], lmbda) =
         { v[k, :] - lmbda*v[k, :]/l2norm(v[k, :])  if l2norm(v[k, :]) > lmbda
         {                   0                      otherwise


    Blocks with l2-norm less than `lmbda` are set to zero, and all other
    blocks are shrunk in l2-norm toward zero by `lmbda`.


    Parameters
    ----------

    v : array
        Input array.

    lmbda : float, optional
        Threshold.

    axis : int | tuple of ints, optional
        Axis or axes over which to calculate the l2-norm. The prox operator is
        applied over all remaining axes.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    prox_l1, .l1l2norm

    """
    blknorm = l2norm(v, axis=axis, keepdims=True)
    with np.errstate(divide='ignore'):
        thresh_mult = np.maximum(1 - lmbda/blknorm, 0)
    return thresh_mult*v


def prox_l2(v, lmbda=1):
    """l2-norm prox operator (block soft thresholding with block of entire v).

    Given inputs v and lmbda, this prox operator solves for x in::

        minimize    l2norm(x) + l2normsqhalf(x - v)/lmbda

    In closed form, the operator is known as block soft thresholding::

        bst(v, lmbda) = { v - lmbda*v/l2norm(v)  if l2norm(v) > lmbda
                        {          0             otherwise


    The l2-norm of the array is shrunk in magnitude toward zero by `lmbda`,
    and it is set entirely to zero if the l2-norm is less than `lmbda`.


    Parameters
    ----------

    v : array
        Input array.

    lmbda : float, optional
        Threshold.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    prox_l1l2, .l2norm

    """
    l2nrm = l2norm(v)
    with np.errstate(divide='ignore'):
        thresh_mult = np.maximum(1 - lmbda/l2nrm, 0)
    return thresh_mult*v


def prox_l2sqhalf(v, lmbda=1):
    """Prox operator for half the squared l2-norm (shrinkage function).

    Given inputs v and lmbda, this prox operator solves for x in::

        minimize    l2normsqhalf(x) + l2normsqhalf(x - v)/lmbda

    In closed form, the operator is known as the shrinkage function::

        shrink(v, lmbda) = v/(1 + lmbda)


    Parameters
    ----------

    v : array
        Input array.

    lmbda : float, optional
        Shrinkage factor.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    prox_l2, .l2normsqhalf

    """
    return v/(1 + lmbda)


def prox_linf(v, lmbda=1):
    """linf-norm prox operator (peak shrinkage).

    Given inputs v and lmbda, this prox operator solves for x in::

        minimize    linfnorm(x) + l2normsqhalf(x - v)/lmbda

    This is known as the shrinkage function, which minimizes the maximum value
    necessary to achieve a reduction of `lmbda` in the l1-norm.


    Parameters
    ----------

    v : array
        Input array.

    lmbda : float, optional
        Shrinkage factor.


    Returns
    -------

    x : array
        Result.


    See Also
    --------

    prox_l1, .linfnorm

    """
    vmod = np.abs(v)
    nz = vmod[vmod.nonzero()]
    s = np.sort(nz, axis=None)[::-1]  # sorted in descending order
    cs = s.cumsum()

    if cs[-1] <= lmbda:
        return np.zeros_like(v)

    ks = 1 + np.arange(len(s))
    idx = np.searchsorted(s*ks <= cs - lmbda, True) - 1

    # peak clipping threshold that will reduce l1-norm of v by lmbda
    peak = (cs[idx] - lmbda) / (1 + idx)
    # now clip all values to the peak
    with np.errstate(divide='ignore'):
        clip_mult = np.minimum(peak/vmod, 1)
    return clip_mult*v
