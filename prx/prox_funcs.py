# ----------------------------------------------------------------------------
# Copyright (c) 2015, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import division
import numpy as np

from .norms import l2norm

__all__ = ['proj_l1', 'proj_l2', 'proj_linf', 'proj_zeros',
           'prox_l1', 'prox_l2', 'prox_l2sqhalf', 'prox_linf']

###***************************************************************************
### Explicit prox and projection functions ***********************************
###***************************************************************************

def proj_l1(x, radius=1):
    """Project x onto l1-ball with specified radius."""
    xmod = np.abs(x)
    nz = xmod[xmod.nonzero()]
    s = np.sort(nz, axis=None)[::-1] # sorted in descending order
    cs = s.cumsum()

    if cs[-1] <= radius:
        return x

    ks = 1 + np.arange(len(s))
    idx = np.searchsorted(s*ks <= cs - radius, True) - 1

    # soft thresholding threshold that will result in l1-norm of radius
    lmbda = (cs[idx] - radius) / (1 + idx)
    # now perform the soft thresholding to complete the projection
    thresh_mult = np.maximum(1 - lmbda/xmod, 0)
    return thresh_mult*x

def proj_l2(x, radius=1):
    """Project x onto l2-ball with specified radius."""
    l2nrm = l2norm(x)
    projmult = radius/l2nrm
    if projmult < 1:
        return projmult*x
    else:
        return x

def proj_linf(x, radius=1):
    """Project x onto linf-ball with specified radius."""
    xmod = np.abs(x)
    projmult = np.minimum(radius/xmod, 1)
    return projmult*x

def proj_zeros(x, z):
    """Project x onto set with specified zeros z (set x[z] = 0)."""
    x[z] = 0
    return x

def prox_l1(x, lmbda=1):
    """l1-norm prox operator (soft thresholding).

    st(x[k], lmbda) = { (1 - lmbda/|x[k]|)*x[k]  if |x[k]| > lmbda
                      {           0              otherwise

    """
    xmod = np.abs(x)
    thresh_mult = np.maximum(1 - lmbda/xmod, 0)
    return thresh_mult*x

def prox_l1l2(x, lmbda=1, axis=-1):
    """Combined l1- and l2-norm prox operator (block soft thresholding).

    The axis argument defines the axes over which to take the l2-norm
    (axis=None specifies all axes and is equivalent to prox_l2).

    For index 'k' NOT along the specified axis:

    bst(x[k, :], lmbda) = { (1 - lmbda/||x[k]||_2)*x[k]  if ||x[k]||_2 > lmbda
                          {           0                  otherwise

    """
    blknorm = l2norm(x, axis=axis, keepdims=True)
    thresh_mult = np.maximum(1 - lmbda/blknorm, 0)
    return thresh_mult*x

def prox_l2(x, lmbda=1):
    """l2-norm prox operator (block soft thresholding with block of entire x).

    bst(x, lmbda) = { (1 - lmbda/||x||_2)*x  if ||x||_2 > lmbda
                    {           0            otherwise

    """
    l2nrm = l2norm(x)
    thresh_mult = np.maximum(1 - lmbda/l2nrm, 0)
    return thresh_mult*x

def prox_l2sqhalf(x, lmbda=1):
    """Prox operator for half the squared l2-norm: 0.5*(||x||_2)**2.

    The prox operator is the shrinkage function:

    shrink(x, lmbda) = x/(1 + lmbda)

    """
    return x/(1 + lmbda)

def prox_linf(x, lmbda=1):
    """linf-norm prox operator (peak shrinkage).

    The prox operator is the peak shrinkage function, which minimizes the
    maximum value for a reduction of lmbda in the l1-norm.

    """
    xmod = np.abs(x)
    nz = xmod[xmod.nonzero()]
    s = np.sort(nz, axis=None)[::-1] # sorted in descending order
    cs = s.cumsum()

    if cs[-1] <= lmbda:
        return np.zeros_like(x)

    ks = 1 + np.arange(len(s))
    idx = np.searchsorted(s*ks <= cs - lmbda, True) - 1

    # peak clipping threshold that will reduce l1-norm of x by lmbda
    peak = (cs[idx] - lmbda) / (1 + idx)
    # now clip all values to the peak
    clip_mult = np.minimum(peak/xmod, 1)
    return clip_mult*x
