#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np
from scipy import stats
import bottleneck as bn

__all__ = ['block_softthresh', 'block_softthresh_d', 'medestnoise', 'softthresh', 'softthresh_d']

complex_variance_est_factor = stats.chi2(2).mean()/stats.chi2(2).ppf(0.5)
complex_std_est_factor = np.sqrt(complex_variance_est_factor)

def medestnoise(x):
    return bn.median(np.abs(x))*complex_std_est_factor

def _block_softthresh(x, theta, axis=-1):
    xsq = x.real**2 + x.imag**2
    blknorm = np.sqrt(xsq.sum(axis=axis, keepdims=True))
    thresh_mult = np.maximum(1 - theta/blknorm, 0)
    xthresh = thresh_mult*x
    return xthresh, blknorm

def block_softthresh(x, theta, axis=-1):
    xthresh, blknorm = _block_softthresh(x, theta, axis=axis)
    return xthresh

def block_softthresh_d(x, theta, axis=-1):
    xthresh, blknorm = _block_softthresh(x, theta, axis=axis)
    xthreshderiv = np.where(blknorm > theta, x/blknorm, 0)
    return xthresh, xthreshderiv

def _softthresh(x, theta):
    xmod = np.abs(x)
    thresh_mult = np.maximum(1 - theta/xmod, 0)
    xthresh = thresh_mult*x
    return xthresh, xmod

def softthresh(x, theta):
    xthresh, xmod = _softthresh(x, theta)
    return xthresh

def softthresh_d(x, theta):
    xthresh, xmod = _softthresh(x, theta)
    xthreshderiv = np.where(xmod > theta, x/xmod, 0)
    return xthresh, xthreshderiv
