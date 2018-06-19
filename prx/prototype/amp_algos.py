# ----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import division

import numpy as np

from .thresholding import medestnoise, softthresh_d

__all__ = ('amp_fdr', 'amp_cfar')


def fdrthresh(x, m):
    return np.sort(np.abs(x), axis=None)[-m]


def amp_cfar(A, Astar, y, x0, lmbda, err=1e-3, maxits=10000, moreinfo=False):
    M = len(y)
    x = x0
    z = y - A(x)
    delta = np.inf
    sigma_est = medestnoise(Astar(z))

    for it in range(maxits):
        Asz = Astar(z)
        xprethresh = x + Asz

        # # estimate the noise level
        # sigma_med = medestnoise(Asz)
        # # need to smooth noise estimate in order to allow convergence
        # # (noise variance -> threshold variance -> x variance -> no convergence)
        # # difficulty is that the "noise" is not stationary, it changes
        # # as the x estimate gets better
        # # so want enough smoothing to converge, but not so much that the noise
        # # estimate significantly lags the true "noise"
        # # -> base amount of smoothing on delta between iterates
        # # the following weight heuristics all get the job done
        # # but the "gaussian weighting" acts a bit like continuation
        # # (setting threshold artificially high to speed convergence, then lowering)
        # # and so tends to converge faster than the others
        # #weight = np.exp(-np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        # #weight = 1/(1 + np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        # #weight = 2*stats.norm.sf(np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        # #weight = np.exp(-((sigma_med - sigma_est)/delta)**2)
        # #sigma_est = weight*sigma_est + (1 - weight)*sigma_med
        # sigma_est = sigma_med

        # thresh = lmbda*sigma_est
        thresh = 1.75
        # apply soft thresholding to get new sparse estimate
        xnew, w = softthresh_d(xprethresh, thresh)
        znew = y - A(xnew) + z/M*np.sum(w)

        delta = np.max(np.abs(xnew - x))
        if (it % 100) == 0:
            print('Iteration {0}, error={1}, delta={2}, thresh={3}'.format(it, np.linalg.norm(z), delta, thresh))
        if delta < err:
            break

        x = xnew
        z = znew

    if moreinfo:
        return xnew, it
    else:
        return xnew


def amp_fdr(A, Astar, y, x0, lmbda, err=1e-3, maxits=10000, moreinfo=False):
    M = len(y)
    x = x0
    z = y - A(x)

    for it in range(maxits):
        xprethresh = x + Astar(z)
        # find the mth largest entry of xprethresh to set threshold
        thresh = lmbda + fdrthresh(xprethresh, M)
        # apply soft thresholding to get new sparse estimate
        xnew, w = softthresh_d(xprethresh, thresh)
        znew = y - A(xnew) + z/M*np.sum(w)

        delta = np.max(np.abs(xnew - x))
        if (it % 100) == 0:
            print('Iteration {0}, error={1}, delta={2}'.format(it, np.linalg.norm(z), delta))
        if delta < err:
            break

        x = xnew
        z = znew

    if moreinfo:
        return xnew, it
    else:
        return xnew
