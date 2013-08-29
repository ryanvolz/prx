# Copyright 2013 Ryan Volz

# This file is part of spyalg.

# Spyalg is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Spyalg is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with spyalg.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import numpy as np

from .thresholding import softthresh, softthreshp
from .ist_algos import medestnoise

__all__ = ['amp_fdr', 'amp_ffar']

def fdrthresh(x, m):
    return np.sort(np.abs(x), axis=None)[-m]

def amp_ffar(A, Astar, y, x0, lmbda, maxits=10000, moreinfo=False):
    M = len(y)
    x = x0
    z = y - A(x)

    for it in xrange(maxits):
        if (it % 100) == 0:
            print 'Iteration {0}, error={1}'.format(it, np.linalg.norm(z))

        Asz = Astar(z)
        xprethresh = x + Asz
        # estimate the noise level
        sigma_est = medestnoise(Asz)
        thresh = lmbda*sigma_est
        # apply soft thresholding to get new sparse estimate
        xnew = softthresh(xprethresh, thresh)
        znew = y - A(xnew) + z/M*np.sum(softthreshp(xprethresh, thresh))
        if np.max(np.abs(xnew - x)) < 1e-6:
            break
        x = xnew
        z = znew

    if moreinfo:
        return xnew, it
    else:
        return xnew

def amp_fdr(A, Astar, y, x0, lmbda, maxits=10000, moreinfo=False):
    M = len(y)
    x = x0
    z = y - A(x)

    for it in xrange(maxits):
        if (it % 100) == 0:
            print 'Iteration {0}, error={1}'.format(it, np.linalg.norm(z))


        xprethresh = x + Astar(z)
        # find the mth largest entry of xprethresh to set threshold
        thresh = lmbda + fdrthresh(xprethresh, M)
        # apply soft thresholding to get new sparse estimate
        xnew = softthresh(xprethresh, thresh)
        znew = y - A(xnew) + z/M*np.sum(softthreshp(xprethresh, thresh))
        if np.max(np.abs(xnew - x)) < 1e-6:
            break
        x = xnew
        z = znew

    if moreinfo:
        return xnew, it
    else:
        return xnew