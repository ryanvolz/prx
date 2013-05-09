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
from scipy import stats
import bottleneck as bn

complex_variance_est_factor = stats.chi2(2).mean()/stats.chi2(2).ppf(0.5)
complex_std_est_factor = np.sqrt(complex_variance_est_factor)

def softthresh(x, theta):
    xmod = np.abs(x)
    xdir = np.exp(1j*np.angle(x))
    return np.maximum(xmod - theta, 0)*xdir

def softthreshp(x, theta):
    xdir = np.exp(1j*np.angle(x))
    return (np.abs(x) > theta)*xdir

def estnoise(x):
    return bn.median(np.abs(x))*complex_std_est_factor

def ist(A, Astar, y, x0, lmbda, relax=1, maxits=10000, moreinfo=False):
    x = x0
    z = y - A(x)

    for it in xrange(maxits):
        if (it % 100) == 0:
            print 'Iteration {0}, error={1}'.format(it, np.linalg.norm(z))

        Asz = Astar(z)
        # estimate the noise level
        sigma_est = relax*estnoise(Asz)
        # apply soft thresholding to get new sparse estimate
        xnew = softthresh(x + relax*Asz, lmbda*sigma_est)
        znew = y - A(xnew)
        if np.max(np.abs(xnew - x)) < 1e-6:
            break
        x = xnew
        z = znew

    if moreinfo:
        return xnew, it
    else:
        return xnew