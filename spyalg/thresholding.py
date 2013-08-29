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

__all__ = ['softthresh', 'softthreshp']

def softthresh(x, theta):
    xmod = np.abs(x)
    xdir = np.exp(1j*np.angle(x))
    return np.maximum(xmod - theta, 0)*xdir

def softthreshp(x, theta):
    xdir = np.exp(1j*np.angle(x))
    return (np.abs(x) > theta)*xdir