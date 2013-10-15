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

import numpy as np

__all__ = ['l1norm', 'l2norm', 'l2normsqhalf', 'linfnorm']

def l1norm(x, axis=None, keepdims=False):
    """l1-norm: sum(abs(x))
    
    If 'axis' argument is specified, take norm over only the indicated axes. 
    Otherwise, take norm over all axes.
    
    """
    return np.abs(x).sum(axis=axis, keepdims=keepdims)

def l1l2norm(x, axis=-1):
    """Combined l1- and l2-norm: l1norm(l2norm(x, axis))
    
    If 'axis' argument is specified, take l2-norm over the indicated axes. 
    Otherwise, take l2-norm over the last axis.
    
    The l1-norm is applied over all remaining axes.
    
    """
    return np.sqrt((x.real**2 + x.imag**2).sum(axis=axis)).sum()

def l2norm(x, axis=None, keepdims=False):
    """l2-norm: sqrt(sum(abs(x)**2))
    
    If 'axis' argument is specified, take norm over only the indicated axes. 
    Otherwise, take norm over all axes.
    
    """
    return np.sqrt((x.real**2 + x.imag**2).sum(axis=axis, keepdims=keepdims))

def l2normsqhalf(x, axis=None, keepdims=False):
    """Half the square of the l2-norm: 0.5*sum(abs(x)**2)
    
    If 'axis' argument is specified, take norm over only the indicated axes. 
    Otherwise, take norm over all axes.
    
    """
    return 0.5*(x.real**2 + x.imag**2).sum(axis=axis, keepdims=False)

def linfnorm(x, axis=None, keepdims=False):
    """linf-norm: max(abs(x))
    
    If 'axis' argument is specified, take norm over only the indicated axes. 
    Otherwise, take norm over all axes.
    
    """
    return np.abs(x).max(axis=axis, keepdims=False)