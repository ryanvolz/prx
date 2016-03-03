#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

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
