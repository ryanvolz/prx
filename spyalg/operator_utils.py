#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from .norms import l2norm

__all__ = ['adjointness_error', 'opnorm']

def get_random_normal(shape, dtype):
    x = np.empty(shape, dtype)
    x.real = np.random.randn(*shape)
    try:
        x.imag = np.random.randn(*shape)
    except TypeError:
        pass
    return x

def adjointness_error(A, Astar, inshape, indtype, its=100):
    """Check adjointness of A and Astar for 'its' instances of random data.
    
    For random unit-normed x and y, this finds the error in the adjoint 
    identity <Ax, y> == <x, A*y>:
        err = abs( vdot(A(x), y) - vdot(x, Astar(y)) ).
    
    The type and shape of the input to A are specified by inshape and indtype.
    
    Returns a vector of the error magnitudes.
    
    """
    x = get_random_normal(inshape, indtype)
    y = A(x)
    outshape = y.shape
    outdtype = y.dtype
    
    errs = np.zeros(its)
    for k in xrange(its):
        x = get_random_normal(inshape, indtype)
        x = x/l2norm(x)
        y = get_random_normal(outshape, outdtype)
        y = y/l2norm(y)
        ip_A = np.vdot(A(x), y)
        ip_Astar = np.vdot(x, Astar(y))
        errs[k] = np.abs(ip_A - ip_Astar)
    
    return errs

def opnorm(A, Astar, inshape, indtype, reltol=1e-8, abstol=1e-6, maxits=100, printrate=None):
    """Estimate the l2-induced operator norm: sup_v ||A(v)||/||v|| for v != 0.
    
    Uses the power iteration method to estimate the operator norm of
    A and Astar.
    
    The type and shape of the input to A are specified by inshape and indtype.
    
    Returns a tuple: (norm of A, norm of Astar, vector inducing maximum scaling).
    """
    v0 = get_random_normal(inshape, indtype)
    v = v0/l2norm(v0)
    norm_f0 = 1
    norm_a0 = 1
    for k in xrange(maxits):
        Av = A(v)
        norm_f = l2norm(Av)
        w = Av/norm_f
        Asw = Astar(w)
        norm_a = l2norm(Asw)
        v = Asw/norm_a
        
        delta_f = abs(norm_f - norm_f0)
        delta_a = abs(norm_a - norm_a0)
        
        if printrate is not None and (k % printrate) == 0:
            print('Iteration {0}, forward norm: {1}, adjoint norm: {2}'.format(k, norm_f, norm_a))
        if (delta_f < abstol + reltol*max(norm_f, norm_f0)
            and delta_a < abstol + reltol*max(norm_a, norm_a0)):
            break
        
        norm_f0 = norm_f
        norm_a0 = norm_a
    
    return norm_f, norm_a, v