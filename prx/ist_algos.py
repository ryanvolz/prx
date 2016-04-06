#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np

from .standard_funcs import L1Norm, L2NormSqHalf
from .norms import linfnorm
from .prox_algos import proxgrad, proxgradaccel
from .thresholding import softthresh

__all__ = ['fista', 'fista_cfar', 
           'ist', 'ist_cfar']

def ist(A, Astar, b, lmbda, x0, **kwargs):
    F = L1Norm(scale=lmbda)
    G = L2NormSqHalf()
    
    return proxgrad(F, G, A, Astar, b, x0, **kwargs)

def fista(A, Astar, b, lmbda, x0, **kwargs):
    F = L1Norm(scale=lmbda)
    G = L2NormSqHalf()
    
    return proxgradaccel(F, G, A, Astar, b, x0, **kwargs)

# iterative soft thresholding with threshold set by constant false alarm rate
# see Maleki and Donoho, 2010
def ist_cfar(A, Astar, y, x0, sigmamult=3, relax=1, reltol=1e-4, abstol=1e-6,
             maxits=10000, threshfun=softthresh, moreinfo=False, printrate=100):
    x = x0
    delta = np.inf
    sigma_est = medestnoise(Astar(z))
    
    tolnorm = linfnorm
    rabstol = abstol*tolnorm(np.ones_like(x0))

    for k in xrange(maxits):
        z = y - A(x)
        Asz = Astar(z)
        
        # estimate the noise level
        sigma_med = medestnoise(Asz)
        # need to smooth noise estimate in order to allow convergence
        # (noise variance -> threshold variance -> x variance -> no convergence)
        # difficulty is that the "noise" is not stationary, it changes
        # as the x estimate gets better
        # so want enough smoothing to converge, but not so much that the noise
        # estimate significantly lags the true "noise"
        # -> base amount of smoothing on delta between iterates
        # the following weight heuristics all get the job done
        # but the "gaussian weighting" acts a bit like continuation
        # (setting threshold artificially high to speed convergence, then lowering)
        # and so tends to converge faster than the others
        #weight = np.exp(-np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        #weight = 1/(1 + np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        #weight = 2*stats.norm.sf(np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        weight = np.exp(-((sigma_med - sigma_est)/delta)**2)
        sigma_est = weight*sigma_est + (1 - weight)*sigma_med
        
        # apply soft thresholding to get new sparse estimate
        thresh = sigmamult*sigma_est
        xnew = threshfun(x + relax*Asz, thresh)
        
        delta = tolnorm(xnew - x)
        stopthresh = rabstol + reltol*tolnorm(xnew)
        
        if printrate is not None and (k % printrate) == 0:
            print('Iteration {0}, delta={1} ({2:0.3}), thresh={3}'.format(k, delta, stopthresh, thresh))
        if delta < stopthresh:
            break
        
        x = xnew

    if moreinfo:
        return dict(x=x, numits=k, delta=delta, sigma_est=sigma_est)
    else:
        return x

# doesn't converge for low threshold levels because noise estimate bounces around too much
# smoothing added makes it converge like fixed threshold fista, but what formula for smoothing?
def fista_cfar(A, Astar, y, x0, sigmamult=3, relax=1, reltol=1e-4, abstol=1e-6,
               maxits=10000, threshfun=softthresh, moreinfo=False, printrate=100):
    x = x0
    xold = x0
    delta = np.inf
    sigma_est = medestnoise(Astar(z))
    
    tolnorm = linfnorm
    rabstol = abstol*tolnorm(np.ones_like(x0))
    
    for k in xrange(maxits):
        # FISTA secret sauce
        w = x + (k - 1)/(k + 2)*(x - xold)
        
        z = y - A(w)
        Asz = Astar(z)
        
        # estimate the noise level
        sigma_med = medestnoise(Asz)
        # need to smooth noise estimate in order to allow convergence
        # (noise variance -> threshold variance -> x variance -> no convergence)
        # difficulty is that the "noise" is not stationary, it changes
        # as the x estimate gets better
        # so want enough smoothing to converge, but not so much that the noise
        # estimate significantly lags the true "noise"
        # -> base amount of smoothing on delta between iterates
        # the following weight heuristics all get the job done
        # but the "gaussian weighting" acts a bit like continuation
        # (setting threshold artificially high to speed convergence, then lowering)
        # and so tends to converge faster than the others
        #weight = np.exp(-np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        #weight = 1/(1 + np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        #weight = 2*stats.norm.sf(np.abs(sigma_med - sigma_est)/np.sqrt(delta))
        weight = np.exp(-((sigma_med - sigma_est)/delta)**2)
        sigma_est = weight*sigma_est + (1 - weight)*sigma_med
        
        # apply soft thresholding to get new iterate
        thresh = sigmamult*sigma_est
        xnew = threshfun(w + relax*Asz, thresh)
        
        delta = tolnorm(xnew - x)
        stopthresh = rabstol + reltol*tolnorm(xnew)
        
        if printrate is not None and (k % printrate) == 0:
            print('Iteration {0}, delta={1} ({2:0.3}), thresh={3}'.format(k, delta, stopthresh, thresh))
        if delta < stopthresh:
            break
        
        xold = x
        x = xnew
    
    if moreinfo:
        return dict(x=x, numits=k, w=w, delta=delta, sigma_est=sigma_est)
    else:
        return x
