#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division
import numpy as np

from .func_classes import (L1Norm, L1L2Norm, L2Norm, L2NormSqHalf, 
                           L2BallInd, LInfBallInd, ZerosInd)
from .prox_algos import proxgrad, proxgradaccel, admm, admmlin, pdhg

__all__ = ['bpdn', 'dantzig', 'l1rls', 'srlasso', 'zcls']

def bpdn(A, Astar, b, eps, x0, **kwargs):
    """Solves the basis pursuit denoising problem using linearized ADMM.
    
    argmin_x ||x||_1
      s.t.   ||A(x) - b||_2 <= eps
    
    If the keyword argument 'axis' is given, the l1-norm is replaced by the 
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.
    
    Additional keyword arguments are passed to admmlin.
    
    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm()
    else:
        F = L1L2Norm(axis=axis)
    G = L2BallInd(radius=eps)
    
    return admmlin(F, G, A, Astar, b, x0, **kwargs)

def dantzig(A, Astar, b, delta, x0, **kwargs):
    """Solves the Dantzig selector problem using linearized ADMM.
    
    argmin_x ||x||_1
      s.t.   ||Astar(A(x) - b)||_inf <= delta
    
    If the keyword argument 'axis' is given, the l1-norm is replaced by the 
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.
    
    Additional keyword arguments are passed to admmlin.
    
    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm()
    else:
        F = L1L2Norm(axis=axis)
    G = LInfBallInd(radius=delta)
    
    # "A" and "Astar" in admmlin notation
    AsA = lambda x: Astar(A(x))
    # "b" in admmlin notation
    Asb = Astar(b)
    
    return admmlin(F, G, AsA, AsA, Asb, x0, **kwargs)

def l1rls_admmlin(A, Astar, b, lmbda, x0, **kwargs):
    """Solves the l1-regularized least squares problem using linearized ADMM.
    
    argmin_x 0.5*(||A(x) - b||_2)**2 + lmbda*||x||_1
    
    This problem is sometimes called the LASSO since it is equivalent
    to the original LASSO formulation
        argmin_x 0.5*(||A(x) - b||_2)**2
          s.t.   ||x||_1 <= tau
    for appropriate lmbda(tau).
    
    This function uses the linearized alternating direction method of 
    multipliers (LADMM).
    
    If the keyword argument 'axis' is given, the l1-norm is replaced by the 
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.
    
    Additional keyword arguments are passed to admmlin.
    
    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm(scale=lmbda)
    else:
        F = L1L2Norm(axis=axis, scale=lmbda)
    G = L2NormSqHalf()
    
    return admmlin(F, G, A, Astar, b, x0, **kwargs)

def l1rls_pdhg(A, Astar, b, lmbda, x0, **kwargs):
    """Solves the l1-regularized least squares problem using PDHG.
    
    argmin_x 0.5*(||A(x) - b||_2)**2 + lmbda*||x||_1
    
    This problem is sometimes called the LASSO since it is equivalent
    to the original LASSO formulation
        argmin_x 0.5*(||A(x) - b||_2)**2
          s.t.   ||x||_1 <= tau
    for appropriate lmbda(tau).
    
    This function uses the primal dual hybrid gradient (PDHG) method.
    
    If the keyword argument 'axis' is given, the l1-norm is replaced by the 
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.
    
    Additional keyword arguments are passed to pdhg.
    
    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm(scale=lmbda)
    else:
        F = L1L2Norm(axis=axis, scale=lmbda)
    G = L2NormSqHalf()
    
    return pdhg(F, G, A, Astar, b, x0, **kwargs)

def l1rls_proxgradaccel(A, Astar, b, lmbda, x0, **kwargs):
    """Solves the l1-regularized least squares problem using proxgradaccel.
    
    argmin_x 0.5*(||A(x) - b||_2)**2 + lmbda*||x||_1
    
    This problem is sometimes called the LASSO since it is equivalent
    to the original LASSO formulation
        argmin_x 0.5*(||A(x) - b||_2)**2
          s.t.   ||x||_1 <= tau
    for appropriate lmbda(tau).
    
    This function uses the accelerated proximal gradient method, which
    is equivalent to FISTA.
    
    If the keyword argument 'axis' is given, the l1-norm is replaced by the 
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.
    
    Additional keyword arguments are passed to proxgradaccel.
    
    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm(scale=lmbda)
    else:
        F = L1L2Norm(axis=axis, scale=lmbda)
    G = L2NormSqHalf()
    
    return proxgradaccel(F, G, A, Astar, b, x0, **kwargs)

def l1rls_proxgrad(A, Astar, b, lmbda, x0, **kwargs):
    """Solves the l1-regularized least squares problem using proxgrad.
    
    argmin_x 0.5*(||A(x) - b||_2)**2 + lmbda*||x||_1
    
    This problem is sometimes called the LASSO since it is equivalent
    to the original LASSO formulation
        argmin_x 0.5*(||A(x) - b||_2)**2
          s.t.   ||x||_1 <= tau
    for appropriate lmbda(tau).
    
    This function uses the proximal gradient method, which is equivalent
    to iterative soft thresholding (ISTA).
    
    If the keyword argument 'axis' is given, the l1-norm is replaced by the 
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.
    
    Additional keyword arguments are passed to proxgrad.
    
    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm(scale=lmbda)
    else:
        F = L1L2Norm(axis=axis, scale=lmbda)
    G = L2NormSqHalf()
    
    return proxgrad(F, G, A, Astar, b, x0, **kwargs)

l1rls = l1rls_proxgradaccel

#def lasso_admmlin(A, Astar, b, tau, x0, **kwargs):
    #"""Solves the LASSO problem using linearized ADMM.
    
    #argmin_x 0.5*(||A(x) - b||_2)**2
      #s.t.   ||x||_1 <= tau
    
    #This definition follows Tibshirani's original formulation.
    #The l1-regularized least squares problem
        #argmin_x 0.5*(||A(x) - b||_2)**2 + lmbda*||x||_1
    #is sometimes called the LASSO since they are equivalent for appropriate
    #selection of lmbda(tau).
    
    #Additional keyword arguments are passed to admmlin.
    
    #"""
    #F = L1BallInd(radius=tau)
    #G = L2NormSqHalf()
    
    #return admmlin(F, G, A, Astar, b, x0, **kwargs)

#def lasso_proxgradaccel(A, Astar, b, tau, x0, **kwargs):
    #"""Solves the LASSO problem using the accelerated proximal gradient method.
    
    #argmin_x 0.5*(||A(x) - b||_2)**2
      #s.t.   ||x||_1 <= tau
    
    #This definition follows Tibshirani's original formulation.
    #The l1-regularized least squares problem
        #argmin_x 0.5*(||A(x) - b||_2)**2 + lmbda*||x||_1
    #is sometimes called the LASSO since they are equivalent for appropriate
    #selection of lmbda(tau).
    
    #Additional keyword arguments are passed to proxgradaccel.
    
    #"""
    #F = L1BallInd(radius=tau)
    #G = L2NormSqHalf()
    
    #return proxgradaccel(F, G, A, Astar, b, x0, **kwargs)

#lasso = lasso_proxgradaccel

def srlasso(A, Astar, b, lmbda, x0, **kwargs):
    """Solves the square root LASSO problem (like L1RLS) using linearized ADMM.
    
    argmin_x ||A(x) - b||_2 + lmbda*||x||_1
    
    The square root LASSO may be preferred to l1-regularized least squares 
    since the selection of the parameter 'lmbda' need not depend on the noise
    level of the measurements 'b' (i.e. b = A(x) + n for some noise n). See 
    Belloni, Chernozhukov, and Wang in Biometrika (2011) for more details.
    
    If the keyword argument 'axis' is given, the l1-norm is replaced by the 
    combined l1- and l2-norm with the l2-norm taken over the specified axes:
        l1norm(l2norm(x, axis)).
    This can be used to solve the 'block' or 'group' sparsity problem.
    
    Additional keyword arguments are passed to proxgrad.
    
    """
    try:
        axis = kwargs.pop('axis')
    except KeyError:
        F = L1Norm(scale=lmbda)
    else:
        F = L1L2Norm(axis=axis, scale=lmbda)
    G = L2Norm()
    
    return admmlin(F, G, A, Astar, b, x0, **kwargs)

def zcls_admmlin(A, Astar, b, zeros, x0, **kwargs):
    """Solves the zero-constrained least squares problem using admmlin.
    
    argmin_x 0.5*(||A(x) - b||_2)**2
      s.t.   x[zeros] == 0
    
    Additional keyword arguments are passed to admmlin.
    
    """
    F = ZerosInd(z=zeros)
    G = L2NormSqHalf()
    
    return admmlin(F, G, A, Astar, b, x0, **kwargs)

def zcls_proxgradaccel(A, Astar, b, zeros, x0, **kwargs):
    """Solves the zero-constrained least squares problem using proxgradaccel.
    
    argmin_x 0.5*(||A(x) - b||_2)**2
      s.t.   x[zeros] == 0
    
    Additional keyword arguments are passed to proxgradaccel.
    
    """
    F = ZerosInd(z=zeros)
    G = L2NormSqHalf()
    
    return proxgradaccel(F, G, A, Astar, b, x0, **kwargs)

zcls = zcls_proxgradaccel