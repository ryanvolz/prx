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

from .norms import l1norm, l2norm, l2normsqhalf, linfnorm

__all__ = ['admm', 'admmlin', 'proxgrad', 'proxgradaccel']

def proxgrad(F, G, A, Astar, b, x0, stepsize=1.0, backtrack=0.5, expand=1.25, 
             reltol=1e-6, abstol=1e-10, maxits=10000, 
             moreinfo=False, printrate=100, xstar=None):
    """Solve: argmin_x ( F(x) + G(A(x) - b) ) using the proximal gradient method.
    
    F and G are callables that return the function value at x or z=A(x)-b.
    
    F(x) must be "proximable", so that the callable F.prox(v, s)
    solves argmin_x ( F(x) + (1/(2s))(||x - v||_2)^2 ).
    
    G(z) must be smooth with gradient function callable as G.grad(z). As a
    function of x, the gradient of the G term is then
        gradGx(x) = Astar(G.grad(A(x) - b)).
    
    A is linear operator, and both the forward (A) and adjoint (Astar)
    operations must be specified.
    
    b is a specified constant.
    
    *********
    Algorithm
    *********
    Parameters:
        stepsize in [0, 2/L]: initial step size when backtracking is used or
            constant step size for all iterations (where L is the Lipschitz 
            constant of G.grad)
        backtrack in [0, 1] or None, amount to decrease step size when local
            Lipschitz condition is violated
        expand >= 1 or None, amount to increase step size after every iteration 
            so that it can adapt to decreasing local Lipschitz constant
    
    Update equation:
        grad_new = Astar(G.grad(A(x) - b))
        x_new = F.prox(x - stepsize*grad_new), stepsize)
    
    Residual (of 0 in subgradF(x_new) + gradGx(x_new)):
        r = (x - x_new)/stepsize + (grad - grad_new)
    
    *********
    Citations
    *********
    The basic algorithm is described in section 4.2 of N. Parikh and S. Boyd, 
    "Proximal Algorithms," Foundations and Trends in Optimization, vol. 1, 
    no. 3, pp. 123-231, 2013.
    
    Expanding backtracking line search is due to K. Scheinberg, D. Goldfarb, 
    and X. Bai, "Fast First-Order Methods for Composite Convex Optimization 
    with Backtracking," pre-print, Apr. 2011 and also S. R. Becker, 
    E. J. Candes, and M. C. Grant, "Templates for convex cone problems with 
    applications to sparse signal recovery," Mathematical Programming 
    Computation, vol. 3, no. 3, pp. 165-218, Aug. 2011.
    
    """
    proxF = F.prox
    gradG = G.grad
    
    stepsize = float(stepsize)
    
    if moreinfo:
        histdtype = [('it', np.int32), ('val', np.float64), 
                     ('step', np.float64), ('resid', np.float64), 
                     ('thresh', np.float64), ('err', np.float64)]
        hist = np.zeros((maxits - 1)//printrate + 1, dtype=histdtype)
    
    x = x0
    Axmb = A(x) - b
    GAxmb = G(Axmb)
    grad = Astar(gradG(Axmb))
    
    tolnorm = l2norm
    rabstol = abstol*tolnorm(np.ones_like(x0))
    
    bts = 0
    
    for k in xrange(maxits):
        # gradient direction
        grad_new = Astar(gradG(Axmb))
        
        # loop for backtracking line search
        while True:
            # proximal gradient step
            x_new = proxF(x - stepsize*grad_new, stepsize)
            Axmb_new = A(x_new) - b # need in backtracking test and re-use for gradient

            if backtrack is None:
                # no backtracking specified
                break
            else:
                xmx = x_new - x
                gval = G(Axmb_new)
                bound = GAxmb + np.vdot(xmx, grad_new).real + l2normsqhalf(xmx)/stepsize
                # test Lipschitz bound, don't need to backtrack if it holds
                if gval <= bound:
                    break
                else:
                    # backtrack
                    stepsize = stepsize*backtrack
                    bts += 1
        
        # residual for convergence check
        r = (x - x_new)/stepsize + grad_new - grad
        
        # update state variables for which we had to track previous values
        grad = grad_new
        x = x_new
        Axmb = Axmb_new
        GAxmb = G(Axmb) # used in backtracking test, and printing function value
        
        # norms for convergence check
        rnorm = tolnorm(r)
        xnorm = tolnorm(x)
        gradnorm = tolnorm(grad)
        
        stopthresh = rabstol + reltol*max(xnorm/stepsize, gradnorm)
        
        if printrate is not None and (k % printrate) == 0:
            val = F(x) + G(Axmb)
            dkt = dict(it=k, val=val, step=stepsize, resid=rnorm, 
                       thresh=stopthresh)
            print(('{it}: val={val:.5}, step={step:.4}, ' + 
                   'resid={resid:.4} ({thresh:.3})').format(**dkt))
            if moreinfo:
                if xstar is not None:
                    dkt['err'] = tolnorm(x_new - xstar)/tolnorm(xstar)
                else:
                    dkt['err'] = np.nan
                hist[k//printrate] = tuple(dkt[key] for key in hist.dtype.names)
        if rnorm < stopthresh:
            break
        
        # expand stepsize
        if expand is not None and backtrack is not None:
            stepsize = stepsize*expand
    
    if printrate is not None:
        if k + 1 >= maxits:
            msg = 'Failed to converge'
        else:
            msg = 'Converged'
        msg += ' after {0} iterations'.format(k + 1)
        if backtrack is not None:
            msg += ' (and {0} backtracks)'.format(bts)
        print(msg)
    
    if moreinfo:
        return dict(x=x, numits=k, r=r, stepsize=stepsize, 
                    hist=hist[:k//printrate])
    else:
        return x

def proxgradaccel(F, G, A, Astar, b, x0, stepsize=1.0, backtrack=0.5, 
                  expand=1.25, reltol=1e-6, abstol=1e-10, 
                  maxits=10000, moreinfo=False, printrate=100,
                  xstar=None):
    """Solve: argmin_x ( F(x) + G(A(x) - b) ) using accelerated prox. gradient.
    
    F and G are callables that return the function value at x or z=A(x)-b.
    
    F(x) must be "proximable", so that the callable F.prox(v, s)
    solves argmin_x ( F(x) + (1/(2s))(||x - v||_2)^2 ).
    
    G(z) must be smooth with gradient function callable as G.grad(z). As a
    function of x, the gradient of the G term is then
        gradGx(x) = Astar(G.grad(A(x) - b)).
    
    A is linear operator, and both the forward (A) and adjoint (Astar)
    operations must be specified.
    
    b is a specified constant.
    
    *********
    Algorithm
    *********
    Parameters:
        stepsize in [0, 2/L]: initial step size when backtracking is used or
            constant step size for all iterations (where L is the Lipschitz 
            constant of G.grad)
        backtrack in [0, 1] or None, amount to decrease step size when local
            Lipschitz condition is violated
        expand >= 1 or None, amount to increase step size after every iteration 
            so that it can adapt to decreasing local Lipschitz constant
    
    Update equations:
      (acceleration)
        gamma = stepsize_old/stepsize
        t = 0.5 + 0.5*sqrt(1 + 4*gamma*t_old**2)
        theta = (t_old - 1)/t
        w = x + theta*(x - x_old)
      (proximal gradient descent)
        grad_new = Astar(G.grad(A(w) - b))
        x_new = F.prox(w_new - stepsize*grad_new), stepsize)
    
    Residual (of 0 in subgradF(x_new) + gradGx(x_new)):
        r = (x - x_new)/stepsize + (grad - grad_new))
    
    *********
    Citations
    *********
    The basic algorithm is described in section 4.3 of N. Parikh and S. Boyd,
    "Proximal Algorithms," Foundations and Trends in Optimization, vol. 1, 
    no. 3, pp. 123-231, 2013.
    
    Expanding backtracking line search is due to K. Scheinberg, D. Goldfarb, 
    and X. Bai, "Fast First-Order Methods for Composite Convex Optimization 
    with Backtracking," pre-print, Apr. 2011 and also S. R. Becker, 
    E. J. Candes, and M. C. Grant, "Templates for convex cone problems with 
    applications to sparse signal recovery," Mathematical Programming 
    Computation, vol. 3, no. 3, pp. 165-218, Aug. 2011.
    
    Adaptive restart of the acceleration comes from B. O'Donoghue and 
    E. Candes, "Adaptive Restart for Accelerated Gradient Schemes," Found 
    Comput Math, pp. 1-18.
    
    """
    proxF = F.prox
    gradG = G.grad
    
    stepsize = float(stepsize)
    
    if moreinfo:
        histdtype = [('it', np.int32), ('val', np.float64), 
                     ('step', np.float64), ('resid', np.float64), 
                     ('thresh', np.float64), ('err', np.float64)]
        hist = np.zeros((maxits - 1)//printrate + 1, dtype=histdtype)
    
    t_old = 1
    stepsize_old = stepsize
    gamma = 1
    
    x = x0
    Ax = A(x)
    x_old = x
    Ax_old = Ax
    grad = Astar(gradG(Ax - b))
    w = x # for adaptive restart check
    
    tolnorm = l2norm
    rabstol = abstol*tolnorm(np.ones_like(x0))
    
    bts = 0
    
    for k in xrange(maxits):
        # "gradient" adaptive restart: 
        # reset "momentum" when acceleration direction (x - x_old) is in 
        # opposite direction of previous step prox gradient step (x - w)
        if np.vdot(w - x, x - x_old).real > 0:
            t_old = 1
        
        # loop for backtracking line search
        while True:
            # acceleration
            t = 0.5 + 0.5*np.sqrt(1 + 4*gamma*t_old**2)
            theta = (t_old - 1)/t
            w = x + theta*(x - x_old)
            Aw = Ax + theta*(Ax - Ax_old) # limit A evals by exploiting linearity
            Awmb = Aw - b # re-use in backtracking test
            # proximal gradient step
            grad_new = Astar(gradG(Awmb))
            x_new = proxF(w - stepsize*grad_new, stepsize)
            Ax_new = A(x_new) # needed to limit A evals as above
            Axmb = Ax_new - b # need in backtracking test and printing

            if backtrack is None:
                # no backtracking specified
                break
            else:
                xmw = x_new - w
                gval = G(Axmb)
                bound = G(Awmb) + np.vdot(xmw, grad_new).real + l2normsqhalf(xmw)/stepsize
                # test Lipschitz bound, don't need to backtrack if it holds
                if gval <= bound:
                    break
                else:
                    # backtrack
                    stepsize = stepsize*backtrack
                    gamma = stepsize_old/stepsize
                    bts += 1
        
        # residual for convergence check
        r = (x - x_new)/stepsize + grad_new - grad
        
        # update state variables for which we had to track previous values
        t_old = t
        stepsize_old = stepsize
        grad = grad_new
        x_old = x
        Ax_old = Ax
        x = x_new
        Ax = Ax_new
        
        # norms for convergence check
        rnorm = tolnorm(r)
        xnorm = tolnorm(x)
        gradnorm = tolnorm(grad)
        
        stopthresh = rabstol + reltol*max(xnorm/stepsize, gradnorm)
        
        if printrate is not None and (k % printrate) == 0:
            val = F(x_new) + G(Axmb)
            dkt = dict(it=k, val=val, step=stepsize, resid=rnorm, 
                       thresh=stopthresh)
            print(('{it}: val={val:.5}, step={step:.4}, ' + 
                   'resid={resid:.4} ({thresh:.3})').format(**dkt))
            if moreinfo:
                if xstar is not None:
                    dkt['err'] = tolnorm(x_new - xstar)/tolnorm(xstar)
                else:
                    dkt['err'] = np.nan
                hist[k//printrate] = tuple(dkt[key] for key in hist.dtype.names)
        if rnorm < stopthresh:
            break
        
        # expand stepsize
        if expand is not None and backtrack is not None:
            stepsize = stepsize*expand
            gamma = stepsize_old/stepsize
    
    if printrate is not None:
        if k + 1 >= maxits:
            msg = 'Failed to converge'
        else:
            msg = 'Converged'
        msg += ' after {0} iterations'.format(k + 1)
        if backtrack is not None:
            msg += ' (and {0} backtracks)'.format(bts)
        print(msg)
    
    if moreinfo:
        return dict(x=x, numits=k, w=w, r=r, stepsize=stepsize, 
                    hist=hist[:k//printrate])
    else:
        return x

def admm(F, G, x0, y0=None, pen=1.0, residgap=2, penfactor=1.5, reltol=1e-6, 
         abstol=1e-10, maxits=10000, moreinfo=False, printrate=100,
         xstar=None):
    """Solve: argmin_x ( F(x) + G(x) ) using ADMM.
    
    ADMM stands for Alternating Direction Method of Multipliers.
    
    F and G are callables that return the function value at x.
    
    F(x) and G(x) must be "proximable", with callable F.prox(v, s) that
    solves argmin_x ( F(x) + (1/(2s))(||x - v||_2)^2 ) and similar for G.
    
    *********
    Algorithm
    *********
    Parameters:
        pen controlled automatically by penalty parameter adjustment
            according to residgap > 1 and penfactor > 1
    
    Update equations:
        x_new = F.prox(z - u, pen)
        z_new = G.prox(x_new + u, pen)
        u_new = u + x_new - z_new
    
    Residuals:
      (primal feasibility)
        r = x_new - z_new
      (dual feasibility)
        s = (z - z_new)/pen
    
    *********
    Citations
    *********
    The basic algorithm is described in section 4.4 of N. Parikh and S. Boyd, 
    "Proximal Algorithms," Foundations and Trends in Optimization, vol. 1, 
    no. 3, pp. 123-231, 2013.
    
    Varying penalty parameter suggested in S. Boyd, N. Parikh, E. Chu, 
    B. Peleato, and J. Eckstein, "Distributed Optimization and Statistical 
    Learning via the Alternating Direction Method of Multipliers," Found. 
    Trends Mach. Learn., vol. 3, no. 1, pp. 1-122, Jan. 2011.
    
    """
    proxF = F.prox
    proxG = G.prox
    if y0 is None:
        y0 = np.zeros_like(x0)
    
    pen = float(pen)
    
    if moreinfo:
        histdtype = [('it', np.int32), ('val', np.float64), 
                     ('pen', np.float64), 
                     ('resid_p', np.float64), ('thresh_p', np.float64),
                     ('resid_d', np.float64), ('thresh_d', np.float64),
                     ('err', np.float64)]
        hist = np.zeros((maxits - 1)//printrate + 1, dtype=histdtype)
    
    x = x0
    u = y0*pen
    z = proxG(x + u, pen)
    r = x - z
    u = u + r
    
    tolnorm = l2norm
    rabstol = abstol*tolnorm(np.ones_like(u0))
    sabstol = abstol*tolnorm(np.ones_like(x0))
    
    for k in xrange(maxits):
        # primal updates
        x = proxF(z - u, pen)
        z_new = proxG(x + u, pen)
        # residual calculation
        r = x - z_new
        s = (z - z_new)/pen
        # dual update
        u = u + r
        
        # # update state variables for which we had to track previous values
        z = z_new
        
        # norms for convergence check
        rnorm = tolnorm(r)
        snorm = tolnorm(s)
        xnorm = tolnorm(x)
        znorm = tolnorm(z)
        unorm = tolnorm(u)
        
        rstopthresh = rabstol + reltol*max(xnorm, znorm)
        sstopthresh = sabstol + reltol*unorm/pen
        
        if printrate is not None and (k % printrate) == 0:
            val = F(x) + G(x)
            dkt = dict(it=k, val=val, pen=pen, resid_p=rnorm, 
                       thresh_p=rstopthresh, resid_d=snorm, 
                       thresh_d=sstopthresh)
            print(('{it}: value={val:.5}, pen={pen:.4}, ' + 
                   'resid_p={resid_p:.4} ({thresh_p:.3}), ' +
                   'resid_d={resid_d:.4} ({thresh_d:.3})').format(**dkt))
            if moreinfo:
                if xstar is not None:
                    dkt['err'] = tolnorm(x_new - xstar)/tolnorm(xstar)
                else:
                    dkt['err'] = np.nan
                hist[k//printrate] = tuple(dkt[key] for key in hist.dtype.names)
        # can't calculate dual function value, so best stopping criterion
        # is to see if primal and dual feasibility residuals are small
        if rnorm < rstopthresh and snorm < sstopthresh:
            break
        
        # penalty parameter adjustment
        if k < 100:
            if rnorm > residgap*snorm:
                pen = pen/penfactor
                # scaled dual variable u=y*pen, so update u with y constant
                u = u/penfactor
            elif snorm > residgap*rnorm:
                pen = pen*penfactor
                # scaled dual variable u=y*pen, so update u with y constant
                u = u*penfactor

    if printrate is not None:
        if k + 1 >= maxits:
            msg = 'Failed to converge'
        else:
            msg = 'Converged'
        msg += ' after {0} iterations'.format(k + 1)
        print(msg)
    
    if moreinfo:
        return dict(x=x, z=z, y=u/pen, numits=k, r=r, s=s, pen=pen, 
                    hist=hist[:k//printrate])
    else:
        return x

def admmlin(F, G, A, Astar, b, x0, y0=None, stepsize=1.0, backtrack=0.5,
            expand=1.25, pen=1.0, residgap=2, penfactor=1.5, relax=1.0, 
            reltol=1e-6, abstol=1e-10, maxits=10000, 
            moreinfo=False, printrate=100, xstar=None):
    """Solve: argmin_x ( F(x) + G(A(x) - b) ) using linearized ADMM.
    
    ADMM stands for Alternating Direction Method of Multipliers.
    
    F and G are callables that return the function value at x or z=A(x)-b.
    
    F(x) and G(x) must be "proximable", with callable F.prox(v, s) that
    solves argmin_x ( F(x) + (1/(2s))(||x - v||_2)^2 ) and similar for G.
    
    A is linear operator, and both the forward (A) and adjoint (Astar)
    operations must be specified.
    
    b is a specified constant.
    
    *********
    Algorithm
    *********
    Parameters:
        stepsize in [0, 1/(||A||_2)**2]: initial step size when backtracking is used or
            constant step size for all iterations (where L is the Lipschitz 
            constant of G.grad)
        backtrack in [0, 1] or None, amount to decrease step size when local
            Lipschitz condition is violated
        expand >= 1 or None, amount to increase step size after every iteration 
            so that it can adapt to decreasing local Lipschitz constant
        pen controlled automatically by penalty parameter adjustment
            according to residgap > 1 and penfactor > 1
        relax in [0, 2] for under-relaxation ([0, 1]) or over-relaxation 
            ([1, 2]) when calculating A(x) = relax*A(x) + (1-relax)*(z - b)
    
    Update equations:
        x_new = F.prox(x + stepsize/pen*Astar(2*u - u_old), stepsize)
        z_new = G.prox(A(x_new) - b + u, pen)
        u_new = u + A(x_new) - b - z_new
    
    Residuals:
      (primal feasibility)
        r = A(x_new) - b - z_new
      (dual feasibility)
        s = Astar(u_new - u)/pen + (x - x_new)/stepsize
    
    *********
    Citations
    *********
    The basic algorithm is similar to the one described in section 4.5 of 
    N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends 
    in Optimization, vol. 1, no. 3, pp. 123-231, 2013.
    
    Varying penalty parameter and over-relaxation suggested in S. Boyd, 
    N. Parikh, E. Chu, B. Peleato, and J. Eckstein, "Distributed Optimization 
    and Statistical Learning via the Alternating Direction Method of 
    Multipliers," Found. Trends Mach. Learn., vol. 3, no. 1, pp. 1-122, 
    Jan. 2011.
    
    Interpretation of 'stepsize' parameter as a stepsize and backtracking with
    respect to it is novel but similar to proximal gradient backtracking.
    
    """
    proxF = F.prox
    proxG = G.prox
    if y0 is None:
        y0 = np.zeros_like(A(x0))
    
    stepsize = float(stepsize)
    pen = float(pen)
    
    if moreinfo:
        histdtype = [('it', np.int32), ('val', np.float64), 
                     ('step', np.float64), ('pen', np.float64), 
                     ('resid_p', np.float64), ('thresh_p', np.float64),
                     ('resid_d', np.float64), ('thresh_d', np.float64),
                     ('err', np.float64)]
        hist = np.zeros((maxits - 1)//printrate + 1, dtype=histdtype)
    
    if expand is not None and backtrack is not None:
        # so initial stepsize is as specified after expansion
        stepsize = stepsize/expand
    
    one_minus_relax = 1 - relax
    
    x = x0
    Ax = A(x)
    u = y0*pen
    Asu = Astar(u)
    Asu_old = Asu
    z = A(x) - b # needed for relaxation, makes Ax_relax = A(x) for first step
    
    tolnorm = l2norm
    bnorm = tolnorm(b)
    rabstol = abstol*tolnorm(np.ones_like(y0))
    sabstol = abstol*tolnorm(np.ones_like(x0))
    
    bts = 0
    
    for k in xrange(maxits):
        # loop for backtracking line search
        while True:
            # x primal update
            x_new = proxF(x - stepsize/pen*(2*Asu - Asu_old), stepsize)
            Ax_new = A(x_new)

            if backtrack is None:
                # no backtracking specified
                break
            else:
                # test majorization bound, don't need to backtrack if it holds
                dAx = l2normsqhalf(Ax_new - Ax)
                dx = l2normsqhalf(x_new - x)
                if dAx/pen <= dx/stepsize:
                    break
                else:
                    # backtrack
                    stepsize = stepsize*backtrack
                    bts += 1
        
        # z primal update
        Axrelax = relax*Ax_new + one_minus_relax*(z + b) # need norm later
        Axmb = Axrelax - b
        z = proxG(Axmb + u, pen)
        # primal residual calculation
        r = Axmb - z
        # dual update
        u = u + r
        # dual residual calculation
        Asu_new = Astar(u)
        s = (Asu_new - Asu)/pen + (Asu_old - Asu)/pen + (x - x_new)/stepsize
        
        # update state variables for which we had to track previous values
        x = x_new
        Ax = Ax_new
        Asu_old = Asu
        Asu = Asu_new
        
        # norms for convergence check
        rnorm = tolnorm(r)
        snorm = tolnorm(s)
        Axnorm = tolnorm(Axrelax)
        znorm = tolnorm(z)
        Asunorm = tolnorm(Asu)
        xnorm = tolnorm(x)
        
        rstopthresh = rabstol + reltol*max(Axnorm, znorm, bnorm)
        sstopthresh = sabstol + reltol*max(Asunorm/pen, xnorm/stepsize)
        
        if printrate is not None and (k % printrate) == 0:
            val = F(x) + G(z) + np.vdot(u/pen, Ax - b - z).real
            dkt = dict(it=k, val=val, step=stepsize, pen=pen, 
                       resid_p=rnorm, thresh_p=rstopthresh,
                       resid_d=snorm, thresh_d=sstopthresh)
            print(('{it}: val={val:.5}, step={step:.4}, pen={pen:.4}, ' + 
                   'resid_p={resid_p:.4} ({thresh_p:.3}), ' + 
                   'resid_d={resid_d:.4} ({thresh_d:.3})').format(**dkt))
            if moreinfo:
                if xstar is not None:
                    dkt['err'] = tolnorm(x_new - xstar)/tolnorm(xstar)
                else:
                    dkt['err'] = np.nan
                hist[k//printrate] = tuple(dkt[key] for key in hist.dtype.names)
        # can't calculate dual function value, so best stopping criterion
        # is to see if primal and dual feasibility residuals are small
        if rnorm < rstopthresh and snorm < sstopthresh:
            break
        
        # penalty parameter adjustment
        if k < 100:
            if rnorm > residgap*snorm:
                pen = pen/penfactor
                # scaled dual variable u=y*pen, so update u with y constant
                u = u/penfactor
                Asu = Asu/penfactor
                Asu_old = Asu_old/penfactor
            elif snorm > residgap*rnorm:
                pen = pen*penfactor
                # scaled dual variable u=y*pen, so update u with y constant
                u = u*penfactor
                Asu = Asu*penfactor
                Asu_old = Asu_old*penfactor
        
        # expand stepsize
        if backtrack is not None and dAx > 0:
            stepsize = dx*pen/dAx

    if printrate is not None:
        if k + 1 >= maxits:
            msg = 'Failed to converge'
        else:
            msg = 'Converged'
        msg += ' after {0} iterations'.format(k + 1)
        if backtrack is not None:
            msg += ' (and {0} backtracks)'.format(bts)
        print(msg)
    
    if moreinfo:
        return dict(x=x, z=z, y=u/pen, numits=k, r=r, s=s, stepsize=stepsize, 
                    pen=pen, hist=hist[:k//printrate])
    else:
        return x