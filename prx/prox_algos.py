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

    The basic algorithm is described in section 4.2 of [1]. Expanding
    backtracking line search is due to [2] and [3].

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
    [1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends in
    Optimization, vol. 1, no. 3, pp. 123-231, 2013.

    [2] K. Scheinberg, D. Goldfarb, and X. Bai, "Fast First-Order Methods for
    Composite Convex Optimization with Backtracking," pre-print, Apr. 2011

    [3] S. R. Becker, E. J. Candes, and M. C. Grant, "Templates for convex cone
    problems with  applications to sparse signal recovery," Mathematical
    Programming Computation, vol. 3, no. 3, pp. 165-218, Aug. 2011.

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

    The basic algorithm is described in section 4.3 of [1]. Expanding
    backtracking line search is due to [2] and [3]. Adaptive restart of the
    acceleration comes from [4].

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
    [1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends in
    Optimization, vol. 1, no. 3, pp. 123-231, 2013.

    [2] K. Scheinberg, D. Goldfarb, and X. Bai, "Fast First-Order Methods for
    Composite Convex Optimization with Backtracking," pre-print, Apr. 2011

    [3] S. R. Becker, E. J. Candes, and M. C. Grant, "Templates for convex cone
    problems with  applications to sparse signal recovery," Mathematical
    Programming Computation, vol. 3, no. 3, pp. 165-218, Aug. 2011.

    [4] B. O'Donoghue and E. Candes, "Adaptive Restart for Accelerated Gradient
    Schemes," Found Comput Math, pp. 1-18.

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

    The basic algorithm is described in section 4.4 of [1]. Varying penalty
    parameter is suggested in [2].

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
    [1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends
    in Optimization, vol. 1, no. 3, pp. 123-231, 2013.

    [2] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, "Distributed
    Optimization and Statistical Learning via the Alternating Direction Method
    of Multipliers," Found. Trends Mach. Learn., vol. 3, no. 1, pp. 1-122,
    Jan. 2011.

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
            expand=1.25, pen=1.0, residgap=10, penfactor=1.5, relax=1.0,
            reltol=1e-6, abstol=1e-10, maxits=10000,
            moreinfo=False, printrate=100, xstar=None):
    """Solve: argmin_x ( F(x) + G(A(x) - b) ) using linearized ADMM.

    ADMM stands for Alternating Direction Method of Multipliers. This
    linearized version of ADMM is sometimes called the Split Inexact Uzawa
    method.

    F and G are callables that return the function value at x or z=A(x)-b.

    F(x) and G(z) must be "proximable", with callable F.prox(v, s) that
    solves argmin_x ( F(x) + (1/(2s))(||x - v||_2)^2 ) and similar for G.

    A is linear operator, and both the forward (A) and adjoint (Astar)
    operations must be specified.

    b is a specified constant.

    The basic algorithm is similar to the one described in section 4.5 of
    [1]. Varying penalty parameter and over-relaxation suggested in [2].
    Interpretation of 'stepsize' parameter as a stepsize and backtracking with
    respect to it is novel but similar to proximal gradient backtracking. See
    [3] for how this algorithm (there called Split Inexact Uzawa on SP_P)
    relates to similar algorithms like PDHG.

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
    [1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends
    in Optimization, vol. 1, no. 3, pp. 123-231, 2013.

    [2] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, "Distributed
    Optimization and Statistical Learning via the Alternating Direction Method
    of Multipliers," Found. Trends Mach. Learn., vol. 3, no. 1, pp. 1-122,
    Jan. 2011.

    [3] E. Esser, X. Zhang, and T. F. Chan, "A General Framework for a Class of
    First Order Primal-Dual Algorithms for Convex Optimization in Imaging
    Science," SIAM Journal on Imaging Sciences, vol. 3, no. 4, pp. 1015-1046,
    Jan. 2010.

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
        if expand is not None and backtrack is not None and dAx > 0:
            stepsize = dx*pen/dAx*expand

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

def pdhg(F, G, A, Astar, b, x0, y0=None, step_p=1.0, step_d=1.0,
         reltol=1e-6, abstol=1e-10, maxits=10000,
         moreinfo=False, printrate=100, xstar=None):
    """Solve: argmin_x ( F(x) + G(A(x) - b) ) using PDHG.

    PDHG stands for Primal Dual Hybrid Gradient.

    F and G are callables that return the function value at x or z=A(x)-b.

    F(x) and G(z) must be "proximable", with callable F.prox(v, s) that
    solves argmin_x ( F(x) + (1/(2s))(||x - v||_2)^2 ) and similar for G.

    A is linear operator, and both the forward (A) and adjoint (Astar)
    operations must be specified.

    b is a specified constant.

    The basic algorithm is described in [1], along with a description of how
    this algorithm relates to similar algorithms like ADMM and linearized ADMM.

    *********
    Algorithm
    *********
    Parameters:
        step_p, step_d: primal and dual step sizes

    Update equations:
        x_new = F.prox(x - step_p*Astar(y), step_p)
        xbar = 2*x_new - x
        y_new = G.conjugate.prox(y + step_d*(A(x_new) - b), step_d)

    Residuals:
      (primal)
        p = (x - x_new)/step_p - Astar(y - y_new)
      (dual)
        d = (y - y_new)/step_d - A(x - x_new)

    *********
    Citations
    *********
    [1] E. Esser, X. Zhang, and T. F. Chan, "A General Framework for a Class of
    First Order Primal-Dual Algorithms for Convex Optimization in Imaging
    Science," SIAM Journal on Imaging Sciences, vol. 3, no. 4, pp. 1015-1046,
    Jan. 2010.

    """
    proxF = F.prox
    Fconj = F.conjugate
    Gconj = G.conjugate
    dualproxG = Gconj.prox
    if y0 is None:
        y0 = np.zeros_like(A(x0))

    pstep = float(step_p)
    dstep = float(step_d)

    if moreinfo:
        histdtype = [('it', np.int32), ('val', np.float64),
                     ('step_p', np.float64), ('step_d', np.float64),
                     ('resid_p', np.float64), ('thresh_p', np.float64),
                     ('resid_d', np.float64), ('thresh_d', np.float64),
                     ('err', np.float64)]
        hist = np.zeros((maxits - 1)//printrate + 1, dtype=histdtype)

    ptheta = 1
    dtheta = 0

    x = x0
    Ax = A(x)
    y = y0
    Asy = Astar(y)
    y_old = y
    Asy_old = Asy

    tolnorm = l2norm
    pabstol = abstol*tolnorm(np.ones_like(x0))
    dabstol = abstol*tolnorm(np.ones_like(y0))

    bts = 0

    for k in xrange(maxits):
        while True:
            # primal update
            # acceleration
            ybar = y + dtheta*(y - y_old)
            Asybar = Asy + dtheta*(Asy - Asy_old)
            # gradient descent step wrt Lagrange multiplier term
            xhat = x - pstep*Asybar
            # prox step
            x_new = proxF(xhat, pstep)
            Ax_new = A(x_new)

            break
            #if backtrack is None:
                #break
            #else:
                #Axmb = Ax_new - b
                #plhs = G(Axmb) + Gconj(ybar) - np.vdot(ybar, Axmb).real
                #prhs = l2normsqhalf(x_new - x)/pstep
                #if plhs <= prhs:
                    #break
                #else:
                    #print(plhs, prhs)
                    ## backtrack
                    #pstep = pstep*backtrack

        while True:
            # dual update
            # acceleration
            xbar = x_new + ptheta*(x_new - x)
            Axbar = Ax_new + ptheta*(Ax_new - Ax)
            # gradient ascent step wrt Lagrange multiplier term
            yhat = y + dstep*(Axbar - b)
            # prox step
            y_new = dualproxG(yhat, dstep)
            Asy_new = Astar(y_new)

            break
            #if backtrack is None:
                #break
            #else:
                #dlhs = Fconj(-Asy_new) + F(xbar) - np.vdot(xbar, -Asy_new).real
                #drhs = l2normsqhalf(y_new - y)/dstep
                #if dlhs <= drhs:
                    #break
                #else:
                    #print(dlhs, drhs)
                    ## backtrack
                    #dstep = dstep*backtrack

        # calculate residuals
        p = (x - x_new)/pstep - (Asy - Asy_new) - dtheta*(Asy - Asy_old)
        d = (y - y_new)/dstep - ptheta*(Ax - Ax_new)

        # update state variables for which we had to track previous values
        x = x_new
        Ax = Ax_new
        y_old = y
        Asy_old = Asy
        y = y_new
        Asy = Asy_new

        # norms for convergence check
        pnorm = tolnorm(p)
        dnorm = tolnorm(d)
        xnorm = tolnorm(x)
        Asynorm = tolnorm(Asy)
        ynorm = tolnorm(y)
        Axnorm = tolnorm(Ax)

        pstopthresh = pabstol + reltol*max(xnorm/pstep, Asynorm)
        dstopthresh = dabstol + reltol*max(ynorm/dstep, Axnorm)

        if printrate is not None and (k % printrate) == 0:
            val = F(x) + G(Ax - b)
            dval = -Fconj(y) - Gconj(y) - np.vdot(y, b).real
            dkt = dict(it=k, val=val, step_p=pstep, step_d=dstep,
                       resid_p=pnorm, thresh_p=pstopthresh,
                       resid_d=dnorm, thresh_d=dstopthresh)
            print(('{it}: val={val:.5}, step_p={step_p:.4}, ' +
                   'step_d={step_d:.4}, ' +
                   'res_p={resid_p:.4} ({thresh_p:.3}), ' +
                   'res_d={resid_d:.4} ({thresh_d:.3})').format(**dkt))
            if moreinfo:
                if xstar is not None:
                    dkt['err'] = tolnorm(x_new - xstar)/tolnorm(xstar)
                else:
                    dkt['err'] = np.nan
                hist[k//printrate] = tuple(dkt[key] for key in hist.dtype.names)
        # can't calculate dual function value, so best stopping criterion
        # is to see if primal and dual residuals are small
        if pnorm < pstopthresh and dnorm < dstopthresh:
            break

        ## penalty parameter adjustment
        #if k < 100:
            #if rnorm > residgap*snorm:
                #pen = pen/penfactor
                ## scaled dual variable u=y*pen, so update u with y constant
                #u = u/penfactor
                #Asu = Asu/penfactor
                #Asu_old = Asu_old/penfactor
            #elif snorm > residgap*rnorm:
                #pen = pen*penfactor
                ## scaled dual variable u=y*pen, so update u with y constant
                #u = u*penfactor
                #Asu = Asu*penfactor
                #Asu_old = Asu_old*penfactor

        # expand stepsize
        #if backtrack is not None and dAx > 0:
            #stepsize = dx*pen/dAx
        #if expand is not None and backtrack is not None:
            #pstep = pstep*expand
            #dstep = dstep*expand

    if printrate is not None:
        if k + 1 >= maxits:
            msg = 'Failed to converge'
        else:
            msg = 'Converged'
        msg += ' after {0} iterations'.format(k + 1)
        #if backtrack is not None:
            #msg += ' (and {0} backtracks)'.format(bts)
        print(msg)

    if moreinfo:
        return dict(x=x, y=y, numits=k, p=p, d=d, pstep=pstep, dstep=dstep,
                    hist=hist[:k//printrate])
    else:
        return x
