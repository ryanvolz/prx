# ----------------------------------------------------------------------------
# Copyright (c) 2014-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Primal-dual algorithms."""

from __future__ import division

import numpy as np

from ..fun.norms import l2norm
from ._common import docstring_wrapper as _docstring_wrapper

__all__ = ('pdhg',)


@_docstring_wrapper
def pdhg(F, G, A, Astar, b, x0, y0=None, step_p=1.0, step_d=1.0,
         reltol=1e-6, abstol=1e-10, maxits=10000,
         moreinfo=False, printrate=100, xstar=None):
    """Solve: argmin_x ( F(x) + G(A(x) - b) ) using PDHG.

    PDHG stands for Primal Dual Hybrid Gradient.


    Notes
    -----

    The basic algorithm is described in [1]_, along with a description of how
    this algorithm relates to similar algorithms like ADMM and linearized ADMM.


    References
    ----------

    .. [1] E. Esser, X. Zhang, and T. F. Chan, "A General Framework for a
       Class of First Order Primal-Dual Algorithms for Convex Optimization in
       Imaging Science," SIAM Journal on Imaging Sciences, vol. 3, no. 4, pp.
       1015-1046, Jan. 2010.

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

    for k in range(maxits):
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
            # if backtrack is None:
            #     break
            # else:
            #     Axmb = Ax_new - b
            #     plhs = G(Axmb) + Gconj(ybar) - np.vdot(ybar, Axmb).real
            #     prhs = l2normsqhalf(x_new - x)/pstep
            #     if plhs <= prhs:
            #         break
            #     else:
            #         print(plhs, prhs)
            #         # backtrack
            #         pstep = pstep*backtrack

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
            # if backtrack is None:
            #     break
            # else:
            #     dlhs = Fconj(-Asy_new) + F(xbar) - np.vdot(xbar, -Asy_new).real
            #     drhs = l2normsqhalf(y_new - y)/dstep
            #     if dlhs <= drhs:
            #         break
            #     else:
            #         print(dlhs, drhs)
            #         # backtrack
            #         dstep = dstep*backtrack

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
            val = float(F(x) + G(Ax - b))
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
                hist[k//printrate] = tuple(dkt[ky] for ky in hist.dtype.names)
        # can't calculate dual function value, so best stopping criterion
        # is to see if primal and dual residuals are small
        if pnorm < pstopthresh and dnorm < dstopthresh:
            break

        # # penalty parameter adjustment
        # if k < 100:
        #     if rnorm > residgap*snorm:
        #         pen = pen/penfactor
        #         # scaled dual variable u=y*pen, so update u with y constant
        #         u = u/penfactor
        #         Asu = Asu/penfactor
        #         Asu_old = Asu_old/penfactor
        #     elif snorm > residgap*rnorm:
        #         pen = pen*penfactor
        #         # scaled dual variable u=y*pen, so update u with y constant
        #         u = u*penfactor
        #         Asu = Asu*penfactor
        #         Asu_old = Asu_old*penfactor
        #
        # expand stepsize
        # if backtrack is not None and dAx > 0:
        #     stepsize = dx*pen/dAx
        # if expand is not None and backtrack is not None:
        #     pstep = pstep*expand
        #     dstep = dstep*expand

    if printrate is not None:
        if k + 1 >= maxits:
            msg = 'Failed to converge'
        else:
            msg = 'Converged'
        msg += ' after {0} iterations'.format(k + 1)
        # if backtrack is not None:
        #     msg += ' (and {0} backtracks)'.format(bts)
        print(msg)

    if moreinfo:
        return dict(x=x, y=y, numits=k+1, p=p, d=d, pstep=pstep, dstep=dstep,
                    hist=hist[:k//printrate])
    else:
        return x
