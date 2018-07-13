# ----------------------------------------------------------------------------
# Copyright (c) 2014-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Alternating direction method of multipliers algorithms."""

from __future__ import division

import numpy as np

from ..fun.norms import l2norm, l2normsqhalf
from ._common import docstring_wrapper as _docstring_wrapper

__all__ = ('admm', 'admmlin')


@_docstring_wrapper
def admm(F, G, x0, y0=None, pen=1.0, residgap=2, penfactor=1.5, reltol=1e-6,
         abstol=1e-10, maxits=10000, moreinfo=False, printrate=100,
         xstar=None):
    """Minimize ``F(x) + G(x)`` using ADMM.

    ADMM, for Alternating Direction Method of Multipliers, is an algorithm for
    solving convex optimization problems with a split objective function. The
    method is also known as Douglas-Rachford splitting. ADMM is most useful
    when the objective is non-smooth and the prox operator of F+G would be
    hard to solve, but the prox operators of F and G separately are easy to
    solve. This implementation includes a varying penalty parameter that
    helps improve convergence.

    Many splittings of the objective function might be possible, including
    swapping the roles of F and G, and the resulting optimization problems
    will be different. If F encodes constraints, then the solution will
    always satisfy them exactly. Conversely, any constraints in G will only
    be satisfied in the limit, and therefore the solution may not meet them
    exactly. It may be wise to try different splittings of a particular
    problem to find out which formulation works best.


    Parameters
    ----------

    F, G : callable with ``prox`` method
        ``F(x)`` and ``G(x)``, the two parts of the objective function. `F`
        and `G` must be "proximable", so that the callable ``F.prox(v, s)``
        returns the x that minimizes

            ``F(x) + ( ||x - v||_2 )^2 / (2*s)``.

    x0 : array
        Initial iterate for x.

    y0 : array, optional
        Initial iterate for ``y = u/pen``, the unscaled dual variable.

    pen : float, optional
        Initial penalty parameter, the function of which is to balance
        progress between the primal and dual spaces.

    residgap : float, optional
        A value greater than one. If the ratio or inverse ratio of the
        primal residual to dual residual exceeds this number within the first
        100 iterations, the penalty parameter will be adjusted by `penfactor`
        to help normalize the residuals.

    penfactor : float, optional
        A value greater than one. If the residual gap is reached, the penalty
        parameter will be multiplied or divided by `penfactor` to help
        normalize the residuals.


    Returns
    -------

    x : array, if `moreinfo` is ``False``
        The solution.

    d : dict, if `moreinfo` is ``True``
        A dictionary consisting of:

            x : array
                The solution.

            z : array
                The final value of the consensus variable, equaling x in the
                limit.

            y : array
                The final unscaled dual variable, equaling u/pen.

            numits : int
                The number of iterations completed.

            r : array
                The final primal feasibility residual.

            s : array
                The final dual feasibility residual.

            pen : float
                The final penalty parameter.

            hist : structured array
                The objective value, penalty, primal and dual residual norms,
                primal and dual convergence thresholds, and relative error at
                each numbered iteration.


    Other Parameters
    ----------------

    {common_parameters}


    See Also
    --------

    admmlin


    Notes
    -----

    It is often convenient to think of ADMM in terms of the consensus form
    of the optimization problem::

        minimize    F(x) + G(z)
        subject to  x - z = 0.

    In this form, a second variable z is introduced with the constraint that
    z equal x. It is natural to then consider an optimization method that
    alternately optimizes in terms of x and z, with a correction to enforce
    the constraint. ADMM is such an iterative first-order optimization method
    where each iteration is composed of three steps:

        Prox step with respect to `F`::

            x_new = F.prox(z - u, pen)

        Prox step with respect to `G`::

            z_new = G.prox(x_new + u, pen)

        Dual step for the constraint that ``x_new == z_new``::

            u_new = u + x_new - z_new

    Convergence is determined in terms of both primal and dual residuals:

        Primal feasibility residual::

            r = x - z_new

        Dual feasibility residual::

            s = (z - z_new)/pen

    The basic algorithm is described in section 4.4 of [1]_. Varying penalty
    parameter is suggested in [2]_.


    References
    ----------

    .. [1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and
       Trends in Optimization, vol. 1, no. 3, pp. 123-231, 2013.

    .. [2] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein,
       "Distributed Optimization and Statistical Learning via the Alternating
       Direction Method of Multipliers," Found. Trends Mach. Learn., vol. 3,
       no. 1, pp. 1-122, Jan. 2011.

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
    rabstol = abstol*tolnorm(np.ones_like(y0))
    sabstol = abstol*tolnorm(np.ones_like(x0))

    for k in range(maxits):
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
            val = float(F(x) + G(x))
            dkt = dict(it=k, val=val, pen=pen, resid_p=rnorm,
                       thresh_p=rstopthresh, resid_d=snorm,
                       thresh_d=sstopthresh)
            print(('{it}: value={val:.5}, pen={pen:.4}, ' +
                   'resid_p={resid_p:.4} ({thresh_p:.3}), ' +
                   'resid_d={resid_d:.4} ({thresh_d:.3})').format(**dkt))
            if moreinfo:
                if xstar is not None:
                    dkt['err'] = tolnorm(x - xstar)/tolnorm(xstar)
                else:
                    dkt['err'] = np.nan
                hist[k//printrate] = tuple(dkt[ky] for ky in hist.dtype.names)
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
        return dict(x=x, z=z, y=u/pen, numits=k+1, r=r, s=s, pen=pen,
                    hist=hist[:k//printrate])
    else:
        return x


@_docstring_wrapper
def admmlin(F, G, A, Astar, b, x0, y0=None, stepsize=1.0, backtrack=0.5,
            expand=1.25, pen=1.0, residgap=10, penfactor=1.5, relax=1.0,
            reltol=1e-6, abstol=1e-10, maxits=10000,
            moreinfo=False, printrate=100, xstar=None):
    """Minimize ``F(x) + G(A(x) - b)`` using linearized ADMM.

    Linearized ADMM is a variant of ADMM that solves convex optimization
    problems with a split objective that explicitly includes an affine
    transformation in the G term. It is often non-trivial to accommodate a
    linear transformation when evaluating the prox operator of most functions,
    so its explicit inclusion in the algorithm itself can bring a large
    benefit. Linearized ADMM is also sometimes called the Split Inexact Uzawa
    method. This implementation includes an adaptive step size and varying
    penalty parameter that helps improve convergence.


    Parameters
    ----------

    F, G : callable with ``prox`` method
        ``F(x)`` and ``G(x)``, the two parts of the objective function. `F`
        and `G` must be "proximable", so that the callable ``F.prox(v, s)``
        returns the x that minimizes

            ``F(x) + ( ||x - v||_2 )^2 / (2*s)``.

    A : callable
        ``A(x)`` is a linear operator, used in the G term of the objective
        function. Although not checked, it must obey the linearity condition

            ``A(a*x + b*y) == a*A(x) + b*A(y)``.

    Astar : callable
        ``Astar(z)``, the adjoint operator of `A`. By definition, `Astar`
        satisfies

            ``vdot(A(x), z) == vdot(x, Astar(z))``

        for all x, z and the inner product ``vdot``. If, for instance, `A`
        represented multiplication by a matrix M, `Astar` would then represent
        multiplcation by the complex conjugate transpose of M.

    b : array
        Constant used in the G term of the objective function.

    x0 : array
        Initial iterate for x.

    y0 : array, optional
        Initial iterate for ``y = u/pen``, the unscaled dual variable.

    stepsize : float, optional
        Initial step size when backtracking is used, or constant step size for
        all iterations when backtracking is not used. Must be between 0 and
        ``1/opnorm(A)**2`` to guarantee convergence, where ``opnorm`` is the
        l2-induced operator norm of `A`.

    backtrack : float | ``None``, optional
        Backtracking multiplication factor between 0 and 1 that decreases the
        step size when the majorization condition is violated. If ``None``,
        no backtracking is used.

    expand : float | ``None``, optional
        Expansion multiplication factor greater than 1 that increases the step
        size after every iteration. This allows the step size to adapt to a
        loosening majorization bound and improve convergence. If ``None``, no
        expansion is used.

    pen : float, optional
        Initial penalty parameter, the function of which is to balance
        progress between the primal and dual spaces.

    residgap : float, optional
        A value greater than one. If the ratio or inverse ratio of the
        primal residual to dual residual exceeds this number within the first
        100 iterations, the penalty parameter will be adjusted by `penfactor`
        to help normalize the residuals.

    penfactor : float, optional
        A value greater than one. If the residual gap is reached, the penalty
        parameter will be multiplied or divided by `penfactor` to help
        normalize the residuals.

    relax : float, optional
        A value between zero and two. Relaxation replaces the quantity
        ``A(x)`` with ``relax*A(x) - (1 - relax)*(z + b)`` to improve
        convergence in some circumstances. A typical value to achieve
        over-relaxation would be between 1.5 and 1.8.


    Returns
    -------

    x : array, if `moreinfo` is ``False``
        The solution.

    d : dict, if `moreinfo` is ``True``
        A dictionary consisting of:

            x : array
                The solution.

            z : array
                The final value of the consensus variable, equaling x in the
                limit.

            y : array
                The final unscaled dual variable, equaling u/pen.

            numits : int
                The number of iterations completed.

            backtracks : int
                The number of backtracks used.

            r : array
                The final primal feasibility residual.

            s : array
                The final dual feasibility residual.

            stepsize : float
                The final step size.

            pen : float
                The final penalty parameter.

            hist : structured array
                The objective value, step size, penalty, primal and dual
                residual norms, primal and dual convergence thresholds, and
                relative error at each numbered iteration.


    Other Parameters
    ----------------

    {common_parameters}


    See Also
    --------

    admm, .proxgrad, ._proxgradaccel


    Notes
    -----

    It is often convenient to think of ADMM in terms of the consensus form
    of the optimization problem::

        minimize    F(x) + G(z)
        subject to  A(x) - b - z = 0.

    In this form, a second variable z is introduced with the constraint that
    z equal ``A(x) - b``. It is natural to then consider an optimization
    method that alternately optimizes in terms of x and z, with a correction
    to enforce the constraint. Linearized ADMM is such an iterative
    first-order optimization method where each iteration is composed of three
    steps:

        Prox step with respect to `F`::

            x_new = F.prox(x + stepsize/pen*Astar(2*u - u_old), stepsize)

        Prox step with respect to `G`::

            z_new = G.prox(A(x_new) - b + u, pen)

        Dual step for the constraint that ``A(x_new) - b == z_new``::

            u_new = u + A(x_new) - b - z_new

    Convergence is determined in terms of both primal and dual residuals:

        Primal feasibility residual::

            r = A(x_new) - b - z_new

        Dual feasibility residual::

            s = Astar(u_new - u)/pen + (x - x_new)/stepsize

    The basic algorithm is similar to the one described in section 4.5 of
    [1]_. Varying penalty parameter and over-relaxation suggested in [2]_.
    Interpretation of `stepsize` parameter as a step size and backtracking
    with respect to it is novel but similar to proximal gradient backtracking.
    See [3]_ for how this algorithm (there called Split Inexact Uzawa on SP_P)
    relates to similar algorithms like primal-dual hybrid gradient.


    References
    ----------

    .. [1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and
       Trends in Optimization, vol. 1, no. 3, pp. 123-231, 2013.

    .. [2] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein,
       "Distributed Optimization and Statistical Learning via the Alternating
       Direction Method of Multipliers," Found. Trends Mach. Learn., vol. 3,
       no. 1, pp. 1-122, Jan. 2011.

    .. [3] E. Esser, X. Zhang, and T. F. Chan, "A General Framework for a
       Class of First Order Primal-Dual Algorithms for Convex Optimization in
       Imaging Science," SIAM Journal on Imaging Sciences, vol. 3, no. 4, pp.
       1015-1046, Jan. 2010.

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
    z = A(x) - b  # needed for relaxation, makes Ax_relax = A(x) for first step

    tolnorm = l2norm
    bnorm = tolnorm(b)
    rabstol = abstol*tolnorm(np.ones_like(y0))
    sabstol = abstol*tolnorm(np.ones_like(x0))

    bts = 0

    for k in range(maxits):
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
        Axrelax = relax*Ax_new + one_minus_relax*(z + b)  # need norm later
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
            val = float(F(x) + G(z) + np.vdot(u/pen, Ax - b - z).real)
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
                hist[k//printrate] = tuple(dkt[ky] for ky in hist.dtype.names)
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
        return dict(x=x, z=z, y=u/pen, numits=k+1, backtracks=bts, r=r, s=s,
                    stepsize=stepsize, pen=pen, hist=hist[:k//printrate])
    else:
        return x
