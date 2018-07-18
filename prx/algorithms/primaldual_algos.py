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

from . import base as _base
from ..fun.norms import l2norm, l2normsqhalf
from ._common import docstring_wrapper as _docstring_wrapper

__all__ = ('_pdhg', 'PDHG')


@_docstring_wrapper
def _pdhg(F, G, A, Astar, b, x0, y0=None, step_p=1.0, step_d=1.0,
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


class PDHG(_base.BaseIterativeAlgorithm):
    """Class for the Primal Dual Hybrid Gradient (PDHG) algorithm.

    {algorithm_description}


    Attributes
    ----------

    {algorithm_attributes}


    See Also
    --------

    {algorithm_see_also}


    Notes
    -----

    {algorithm_notes}


    References
    ----------

    {algorithm_references}

    """

    _doc_algorithm_description = """
    Primal Dual Hybrid Gradient (PDHG) solves convex optimization problems with
    a split objective of the form ``F(x) + G(A(x) - b)``. It is often
    non-trivial to accommodate a linear transformation when evaluating the prox
    operator of most functions, so its explicit inclusion in the algorithm
    itself can bring a large benefit. This implementation includes a adaptive
    step sizes that help improve convergence.

    """

    _doc_algorithm_self = ':class:`.PDHG`'

    _doc_algorithm_see_also = ':class:`.ADMMLin`, :class:`.ProdGradAccel`'

    _doc_algorithm_notes = """
    The basic algorithm is described in [#EZC10]_, along with a description of
    how this algorithm relates to similar algorithms like ADMM and linearized
    ADMM.

    """

    _doc_algorithm_references = """
    .. [#EZC10] {EZC10}

    """

    _doc_algorithm_objective_attributes = """
    x_ : array_like
        Value of the optimization variable, set after minimization has
        converged through :meth:`minimize` or :meth:`self.alg.iterate`.

    """

    _doc_initial_state_argument = """
    state : dict
        Initial state dictionary containing:

            x : array_like
                Initial value for the optimization variable.

            y : array_like, optional
                Initial value for `y`, the dual variable.

    """

    _doc_keyword_arguments = """
    A : callable
        ``A(x)`` is a linear operator, used in the `G` term of the objective
        function: ``G(A(x) - b)``. Although not checked, it must obey the
        linearity condition

            ``A(a*x + b*y) == a*A(x) + b*A(y)``.

    Astar : callable
        ``Astar(z)``, the adjoint operator of `A`. By definition, `Astar`
        satisfies

            ``vdot(A(x), z) == vdot(x, Astar(z))``

        for all x, z and the inner product ``vdot``. If, for instance, `A`
        represented multiplication by a matrix M, `Astar` would then
        represent multiplcation by the complex conjugate transpose of M.

    b : np.ndarray
        Constant used in the `G` term of the objective function:
        ``G(A(x) - b)``.

    """

    _doc_algorithm_parameters = """
    step_size_p : float, optional
        Initial primal step size when backtracking is used, or constant primal
        step size for all iterations when backtracking is not used. Must be
        between 0 and ``penalty/opnorm(A)**2`` to guarantee convergence, where
        ``opnorm`` is the l2-induced operator norm of `A`.

    step_size_d : float, optional
        Initial dual step size when backtracking is used, or constant dual
        step size for all iterations when backtracking is not used. Must be
        between 0 and ``penalty/opnorm(A)**2`` to guarantee convergence, where
        ``opnorm`` is the l2-induced operator norm of `A`.

    backtrack : float | ``None``, optional
        Backtracking multiplication factor between 0 and 1 that decreases
        the step size when the local Lipschitz condition is violated. If
        ``None``, no backtracking is used.

    expand : float | ``None``, optional
        Expansion multiplication factor greater than 1 that increases the
        step size after every iteration. This allows the step size to adapt
        to a decreasing local Lipschitz constant and improve convergence.
        If ``None``, no expansion is used.

    {algorithm_parameters}

    """

    def __init__(
        self, objective, step_size_p=1.0, step_size_d=1.0, backtrack=0.5,
        expand=1.25, **kwargs
    ):
        """."""
        super(PDHG, self).__init__(objective, **kwargs)
        self.step_size_p = step_size_p
        self.step_size_d = step_size_d
        self.backtrack = backtrack
        self.expand = expand

        self.print_str = (
            '{_iter}: val={_val:.5}, step_p={step_size_p:.4},'
            ' step_d={step_size_d:.4},'
            '\n\tresid_p={_resid_p_nrm:.4} ({_resid_p_thresh:.3}),'
            ' resid_d={_resid_d_nrm:.4} ({_resid_d_thresh:.3})'
        )

    def validate_params(self):
        """."""
        self.step_size_p = float(self.step_size_p)
        if self.step_size_p <= 0:
            raise ValueError('step_size_p must be positive')

        self.step_size_d = float(self.step_size_d)
        if self.step_size_d <= 0:
            raise ValueError('step_size_d must be positive')

        if self.backtrack is not None:
            self.backtrack = float(self.backtrack)
            if self.backtrack <= 0 or self.backtrack >= 1:
                raise ValueError('backtrack must be None or between 0 and 1')

        if self.expand is not None:
            self.expand = float(self.expand)
            if self.expand <= 1:
                raise ValueError('expand must be None or greater than 1')

        # check for Objective compatibility
        for pname in ('F', 'proxF', 'G', 'proxGconj'):  # , 'Fconj', 'Gconj'):
            p = getattr(self.objective, pname)
            if p is None or not callable(p):
                errstr = 'self.objective.{0} must be set to a callable'
                raise ValueError(errstr.format(pname))

        return super(PDHG, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(PDHG, self).get_params(deep=deep)
        params.update(
            step_size=self.step_size, backtrack=self.backtrack,
            expand=self.expand, penalty=self.penalty, resid_gap=self.resid_gap,
            penalty_factor=self.penalty_factor, relax=self.relax,
        )
        return params

    def minimize(self, state, A, Astar, b):
        """."""
        return super(PDHG, self).minimize(state, A=A, Astar=Astar, b=b)

    def iterate(self, state, A, Astar, b):
        """."""
        # get initial iterate value
        try:
            x0 = state['x']
        except KeyError:
            errstr = (
                'Keyword arguments for state must include an initial value for'
                ' x.'
            )
            raise ValueError(errstr)

        try:
            y0 = state['y']
        except KeyError:
            y0 = np.zeros_like(A(x0))

        # set absolute tolerance threshold based on taking the tolerance norm
        # of a residual vector with all entries equal to abs_tol
        abs_tol_p_thresh = self.abs_tol * self.tol_norm(np.ones_like(x0))
        abs_tol_d_thresh = self.abs_tol * self.tol_norm(np.ones_like(y0))

        theta_p = 1
        theta_d = 0
        self.backtrack = None

        # initialize state
        Axmb0 = A(x0) - b
        z0 = Axmb0
        Asy0 = Astar(y0)
        self.objective.state_ = dict(
            x=x0, Axmb=Axmb0, z=z0, y=y0, y_old=y0, Asy=Asy0, Asy_old=Asy0,
            h=Axmb0-z0,
            step_size_p=self.step_size_p, step_size_d=self.step_size_d,
            _iter=0, _backtracks=0,
            _resid_p=np.full_like(x0, np.inf), _resid_p_nrm=np.inf,
            _resid_d=np.full_like(y0, np.inf), _resid_d_nrm=np.inf,
            _resid_p_thresh=abs_tol_p_thresh, _resid_d_thresh=abs_tol_d_thresh,
        )
        # update with passed-in state
        self.objective.state_.update(state)

        yield self.objective.state_

        s = self.objective.state_
        for s['_iter'] in range(1, self.max_iter + 1):
            # inner loop for backtracking line search
            while True:
                # primal update
                # acceleration
                ybar = (1 + theta_d) * s['y'] - theta_d * s['y_old']
                Asybar = (1 + theta_d) * s['Asy'] - theta_d * s['Asy_old']
                # gradient descent step wrt Lagrange multiplier term
                xhat = s['x'] - s['step_size_p'] * Asybar
                # prox step
                x = self.objective.proxF(xhat, s['step_size_p'])
                Axmb = A(x) - b

                if self.backtrack is None:
                    # no backtracking specified
                    break
                else:
                    plhs = (
                        self.objective.G(Axmb) + self.objective.Gconj(ybar)
                        - np.vdot(ybar, Axmb).real
                    )
                    prhs = l2normsqhalf(x - s['x']) / s['step_size_p']
                    # test Lipschitz bound, don't need to backtrack if it holds
                    if plhs <= prhs:
                        break
                    else:
                        # backtrack
                        s['step_size_p'] = s['step_size_p'] * self.backtrack
                        s['_backtracks'] += 1

            # inner loop for backtracking line search
            while True:
                # dual update
                # acceleration
                xbar = (1 + theta_p) * x - theta_p * s['x']
                Axmbbar = (1 + theta_p) * Axmb - theta_p * s['Axmb']
                # gradient descent step wrt Lagrange multiplier term
                yhat = s['y'] - s['step_size_d'] * Axmbbar
                # prox step
                y = self.objective.proxGconj(yhat, s['step_size_d'])
                Asy = Astar(y)

                if self.backtrack is None:
                    # no backtracking specified
                    break
                else:
                    dlhs = (
                        self.objective.Fconj(-Asy) + self.objective.F(xbar)
                        - np.vdot(xbar, -Asy).real
                    )
                    drhs = l2normsqhalf(y - s['y']) / s['step_size_d']
                    # test Lipschitz bound, don't need to backtrack if it holds
                    if dlhs <= drhs:
                        break
                    else:
                        # backtrack
                        s['step_size_d'] = s['step_size_d'] * self.backtrack
                        s['_backtracks'] += 1

            # calculate residuals
            resid_p = (
                (s['x'] - x) / s['step_size_p'] - (s['Asy'] - Asy)
                - theta_d * (s['Asy'] - s['Asy_old'])
            )
            resid_d = (
                (s['y'] - y) / s['step_size_d'] - theta_p * (s['Axmb'] - Axmb)
            )

            # norms for convergence check
            resid_p_nrm = self.tol_norm(resid_p)
            resid_d_nrm = self.tol_norm(resid_d)
            x_nrm = self.tol_norm(x)
            Axmb_nrm = self.tol_norm(Axmb)
            y_nrm = self.tol_norm(y)
            Asy_nrm = self.tol_norm(Asy)

            # thresholds for convergence check
            resid_p_thresh = (
                abs_tol_p_thresh
                + self.rel_tol * min(x_nrm / s['step_size_p'], Asy_nrm)
            )
            resid_d_thresh = (
                abs_tol_d_thresh
                + self.rel_tol * min(y_nrm / s['step_size_d'], Axmb_nrm)
            )

            # update state variables
            s['x'] = x
            s['Axmb'] = Axmb
            s['y_old'] = s['y']
            s['y'] = y
            s['Asy_old'] = s['Asy']
            s['Asy'] = Asy

            s['z'] = Axmb
            s['h'] = resid_p

            # update informational state variables
            s['_resid_p'] = resid_p
            s['_resid_d'] = resid_d
            s['_resid_p_nrm'] = resid_p_nrm
            s['_resid_d_nrm'] = resid_d_nrm
            s['_resid_p_thresh'] = resid_p_thresh
            s['_resid_d_thresh'] = resid_d_thresh

            # yield state at this iteration step
            yield s

            # check for convergence
            # can't calculate dual function value, so best stopping criterion
            # is to see if primal and dual feasibility residuals are small
            resid_p_ratio = resid_p_nrm / resid_p_thresh
            resid_d_ratio = resid_d_nrm / resid_d_thresh
            if resid_p_ratio < 1 and resid_d_ratio < 1:
                break

            # expand stepsize
            if self.expand is not None and self.backtrack is not None:
                s['step_size_p'] = s['step_size_p'] * self.expand
                s['step_size_d'] = s['step_size_d'] * self.expand

        # iterations have converged, store the resulting iterate
        self.objective.x_ = s['x']
