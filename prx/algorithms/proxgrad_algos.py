# -----------------------------------------------------------------------------
# Copyright (c) 2014-2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------
"""Proximal gradient algorithms."""

from __future__ import division

import numpy as np

from . import base as _base
from ..fun.norms import l2norm, l2normsqhalf, linfnorm
from ._common import docstring_wrapper as _docstring_wrapper

__all__ = ('_proxgrad', '_proxgradaccel', 'ProxGrad', 'ProxGradAccel')


@_docstring_wrapper
def _proxgrad(F, G, A, Astar, b, x0, stepsize=1.0, backtrack=0.5, expand=1.25,
              reltol=1e-6, abstol=1e-10, maxits=10000,
              moreinfo=False, printrate=100, xstar=None):
    """Minimize ``F(x) + G(A(x) - b)`` using proximal gradient descent.

    The proximal gradient method is a useful basic algorithm for solving
    convex optimization problems with a mixed smooth and non-smooth objective
    function. This implementation includes an adaptive step size (backtracking
    and expansion) that helps improve convergence.

    The default parameters work best when the linear operator `A` is scaled to
    have an operator norm of one.


    Parameters
    ----------

    F : callable with ``prox`` method
        ``F(x)``, the possibly non-smooth part of the objective function,
        evaluated in terms of x directly. `F` must be "proximable", so that
        the callable ``F.prox(v, s)`` returns the x that minimizes

            ``F(x) + ( ||x - v||_2 )^2 / (2*s)``.

    G : callable with ``grad`` method
        ``G(z)``, the smooth part of the objective function, evaluated at
        ``z = A(x) - b``. `G` must have a gradient function callable as
        ``G.grad(z)``. As a function of x, the gradient of the G term is then:

            ``gradGx(x) = Astar(G.grad(A(x) - b))``.

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

    stepsize : float, optional
        Initial step size when backtracking is used, or constant step size for
        all iterations when backtracking is not used. Must be between 0 and
        2/L to guarantee convergence, where L is the Lipschitz constant of
        the gradient of ``G(A(x) - b)``.

    backtrack : float | ``None``, optional
        Backtracking multiplication factor between 0 and 1 that decreases the
        step size when the local Lipschitz condition is violated. If ``None``,
        no backtracking is used.

    expand : float | ``None``, optional
        Expansion multiplication factor greater than 1 that increases the step
        size after every iteration. This allows the step size to adapt to a
        decreasing local Lipschitz constant and improve convergence. If
        ``None``, no expansion is used.


    Returns
    -------

    x : array, if `moreinfo` is ``False``
        The solution.

    d : dict, if `moreinfo` is ``True``
        A dictionary consisting of:

            x : array
                The solution.

            numits : int
                The number of iterations completed.

            backtracks : int
                The number of backtracks used.

            r : array
                The final residual.

            stepsize : float
                The final step size.

            hist : structured array
                The objective value, step size, residual norm, convergence
                threshold, and relative error at each numbered iteration.


    Other Parameters
    ----------------

    {common_parameters}


    See Also
    --------

    _proxgradaccel, ._admmlin


    Notes
    -----

    Proximal gradient is an iterative first-order optimization method where
    each iteration is composed of two steps:

        Gradient step with respect to `G`::

            grad_new = Astar(G.grad(A(x) - b))
            z_new = x - stepsize*grad_new

        Prox step with respect to `F`::

            x_new = F.prox(z_new, stepsize)

    Convergence is determined in terms of the residual::

        r = (x - x_new)/stepsize + grad_new - grad

    The basic algorithm is described in section 4.2 of [1]_. Expanding
    backtracking line search is due to [2]_ and [3]_.


    References
    ----------

    .. [1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and
       Trends in Optimization, vol. 1, no. 3, pp. 123-231, 2013.

    .. [2] K. Scheinberg, D. Goldfarb, and X. Bai, "Fast First-Order
       Methods for Composite Convex Optimization with Backtracking,"
       pre-print, Apr. 2011.

    .. [3] S. R. Becker, E. J. Candes, and M. C. Grant, "Templates for
       convex cone problems with  applications to sparse signal recovery,"
       Mathematical Programming Computation, vol. 3, no. 3, pp. 165-218,
       Aug. 2011.

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

    for k in range(maxits):
        # gradient direction
        grad_new = Astar(gradG(Axmb))

        # loop for backtracking line search
        while True:
            # proximal gradient step
            x_new = proxF(x - stepsize*grad_new, stepsize)
            # need in backtracking test and re-use for gradient
            Axmb_new = A(x_new) - b

            if backtrack is None:
                # no backtracking specified
                break
            else:
                xmx = x_new - x
                gval = G(Axmb_new)
                bound = (
                    GAxmb + np.vdot(xmx, grad_new).real
                    + l2normsqhalf(xmx)/stepsize
                )
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
        GAxmb = G(Axmb)  # used in backtracking test + printing function value

        # norms for convergence check
        rnorm = tolnorm(r)
        xnorm = tolnorm(x)
        gradnorm = tolnorm(grad)

        stopthresh = rabstol + reltol*max(xnorm/stepsize, gradnorm)

        if printrate is not None and (k % printrate) == 0:
            val = float(F(x) + G(Axmb))
            dkt = dict(it=k, val=val, step=stepsize, resid=rnorm,
                       thresh=stopthresh)
            print(('{it}: val={val:.5}, step={step:.4}, ' +
                   'resid={resid:.4} ({thresh:.3})').format(**dkt))
            if moreinfo:
                if xstar is not None:
                    dkt['err'] = tolnorm(x_new - xstar)/tolnorm(xstar)
                else:
                    dkt['err'] = np.nan
                hist[k//printrate] = tuple(dkt[ky] for ky in hist.dtype.names)
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
        return dict(x=x, numits=k+1, backtracks=bts, r=r,
                    stepsize=stepsize, hist=hist[:k//printrate])
    else:
        return x


class ProxGrad(_base.AffineOptArgsMixin, _base.BaseIterativeAlgorithm):
    """Class for the proximal gradient algorithm.

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
    The proximal gradient method is a useful basic algorithm for solving
    convex optimization problems with a mixed smooth and non-smooth objective
    function of the form ``F(x) + G(A(x) - b)``. This implementation includes
    an adaptive step size (backtracking and expansion) that helps improve
    convergence. The default parameters work best when the linear operator `A`
    is scaled to have an operator norm of one.

    """

    _doc_algorithm_self = ':class:`.ProxGrad`'

    _doc_algorithm_see_also = ':class:`.ProxGradAccel`, :class:`.ADMMLin`'

    _doc_algorithm_notes = """
    Proximal gradient is an iterative first-order optimization method where
    each iteration is composed of two steps:

        Gradient step with respect to `G`::

            grad_new = Astar(gradG(A(x) - b))
            xp = x - step_size*grad_new

        Prox step with respect to `F`::

            x_new = proxF(xp, step_size)

    Convergence is determined in terms of the residual::

        r = (x_new - x) - step_size * (grad_new - grad)

    The basic algorithm is described in section 4.2 of [#PB14]_. Expanding
    backtracking line search is due to [#SGB14]_ and [#BCG11]_.

    """

    _doc_algorithm_references = """
    .. [#PB14] {PB14}

    .. [#SGB14] {SGB14}

    .. [#BCG11] {BCG11}

    """

    _doc_algorithm_objective_attributes = """
    x_ : array_like
        Value of the optimization variable, set after minimization has
        converged through :meth:`minimize` or :meth:`self.alg.iterate`.

    """

    _doc_algorithm_parameters = """
    step_size : float, optional
        Initial step size when backtracking is used, or constant step size
        for all iterations when backtracking is not used. Must be between 0
        and 2/L to guarantee convergence, where L is the Lipschitz constant
        of the gradient of ``G(A(x) - b)``.

    step_dir : callable, optional
        Function ``step_dir(x, grad)`` that returns a step direction given a
        point ``x`` and the gradient of `G` at that point. If ``None`` (the
        default), the gradient is used as the descent direction, i.e.
        ``step_dir(x, grad)`` returns ``grad``.

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
        self, objective, step_size=1.0, step_dir=None, backtrack=0.5,
        expand=1.25, **kwargs
    ):
        """."""
        super(ProxGrad, self).__init__(objective, **kwargs)
        self.step_size = step_size
        self.backtrack = backtrack
        self.expand = expand

        if step_dir is None:
            def step_dir(x, grad):
                """Return a step direction at `x` given the gradient `grad`."""
                return grad
        self.step_dir = step_dir

        self.print_str = (
            '{_iter}: val={_val:.5}, step={step_size:.4},'
            ' resid={_resid_nrm:.4} ({_resid_thresh:.3})'
        )

    def validate_params(self):
        """."""
        self.step_size = float(self.step_size)
        if self.step_size <= 0:
            raise ValueError('step_size must be positive')

        if not callable(self.step_dir):
            raise ValueError('step_dir must be a callable taking (x, grad)')

        if self.backtrack is not None:
            self.backtrack = float(self.backtrack)
            if self.backtrack <= 0 or self.backtrack >= 1:
                raise ValueError('backtrack must be None or between 0 and 1')

        if self.expand is not None:
            self.expand = float(self.expand)
            if self.expand <= 1:
                raise ValueError('expand must be None or greater than 1')

        # check for Objective compatibility
        for pname in ('F', 'proxF', 'G', 'gradG'):
            p = getattr(self.objective, pname)
            if p is None or not callable(p):
                errstr = 'self.objective.{0} must be set to a callable'
                raise ValueError(errstr.format(pname))

        return super(ProxGrad, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(ProxGrad, self).get_params(deep=deep)
        params.update(
            step_size=self.step_size, step_dir=self.step_dir,
            backtrack=self.backtrack, expand=self.expand,
        )
        return params

    def prepare(self, state, A=None, Astar=None, b=None):
        """."""
        return super(ProxGrad, self).prepare(state, A=A, Astar=Astar, b=b)

    def minimize(self, state, A=None, Astar=None, b=None):
        """."""
        return super(ProxGrad, self).minimize(
            state, A=A, Astar=Astar, b=b,
        )

    def iterate(self, state, A=None, Astar=None, b=None):
        """."""
        # validate parameters and arguments
        kwargs = self.prepare(state, A=A, Astar=Astar, b=b)
        A, Astar, b = (kwargs['A'], kwargs['Astar'], kwargs['b'])

        # get initial iterate value
        x0 = state['x']

        # set absolute tolerance threshold based on taking the tolerance norm
        # of a residual vector with all entries equal to abs_tol
        abs_tol_thresh = self.abs_tol * self.tol_norm(np.ones_like(x0))

        # initialize state
        z0 = A(x0) - b
        grad0 = Astar(self.objective.gradG(z0))
        step_dir0 = self.step_dir(x0, grad0)
        self.objective.state_ = dict(
            x=x0, z=z0, step_dir=step_dir0, step_size=self.step_size, _iter=0,
            _backtracks=0, _resid=np.full_like(x0, np.inf), _resid_nrm=np.inf,
            _resid_thresh=abs_tol_thresh,
        )
        # update with passed-in state
        self.objective.state_.update(state)

        yield self.objective.state_

        s = self.objective.state_
        for s['_iter'] in range(1, self.max_iter + 1):
            # get step direction
            grad = Astar(self.objective.gradG(s['z']))
            step_dir = self.step_dir(s['x'], grad)

            # inner loop for backtracking line search
            while True:
                # proximal gradient step
                x = self.objective.proxF(
                    s['x'] - s['step_size']*step_dir, s['step_size'],
                )
                z = A(x) - b  # needed to limit A evals as above
                xmx = x - s['x']

                if self.backtrack is None:
                    # no backtracking specified
                    break
                else:
                    gval = self.objective.G(z)
                    bound = (
                        self.objective.G(s['z']) + np.vdot(xmx, step_dir).real
                        + l2normsqhalf(xmx)/s['step_size']
                    )
                    # test Lipschitz bound, don't need to backtrack if it holds
                    if gval <= bound:
                        break
                    else:
                        # backtrack
                        s['step_size'] = s['step_size'] * self.backtrack
                        s['_backtracks'] += 1

            # residual for convergence check
            resid = xmx - s['step_size']*(step_dir - s['step_dir'])

            # norms for convergence check
            resid_nrm = self.tol_norm(resid)
            x_nrm = self.tol_norm(x)
            step_dir_nrm = self.tol_norm(step_dir)

            # threshold for convergence check
            resid_thresh = (
                abs_tol_thresh
                + self.rel_tol*min(x_nrm, s['step_size'] * step_dir_nrm)
            )

            # update state variables for which we had to track previous values
            s['step_dir'] = step_dir
            s['x'] = x
            s['z'] = z

            # update informational state variables
            s['_resid'] = resid
            s['_resid_nrm'] = resid_nrm
            s['_resid_thresh'] = resid_thresh

            # yield state at this iteration step
            yield s

            # check for convergence
            if resid_nrm < resid_thresh:
                break

            # expand stepsize and reset step ratio
            if self.expand is not None and self.backtrack is not None:
                s['step_size'] = s['step_size'] * self.expand

        # iterations have converged, store the resulting iterate
        self.objective.x_ = s['x']


@_docstring_wrapper
def _proxgradaccel(F, G, A, Astar, b, x0, stepsize=1.0, backtrack=0.5,
                   expand=1.25, restart=True, reltol=1e-6, abstol=1e-10,
                   maxits=10000, moreinfo=False, printrate=100,
                   xstar=None):
    """Minimize ``F(x) + G(A(x) - b)`` using accelerated proximal gradient.

    The accelerated proximal gradient method is a modified form of proximal
    gradient descent with a better theoretical convergence rate. It is useful
    for solving convex optimization problems with a mixed smooth and
    non-smooth objective function. This implementation includes an adaptive
    step size (backtracking and expansion) and an adaptive restart of the
    acceleration term that both help improve the convergence rate further.

    The default parameters work best when the linear operator `A` is scaled to
    have an operator norm of one.


    Parameters
    ----------

    F : callable with ``prox`` method
        ``F(x)``, the possibly non-smooth part of the objective function,
        evaluated in terms of x directly. `F` must be "proximable", so that
        the callable ``F.prox(v, s)`` returns the x that minimizes

            ``F(x) + ( ||x - v||_2 )^2 / (2*s)``.

    G : callable with ``grad`` method
        ``G(z)``, the smooth part of the objective function, evaluated at
        ``z = A(x) - b``. `G` must have a gradient function callable as
        ``G.grad(z)``. As a function of x, the gradient of the G term is then:

            ``gradGx(x) = Astar(G.grad(A(x) - b))``.

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

    b : np.ndarray
        Constant used in the G term of the objective function.

    x0 : array
        Initial iterate for x.

    stepsize : float, optional
        Initial step size when backtracking is used, or constant step size for
        all iterations when backtracking is not used. Must be between 0 and
        2/L to guarantee convergence, where L is the Lipschitz constant of
        the gradient of ``G(A(x) - b)``.

    backtrack : float | ``None``, optional
        Backtracking multiplication factor between 0 and 1 that decreases the
        step size when the local Lipschitz condition is violated. If ``None``,
        no backtracking is used.

    expand : float | ``None``, optional
        Expansion multiplication factor greater than 1 that increases the step
        size after every iteration. This allows the step size to adapt to a
        decreasing local Lipschitz constant and improve convergence. If
        ``None``, no expansion is used.

    restart : bool, optional
        If ``True``, use adaptive restart to reset the acceleration term when
        the acceleration direction opposes the previous step. If ``False``,
        never restart the acceleration term.


    Returns
    -------

    x : array, if `moreinfo` is ``False``
        The solution.

    d : dict, if `moreinfo` is ``True``
        A dictionary consisting of:

            x : array
                The solution.

            numits : int
                The number of iterations completed.

            backtracks : int
                The number of backtracks used.

            w : array
                The final iterate after the acceleration step.

            r : array
                The final residual.

            stepsize : float
                The final step size.

            hist : structured array
                The objective value, step size, residual norm, convergence
                threshold, and relative error at each numbered iteration.


    Other Parameters
    ----------------

    {common_parameters}


    See Also
    --------

    _proxgrad, ._admmlin


    Notes
    -----

    Accelerated proximal gradient is an iterative first-order optimization
    method where each iteration is composed of three steps:

        Acceleration::

            t = 0.5 + 0.5*sqrt(1 + 4*t_old**2)
            theta = (t_old - 1)/t
            w = x + theta*(x - x_old)

        Gradient step with respect to `G`::

            grad_new = Astar(gradG(A(w) - b))
            z_new = x - stepsize*grad_new

        Prox step with respect to `F`::

            x_new = proxF(z_new, stepsize)

    Convergence is determined in terms of the residual::

        r = (x - x_new)/stepsize + grad_new - grad

    The basic algorithm is described in section 4.3 of [1]_. Expanding
    backtracking line search is due to [2]_ and [3]_. Adaptive restart of the
    acceleration comes from [4]_.


    References
    ----------

    .. [1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and
       Trends in Optimization, vol. 1, no. 3, pp. 123-231, 2013.

    .. [2] K. Scheinberg, D. Goldfarb, and X. Bai, "Fast First-Order
       Methods for Composite Convex Optimization with Backtracking,"
       pre-print, Apr. 2011.

    .. [3] S. R. Becker, E. J. Candes, and M. C. Grant, "Templates for
       convex cone problems with  applications to sparse signal recovery,"
       Mathematical Programming Computation, vol. 3, no. 3, pp. 165-218,
       Aug. 2011.

    .. [4] B. O'Donoghue and E. Candes, "Adaptive Restart for Accelerated
       Gradient Schemes," Found Comput Math, pp. 1-18.

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
    w = x  # for adaptive restart check

    tolnorm = linfnorm
    rabstol = abstol*tolnorm(np.ones_like(x0))

    bts = 0

    for k in range(maxits):
        # "gradient" adaptive restart:
        # reset "momentum" when acceleration direction (x - x_old) is in
        # opposite direction of previous step prox gradient step (x - w)
        if restart and np.real(np.vdot(x - w, x - x_old)) < 0:
            t_old = 1

        # loop for backtracking line search
        while True:
            # acceleration
            t = 0.5 + 0.5*np.sqrt(1 + 4*gamma*t_old**2)
            theta = (t_old - 1)/t
            # restart acceleration if theta gets too small
            # (e.g. from big step size change)
            if theta < 0.1:
                theta, t_old = 0, 1
            w = (1 + theta) * x - theta * x_old
            # limit A evals by exploiting linearity
            Aw = (1 + theta) * Ax - theta * Ax_old
            Awmb = Aw - b  # re-use in backtracking test

            # proximal gradient step
            grad_new = Astar(gradG(Awmb))
            x_new = proxF(w - stepsize*grad_new, stepsize)
            Ax_new = A(x_new)  # needed to limit A evals as above
            Axmb = Ax_new - b  # need in backtracking test and printing
            xmw = x_new - w

            if backtrack is None:
                # no backtracking specified
                break
            else:
                gval = G(Axmb)
                bound = (
                    G(Awmb) + np.real(np.vdot(xmw, grad_new))
                    + l2normsqhalf(xmw)/stepsize
                )
                # test Lipschitz bound, don't need to backtrack if it holds
                if gval <= bound:
                    break
                else:
                    # backtrack
                    stepsize = stepsize*backtrack
                    gamma = stepsize_old/stepsize
                    bts += 1

        # residual for convergence check
        r = xmw - stepsize*(grad_new - grad)

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
        gradnorm = tolnorm(stepsize * grad)

        stopthresh = rabstol + reltol*min(xnorm, gradnorm)

        if printrate is not None and (k % printrate) == 0:
            val = float(F(x_new) + G(Axmb))
            dkt = dict(it=k, val=val, step=stepsize, resid=rnorm,
                       thresh=stopthresh)
            print(('{it}: val={val:.5}, step={step:.4}, ' +
                   'resid={resid:.4} ({thresh:.3})').format(**dkt))
            if moreinfo:
                if xstar is not None:
                    dkt['err'] = tolnorm(x_new - xstar)/tolnorm(xstar)
                else:
                    dkt['err'] = np.nan
                hist[k//printrate] = tuple(dkt[ky] for ky in hist.dtype.names)
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
        return dict(x=x, numits=k+1, backtracks=bts, w=w, r=r,
                    stepsize=stepsize, hist=hist[:k//printrate])
    else:
        return x


class ProxGradAccel(_base.AffineOptArgsMixin, _base.BaseIterativeAlgorithm):
    """Class for the accelerated proximal gradient algorithm.

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
    The accelerated proximal gradient algorithm in general minimizes objectives
    of the form ``F(x) + G(A(x) - b)``. It is a modified form of proximal
    gradient descent with a better theoretical convergence rate. It is useful
    for solving convex optimization problems with a mixed smooth and
    non-smooth objective function. This implementation includes an adaptive
    step size (backtracking and expansion) and an adaptive restart of the
    acceleration term that both help improve the convergence rate further. The
    default parameters work best when the linear operator `A` is scaled to have
    an operator norm of one.

    """

    _doc_algorithm_self = ':class:`.ProxGradAccel`'

    _doc_algorithm_see_also = ':class:`.ProxGrad`, :class:`.ADMMLin`'

    _doc_algorithm_notes = """
    Accelerated proximal gradient is an iterative first-order optimization
    method where each iteration is composed of three steps:

        Acceleration::

            t = 0.5 + 0.5*sqrt(1 + 4*t_old**2)
            theta = (t_old - 1)/t
            xp = x + theta*(x - x_old)

        Gradient step with respect to `G`::

            grad_new = Astar(G.grad(A(xp) - b))
            xpp = xp - step_size*grad_new

        Prox step with respect to `F`::

            x_new = F.prox(xpp, step_size)

    Convergence is determined in terms of the residual::

        r = (x_new - x) - step_size * (grad_new - grad)

    The basic algorithm is described in section 4.3 of [#PB14]_. Expanding
    backtracking line search is due to [#SGB14]_ and [#BCG11]_. Adaptive
    restart of the acceleration comes from [#OC13]_.

    """

    _doc_algorithm_references = """
    .. [#PB14] {PB14}

    .. [#SGB14] {SGB14}

    .. [#BCG11] {BCG11}

    .. [#OC13] {OC13}

    """

    _doc_algorithm_objective_attributes = """
    x_ : array_like
        Value of the optimization variable, set after minimization has
        converged through :meth:`minimize` or :meth:`self.alg.iterate`.

    """

    _doc_algorithm_parameters = """
    step_size : float, optional
        Initial step size when backtracking is used, or constant step size
        for all iterations when backtracking is not used. Must be between 0
        and 2/L to guarantee convergence, where L is the Lipschitz constant
        of the gradient of ``G(A(x) - b)``.

    step_dir : callable, optional
        Function ``step_dir(x, grad)`` that returns a step direction given a
        point ``x`` and the gradient of `G` at that point. If ``None`` (the
        default), the gradient is used as the descent direction, i.e.
        ``step_dir(x, grad)`` returns ``grad``.

    backtrack : float | ``None``, optional
        Backtracking multiplication factor between 0 and 1 that decreases
        the step size when the local Lipschitz condition is violated. If
        ``None``, no backtracking is used.

    expand : float | ``None``, optional
        Expansion multiplication factor greater than 1 that increases the
        step size after every iteration. This allows the step size to adapt
        to a decreasing local Lipschitz constant and improve convergence.
        If ``None``, no expansion is used.

    restart : bool, optional
        If ``True``, use adaptive restart to reset the acceleration term
        when the acceleration direction opposes the previous step. If
        ``False``, never restart the acceleration term.

    {algorithm_parameters}

    """

    def __init__(
        self, objective, step_size=1.0, step_dir=None, backtrack=0.5,
        expand=1.25, restart=True, **kwargs
    ):
        """."""
        super(ProxGradAccel, self).__init__(objective, **kwargs)
        self.step_size = step_size
        self.backtrack = backtrack
        self.expand = expand
        self.restart = restart

        if step_dir is None:
            def step_dir(x, grad):
                """Return a step direction at `x` given the gradient `grad`."""
                return grad
        self.step_dir = step_dir

        self.print_str = (
            '{_iter}: val={_val:.5}, step={step_size:.4},'
            ' resid={_resid_nrm:.4} ({_resid_thresh:.3})'
        )

    def validate_params(self):
        """."""
        self.step_size = float(self.step_size)
        if self.step_size <= 0:
            raise ValueError('step_size must be positive')

        if not callable(self.step_dir):
            raise ValueError('step_dir must be a callable taking (x, grad)')

        if self.backtrack is not None:
            self.backtrack = float(self.backtrack)
            if self.backtrack <= 0 or self.backtrack >= 1:
                raise ValueError('backtrack must be None or between 0 and 1')

        if self.expand is not None:
            self.expand = float(self.expand)
            if self.expand <= 1:
                raise ValueError('expand must be None or greater than 1')

        if not isinstance(self.restart, bool):
            raise ValueError('restart must either be True or False')

        # check for Objective compatibility
        for pname in ('F', 'proxF', 'G', 'gradG'):
            p = getattr(self.objective, pname)
            if p is None or not callable(p):
                errstr = 'self.objective.{0} must be set to a callable'
                raise ValueError(errstr.format(pname))

        return super(ProxGradAccel, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(ProxGradAccel, self).get_params(deep=deep)
        params.update(
            step_size=self.step_size, step_dir=self.step_dir,
            backtrack=self.backtrack, expand=self.expand,
            restart=self.restart,
        )
        return params

    def prepare(self, state, A=None, Astar=None, b=None):
        """."""
        return super(ProxGradAccel, self).prepare(state, A=A, Astar=Astar, b=b)

    def minimize(self, state, A=None, Astar=None, b=None):
        """."""
        return super(ProxGradAccel, self).minimize(
            state, A=A, Astar=Astar, b=b,
        )

    def iterate(self, state, A=None, Astar=None, b=None):
        """."""
        # validate parameters and arguments
        kwargs = self.prepare(state, A=A, Astar=Astar, b=b)
        A, Astar, b = (kwargs['A'], kwargs['Astar'], kwargs['b'])

        # get initial iterate value
        x0 = state['x']

        # set absolute tolerance threshold based on taking the tolerance norm
        # of a residual vector with all entries equal to abs_tol
        abs_tol_thresh = self.abs_tol * self.tol_norm(np.ones_like(x0))

        # initialize state
        z0 = A(x0) - b
        grad0 = Astar(self.objective.gradG(z0))
        step_dir0 = self.step_dir(x0, grad0)
        self.objective.state_ = dict(
            x=x0, z=z0, x_old=x0, z_old=z0, step_dir=step_dir0,
            step_size=self.step_size, step_ratio=1.0, t=1.0,
            _iter=0, _backtracks=0, _resid=np.full_like(x0, np.inf),
            _resid_nrm=np.inf, _resid_thresh=abs_tol_thresh,
        )
        # update with passed-in state
        self.objective.state_.update(state)

        yield self.objective.state_

        s = self.objective.state_
        for s['_iter'] in range(1, self.max_iter + 1):
            # inner loop for backtracking line search
            while True:
                # acceleration
                t = 0.5 + 0.5*np.sqrt(1 + 4*s['step_ratio']*s['t']**2)
                theta = (s['t'] - 1) / t
                # restart acceleration if theta gets too small
                # (e.g. from big step size change)
                if self.restart and theta < 0.1:
                    theta, s['t'] = 0, 1
                xp = (1 + theta) * s['x'] - theta * s['x_old']
                # limit A evals by exploiting linearity
                zp = (1 + theta) * s['z'] - theta * s['z_old']

                # proximal gradient step
                grad = Astar(self.objective.gradG(zp))
                step_dir = self.step_dir(xp, grad)
                x = self.objective.proxF(
                    xp - s['step_size']*step_dir, s['step_size'],
                )
                z = A(x) - b  # needed to limit A evals as above
                xmxp = x - xp

                if self.backtrack is None:
                    # no backtracking specified
                    break
                else:
                    gval = self.objective.G(z)
                    bound = (
                        self.objective.G(zp) + np.vdot(xmxp, step_dir).real
                        + l2normsqhalf(xmxp)/s['step_size']
                    )
                    # test Lipschitz bound, don't need to backtrack if it holds
                    if gval <= bound:
                        break
                    else:
                        # backtrack
                        s['step_size'] = s['step_size'] * self.backtrack
                        s['step_ratio'] = s['step_ratio'] / self.backtrack
                        s['_backtracks'] += 1

            # residual for convergence check
            resid = xmxp - s['step_size']*(step_dir - s['step_dir'])

            # norms for convergence check
            resid_nrm = self.tol_norm(resid)
            x_nrm = self.tol_norm(x)
            step_dir_nrm = self.tol_norm(step_dir)

            # threshold for convergence check
            resid_thresh = (
                abs_tol_thresh
                + self.rel_tol*min(x_nrm, s['step_size'] * step_dir_nrm)
            )

            # update state variables for which we had to track previous values
            s['t'] = t
            s['step_dir'] = step_dir
            s['x_old'] = s['x']
            s['z_old'] = s['z']
            s['x'] = x
            s['z'] = z

            # update informational state variables
            s['_resid'] = resid
            s['_resid_nrm'] = resid_nrm
            s['_resid_thresh'] = resid_thresh

            # yield state at this iteration step
            yield s

            # check for convergence
            if resid_nrm < resid_thresh:
                break

            # expand stepsize and reset step ratio
            if self.expand is not None and self.backtrack is not None:
                s['step_size'] = s['step_size'] * self.expand
                s['step_ratio'] = 1.0 / self.expand
            else:
                s['step_ratio'] = 1.0

            # "gradient" adaptive restart:
            # reset "momentum" when acceleration direction (x - x_old) is in
            # opposite direction of previous prox gradient step (x - xp)
            if (self.restart and
                    np.real(np.vdot(s['x'] - xp, s['x'] - s['x_old'])) < 0):
                s['t'] = 1

        # iterations have converged, store the resulting iterate
        self.objective.x_ = s['x']
