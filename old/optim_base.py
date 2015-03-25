#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from collections import namedtuple
from bunch import Bunch
#from functools import wraps

def itroutine(func):
    def wrapper(*args,**kw):
        gen = func(*args, **kw)
        ret = gen.next()
        return gen, ret
    wrapper.__name__ = func.__name__
    wrapper.__dict__ = func.__dict__
    wrapper.__doc__  = func.__doc__
    return wrapper

def Optiter(algo, coroutines):
    # initialize algorithm routine (progress to first yield) and get state
    state, obs = algo.next()
    state = (yield state)

    while True:
        # send state, algorithm produces new state and observables
        state, obs = algo.send((state, obs))
        # controlling loop monitors and assigns iteration number
        it = (yield state, obs)
        for cor in coroutines:
            # coroutines can operate on state and observables but not modify them
            cor.send(it, state, obs)

class IterOptions(namedtuple('IterOptions',
                             ['reltol', 'abstol', 'maxits', 'moreinfo',
                              'printrate', 'saverate'])):
    def __new__(cls, **kwargs):
        defaults = dict(reltol=1e-6, abstol=1e-10, maxits=10000,
                        moreinfo=False, printrate=100, saverate=None)
        defaults.update(kwargs)
        return super(IterOptions, cls).__new__(cls, **defaults)

class StateChange(Exception):
    def __init__(self, state):
        self.state = state
        super(StateChange, self).__init__(state)

class StoppingCriterion(Exception):
    def __init__(self, opts):
        self.opts = opts
        super(StoppingCriterion, self).__init__(opts)

class Observe(Exception):
    def __init__(self, it):
        self.it = it
        super(Observe, self).__init__(it)

def Runner(algo, opts=IterOptions()):
    state, printstr = algo.next()

    def runner(**state):
        state = algo.send(state)
        for it in xrange(opts.maxits):
            state = algo.next()

            crit = algo.raise(StoppingCriterion(opts))

            prnt = opts.printrate is not None and (it % opts.printrate) == 0
            save = opts.saverate is not None and (it % opts.saverate) == 0
            if prnt or save:
                obs = algo.raise(Observe(it))

                obs.update(crit)

                if prnt:
                    print(printstr.format(obs))
                if save:

            # check stopping criterion
            if crit.resid < crit.thresh:
                break

        if opts.moreinfo:
            return state
        else:
            return state.x

    return runner

def optim(maxits=10000, moreinfo=False, optiter, **state):
    # get default state
    state, obs = optiter.next()
    # initialize state
    state.update(state)

    try:
        for it in xrange(maxits):
            state, obs = optiter.send(it)
    except StopIteration:
        pass

    if moreinfo:
        return state.update(obs)
    else:
        return state.x

def continuer(optiter0, optiter1):
    try:
        for it in xrange(maxits):
            state, obs = optiter0.send(it)
    except StopIteration:
        pass

    # need a way to set state of optiter1 other than initialization?
    # - raise exception to notify of state change!
    state_old = state
    obs_old = obs
    #state, obs = optiter1.next()
    #state.update(state_old)
    try:
        for it in xrange(obs_old.it, obs_old.it + maxits):
            state, obs = optiter1.send(it)
    except StopIteration:
        pass

    return state.update(obs)

def ProxGrad(object):
    #state = Bunch(x=None, stepsize=1.0, _Axmb=None, _grad=None)
    State = namedtuple('State', ['x', 'stepsize', 'Axmb', 'grad'])

    printstr = ('{it}: val={val:.5}, step={step:.4}, ' +
                'resid={resid:.4} ({thresh:.3})')

    histdtype = [('it', np.int32), ('val', np.float64),
                 ('step', np.float64), ('resid', np.float64),
                 ('thresh', np.float64)]

    def __init__(self, F, G, A, Astar, b):
        self.F = F
        self.G = G
        self.A = A
        self.Astar = Astar
        self.b = b

    def loop(self):
        F = self.F
        G = self.G
        A = self.A
        Astar = self.Astar
        b = self.b

        proxF = F.prox
        gradG = G.grad

        x = None
        stepsize = 1.0
        Axmb = None
        grad = None

        while True:
            x, stepsize, Axmb, grad = (yield State(x, stepsize, Axmb, grad))

            # proximal gradient step
            x = proxF(x - stepsize*grad, stepsize)
            Axmb = A(x) - b # need for gradient, but also useful in backtracking
            grad = Astar(gradG(Axmb))

    def convergence(self):
        x, stepsize, Axmb, grad = yield

        while True:
            x_new, stepsize_new, Axmb_new, grad_new = yield
            r = (x - x_new)/stepsize_new + grad_new - grad

    def printer(self):
        F = self.F
        G = self.G

        while True:
            x, stepsize, Axmb, grad = yield
            val = F(x) + G(Axmb)
            dkt = dict(it=k, val=val, step=stepsize, resid=rnorm,
                        thresh=stopthresh)
            print(('{it}: val={val:.5}, step={step:.4}, ' +
                    'resid={resid:.4} ({thresh:.3})').format(**dkt))

def ProxGrad(F, G, A, Astar, b):
    proxF = F.prox
    gradG = G.grad

    # initialize state
    # Variables beginning with an underscore are derived state variables
    # that can be calculated from the non-underscore variables. These
    # are included because
    # All non-underscore variables that are None should be set by the
    # controlling function to which we yield
    S = Bunch(x=None, stepsize=1.0, _Axmb=None, _grad=None, _r=None)

    # default status string for printing, values populated by observables and
    # stopping criterion dictionaries
    printstr = ('{it}: val={val:.5}, step={step:.4}, ' +
                'resid={resid:.4} ({thresh:.3})')

    # default
    histdtype = [('it', np.int32), ('val', np.float64),
                 ('step', np.float64), ('resid', np.float64),
                 ('thresh', np.float64), ('err', np.float64)]
    S = (yield S)

    # make sure stepsize is a float
    S.stepsize = float(S.stepsize)

    # calculate derived state (convenience variables)
    S._Axmb = A(S.x) - b
    S._grad = Astar(gradG(Axmb)

    yield S

    while True:
        # proximal gradient step
        x = proxF(S.x - S.stepsize*S._grad, S.stepsize)
        Axmb = A(x) - b # need for gradient, but also useful in backtracking
        grad = Astar(gradG(Axmb))

        # residual for convergence check
        r = (S.x - x)/S.stepsize + grad - S._grad

        # update state variables
        S.x = x
        S._Axmb = Axmb
        S._grad = grad
        S._r = r

        # yield to controlling caller so it can operate on state
        while True:
            try:
                yield S
            except StateChange as e:
                Sn = e.state
                # re-calculate derived state
                Axmb = A(Sn.x) - b
                grad = Astar(gradG(Axmb))
                r = (S.x - Sn.x)/S.stepsize + grad - S._grad
                # assign new state
                S = Sn
                S._Axmb = Axmb
                S._grad = grad
                S._r = r

                yield S
            except StoppingCriterion as e:
                opts = e.opts

                rnorm = tolnorm(S._r)
                xnorm = tolnorm(S.x)
                gradnorm = tolnorm(S._grad)

                stopthresh = opts.abstol + opts.reltol*max(xnorm/stepsize, gradnorm)

                crit = Bunch(resid=rnorm, thresh=stopthresh)

                yield crit
            except Observe as e:
                it = e.it

                val = F(x) + G(Axmb)
                obs = Bunch(it=it, val=val, step=S.stepsize)

                yield obs
            else:
                # no exceptions raised, stop trying to yield current state
                break

def ProxGradBT(F, G, A, Astar, b, backtrack=0.5, expand=1.25):
    proxgrad = ProxGrad(F, G, A, Astar, b)
    S = proxgrad.next()
    S.bts = 0
    S._GAxmb = None
    S = (yield S)

    while True:
        Sn = proxgrad.send(S)

        xmx = Sn.x - S.x
        GAxmb = G(Sn._Axmb)
        bound = S._GAxmb + np.vdot(xmx, S._grad).real + l2normsqhalf(xmx)/Sn.stepsize
        # test Lipschitz bound, don't need to backtrack if it holds
        if GAxmb <= bound:
            Sn._GAxmb = GAxmb
            S = (yield Sn)
        else:
            # backtrack
            S.stepsize = S.stepsize*backtrack
            S.bts += 1
            S = (yield S)

        # expand stepsize
        if expand is not None and backtrack is not None:
            S.stepsize = S.stepsize*expand
