# ----------------------------------------------------------------------------
# Copyright (c) 2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

from functools import wraps

def backends(*algorithms):
    """Decorator for specifying algorithms used for an optimization problem."""
    algnames = [a.__name__ for a in algorithms]
    algos = dict(zip(algnames, algorithms))
    def problem_decorator(f):
        @wraps(f)
        def algorithm(*args, **kwargs):
            # match solver kwarg to available algorithms
            solvername = kwargs.pop('solver', None)
            if solvername is None:
                solvername = algorithms[0].__name__
            try:
                solver = algos[solvername]
            except KeyError:
                s = '{0} is not an available solver for {1}'
                s.format(solvername, f.__name__)
                raise ValueError(s)

            # get algorithm arguments from problem function
            algargs, algkwargs = f(*args, **kwargs)

            return solver(*algargs, **algkwargs)

        solvers = '``\'' + '\'`` | ``\''.join(algnames) + '\'``'
        seealso = '.' + ', .'.join(algnames)
        algorithm.__doc__ = algorithm.__doc__.format(
            solvers=solvers, seealso=seealso,
        )
        algorithm.algorithms = algos
        return algorithm
    return problem_decorator
