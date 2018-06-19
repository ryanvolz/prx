# ----------------------------------------------------------------------------
# Copyright (c) 2016, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

__all__ = ('docstring_wrapper')


def docstring_wrapper(fun):
    common = """reltol : float, optional
        Relative tolerance used on the residual for testing convergence. The
        algorithm stops when both tolerances are satisfied.

    abstol : float, optional
        Absolute tolerance used on the residual for testing convergence. The
        algorithm stops when both tolerances are satisfied.

    maxits : int, optional
        Maximum number of iterations to take before stopping regardless of
        convergence.

    moreinfo : bool, optional
        If ``False``, return only the solution. If ``True``, return a
        dictionary including the solution and other useful information.

    printrate : int | ``None``, optional
        Number of iterations between printed status updates. If ``None``, do
        not print algorithm status or final summary.

    xstar : ``None`` | array, optional
        If not ``None``, use `xstar` as the true solution to compute the
        relative error of the iterates. The result will be included in the
        returned dictionary when `moreinfo` is ``True``."""
    fun.__doc__ = fun.__doc__.format(common_parameters=common)
    return fun
