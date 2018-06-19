#-----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------
"""Norm functions.

.. currentmodule:: prx.fun.norms

.. autosummary::
    :toctree:

    l1norm
    l1l2norm
    l2norm
    l2normsqhalf
    linfnorm

"""

import numpy as np

__all__ = ['l1norm', 'l1l2norm', 'l2norm', 'l2normsqhalf', 'linfnorm']

def l1norm(x, axis=None, keepdims=False):
    """l1-norm: ``sum(abs(x))``


    Parameters
    ----------

    x : array
        Input array.

    axis : None | int | tuple of ints, optional
        Axis or axes over which to calculate the norm. If None, calculate the
        norm over all axes.

    keepdims : bool, optional
        If False, eliminate dimensions specified by `axis`. If True, keep
        these dimensions with length 1, so that the returned result can be
        broadcast against the input array.


    Returns
    -------

    value : float | array
        Norm value or array of values.


    See Also
    --------

    l1l2norm, l2norm, linfnorm

    """
    return np.abs(x).sum(axis=axis, keepdims=keepdims)

def l1l2norm(x, axis=-1):
    """Combined l1- and l2-norm: ``l1norm(l2norm(x, axis))``


    Parameters
    ----------

    x : array
        Input array.

    axis : int | tuple of ints, optional
        Axis or axes over which to calculate the l2-norm. The l1-norm is
        applied over all remaining axes.


    Returns
    -------

    value : float
        Norm value.


    See Also
    --------

    l1norm, l2norm

    """
    return np.sqrt((x.real**2 + x.imag**2).sum(axis=axis)).sum()

def l2norm(x, axis=None, keepdims=False):
    """l2-norm: ``sqrt(sum(abs(x)**2))``


    Parameters
    ----------

    x : array
        Input array.

    axis : None | int | tuple of ints, optional
        Axis or axes over which to calculate the norm. If None, calculate the
        norm over all axes.

    keepdims : bool, optional
        If False, eliminate dimensions specified by `axis`. If True, keep
        these dimensions with length 1, so that the returned result can be
        broadcast against the input array.


    Returns
    -------

    value : float | array
        Norm value or array of values.


    See Also
    --------

    l2normsqhalf, l1norm, linfnorm

    """
    return np.sqrt((x.real**2 + x.imag**2).sum(axis=axis, keepdims=keepdims))

def l2normsqhalf(x, axis=None, keepdims=False):
    """Half the square of the l2-norm: ``0.5*sum(abs(x)**2)``


    Parameters
    ----------

    x : array
        Input array.

    axis : None | int | tuple of ints, optional
        Axis or axes over which to calculate the norm. If None, calculate the
        norm over all axes.

    keepdims : bool, optional
        If False, eliminate dimensions specified by `axis`. If True, keep
        these dimensions with length 1, so that the returned result can be
        broadcast against the input array.


    Returns
    -------

    value : float | array
        Norm value or array of values.


    See Also
    --------

    l2norm

    """
    return 0.5*(x.real**2 + x.imag**2).sum(axis=axis, keepdims=keepdims)


def linfnorm(x, axis=None, keepdims=False):
    """linf-norm: ``max(abs(x))``


    Parameters
    ----------

    x : array
        Input array.

    axis : None | int | tuple of ints, optional
        Axis or axes over which to calculate the norm. If None, calculate the
        norm over all axes.

    keepdims : bool, optional
        If False, eliminate dimensions specified by `axis`. If True, keep
        these dimensions with length 1, so that the returned result can be
        broadcast against the input array.


    Returns
    -------

    value : float | array
        Norm value or array of values.


    See Also
    --------

    l1norm, l2norm

    """
    return np.abs(x).max(axis=axis, keepdims=keepdims)
