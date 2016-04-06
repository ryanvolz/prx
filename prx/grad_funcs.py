# ----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import division
import numpy as np

__all__ = ['grad_l2sqhalf']

def grad_l2sqhalf(x):
    """Gradient of half the squared l2-norm: 0.5*(||x||_2)**2.

    The gradient function is the identity:
        I(x) = x.

    """
    return x
