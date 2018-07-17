# ----------------------------------------------------------------------------
# Copyright (c) 2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Bibliographic references for optimization algorithms."""

__all__ = ('AlgorithmReferences', 'refs')


refs = dict(
    BT09="""A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
        Algorithm for Linear Inverse Problems," SIAM Journal on Imaging
        Sciences, vol. 2, no. 1, pp. 183-202, Jan. 2009.""",
    BCG11="""S. R. Becker, E. J. Candes, and M. C. Grant, "Templates for
        convex cone problems with  applications to sparse signal recovery,"
        Mathematical Programming Computation, vol. 3, no. 3, pp. 165-218,
        Aug. 2011.""",
    BPC_11="""S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein,
        "Distributed Optimization and Statistical Learning via the Alternating
        Direction Method of Multipliers," Found. Trends Mach. Learn., vol. 3,
        no. 1, pp. 1-122, Jan. 2011.""",
    EZC10="""E. Esser, X. Zhang, and T. F. Chan, "A General Framework for a
        Class of First Order Primal-Dual Algorithms for Convex Optimization in
        Imaging Science," SIAM Journal on Imaging Sciences, vol. 3, no. 4, pp.
        1015-1046, Jan. 2010.""",
    OC13="""B. O'Donoghue and E. Candes, "Adaptive Restart for Accelerated
        Gradient Schemes," Found Comput Math, pp. 1-18, 2013.""",
    PB14="""N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and
        Trends in Optimization, vol. 1, no. 3, pp. 123-231, 2013.""",
    SGB14="""K. Scheinberg, D. Goldfarb, and X. Bai, "Fast First-Order
        Methods for Composite Convex Optimization with Backtracking,"
        Found Comput Math, pp. 389-417, Mar. 2014.""",
)


class AlgorithmReferences(object):
    """Mixin class to add reference strings for optimization algorithms."""

    _docstring_subs = refs
