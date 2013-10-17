# Copyright 2013 Ryan Volz

# This file is part of spyalg.

# Spyalg is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Spyalg is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with spyalg.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import numpy as np

from .func_ops import *
from .grad_funcs import grad_l2sqhalf
from .prox_funcs import (proj_l2, proj_linf, proj_zeros, prox_l1, 
                         prox_l1l2, prox_l2, prox_l2sqhalf)
from .norms import l1norm, l1l2norm, l2norm, l2normsqhalf, linfnorm

__all__ = ['L1Norm', 'L1L2Norm', 'L2Norm', 'L2NormSqHalf', 'L2BallInd', 
           'LInfBallInd', 'ZerosInd']

def prepend_docstring(parent):
    def decorator(f):
        if f.__doc__ is None:
            f.__doc__ = parent.__doc__
        else:
            f.__doc__ = parent.__doc__ + f.__doc__
        return f
    
    return decorator

###****************************************************************************
### Base classes for function-prox objects ************************************
###****************************************************************************
class FunctionWithGradProx(object):
    """Function with gradient and prox operator.
    
    The function is evaluated by calling obj(x) or obj.fun(x).
    Its gradient is evaluated by calling obj.grad(x).
    Its prox operator is evaluated by calling obj.prox(x, lmbda).
    
    The prox operator gives the solution to the problem:
        prox(v, lmbda) = argmin_x ( g(x) + 1/(2*lmbda)*(||x - v||_2)**2 ).
    
    """
    def __init__(self, scale=None, stretch=None, shift=None, linear=None, 
                 const=None):
        """Create function with gradient and prox operator.
        
        g = FunctionWithGradProx(scale=s, stretch=a, shift=b, linear=c, const=d) 
        for the function f(x) creates an object for evaluating 
        the value, gradient, and prox operator of:
            g(x) = s*f(a*x + b) + <c, x> + d
        
        """
        # change 'None's to identities
        if scale is None:
            scale = 1
        if stretch is None:
            stretch = 1
        if shift is None:
            shift = 0
        if linear is None:
            linear = 0
        if const is None:
            const = 0
        
        self._scale = scale
        self._stretch = stretch
        self._shift = shift
        self._linear = linear
        self._const = const
        # modify functions to apply parameters
        # (order of operations is important!)
        if shift is not None and np.any(shift != 0):
            self.fun = shift_fun(self.fun, shift)
            self.grad = shift_grad(self.grad, shift)
            self.prox = shift_prox(self.prox, shift)
        if stretch is not None and stretch != 1:
            self.fun = stretch_fun(self.fun, stretch)
            self.grad = stretch_grad(self.grad, stretch)
            self.prox = stretch_prox(self.prox, stretch)
        if scale is not None and scale != 1:
            self.fun = scale_fun(self.fun, scale)
            self.grad = scale_grad(self.grad, scale)
            self.prox = scale_prox(self.prox, scale)
        if linear is not None and np.any(linear != 0):
            self.fun = addlinear_fun(self.fun, linear)
            self.grad = addlinear_grad(self.grad, linear)
            self.prox = addlinear_prox(self.prox, linear)
        if const is not None and np.any(const != 0):
            self.fun = addconst_fun(self.fun, const)
            # added constant does not change grad or prox functions
    
    def __call__(self, x):
        return self.fun(x)
    
    @property
    def conjugate(self):
        """Get object for the conjugate function.
        
        The convex conjugate of f(x) is defined as 
            f^*(y) = sup_x ( <y, x> - f(x) ).
        
        Additionally, if g(x) = s*f(a*x + b) + <c, x> + d, then
            g^*(y) = s*f^*(y/(a*s) - c/(a*s)) - <b/a, y> - d.
        
        """
        Conjugate = self._conjugate_class
        return Conjugate(**self._conjugate_args)
    
    @property
    def _conjugate_class(self):
        """Return the class for the conjugate function."""
        raise NotImplementedError
        
    @property
    def _conjugate_args(self):
        """Return the keyword arguments for the conjugate function in a dict."""
        scale = self._scale
        stretch = 1/(self._scale*self._stretch)
        shift = -stretch*self._linear
        linear = -self._shift/self._stretch
        const = -self._const
        
        return dict(scale=scale, stretch=stretch, shift=shift, linear=linear, 
                    const=const)
    
    def fun(self, x):
        raise NotImplementedError
    
    def grad(self, x):
        raise NotImplementedError
    
    def prox(self, x, lmbda=1):
        raise NotImplementedError
    
    @property
    def const(self):
        return self._const
    
    @property
    def linear(self):
        return self._linear
    
    @property
    def scale(self):
        return self._scale
    
    @property
    def shift(self):
        return self._shift
    
    @property
    def stretch(self):
        return self._stretch

class LinearFunctionWithGradProx(FunctionWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, scale=None, stretch=None, shift=None, linear=None, 
                 const=None):
        # we can eliminate stretching and shifting:
        # s*f(a*x + b) + <c, x> + d ==> a*s*f(x) + <c, x> + (s*f(b) + d)
        
        # change 'None's to identities
        if scale is None:
            scale = 1
        if stretch is None:
            stretch = 1
        if shift is None:
            shift = 0
        if const is None:
            const = 0
        # absorb stretch into scale
        scale = scale*stretch
        stretch = None
        # absorb shift into const
        const = const + scale*self.fun(shift)
        shift = None
        # turn identities into Nones
        if scale == 1:
            scale = None
        if const == 0:
            const = None
        
        super(LinearFunctionWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift, 
            linear=linear, const=const)

class NormFunctionWithGradProx(FunctionWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, scale=None, stretch=None, shift=None, linear=None, 
                 const=None):
        # we can eliminate stretching:
        # s*f(a*x + b) + <c, x> + d ==> a*s*f(x + b/a) + <c, x> + d
        
        # absorb stretch into scale and shift
        if stretch is not None:
            if shift is not None:
                shift = shift/stretch
            if scale is not None:
                scale = scale*stretch
            else:
                scale = stretch
            stretch = None
        
        super(NormFunctionWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift, 
            linear=linear, const=const)

class NormSqFunctionWithGradProx(FunctionWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, scale=None, stretch=None, shift=None, linear=None, 
                 const=None):
        # we can eliminate stretching:
        # s*f(a*x + b) + <c, x> + d ==> (a**2)*s*f(x + b/a) + <c, x> + d
        
        # absorb stretch into scale and shift
        if stretch is not None:
            if shift is not None:
                shift = shift/stretch
            if scale is not None:
                scale = scale*stretch**2
            else:
                scale = stretch**2
            stretch = None
        
        super(NormSqFunctionWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift, 
            linear=linear, const=const)

class IndicatorWithGradProx(FunctionWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(FunctionWithGradProx.__init__)
    def __init__(self, radius=1, scale=None, stretch=None, shift=None, 
                 linear=None, const=None):
        # we can eliminate scaling:
        # s*f(a*x + b) + <c, x> + d ==> f(a*x + b) + <c, x> + d
        
        # eliminate scale
        if scale is not None:
            scale = None
        
        super(IndicatorWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift, 
            linear=linear, const=const)

class NormBallWithGradProx(IndicatorWithGradProx):
    __doc__ = FunctionWithGradProx.__doc__
    @prepend_docstring(IndicatorWithGradProx.__init__)
    def __init__(self, radius=1, scale=None, stretch=None, shift=None, 
                 linear=None, const=None):
        """The parameter 'radius' sets the radius of the norm ball indicator.
        
        """
        # we can eliminate stretching:
        # s*f_r(a*x + b) + <c, x> + d ==> s*f_(r/a)(x + b/a) + <c, x> + d
        
        # absorb stretch into radius and shift
        if stretch is not None:
            radius = radius/stretch
            if shift is not None:
                shift = shift/stretch
            stretch = None
        
        # set radius parameter
        self._radius = radius
        
        # we can also eliminate scaling, 
        # but this is taken care of by parent class
        super(NormBallWithGradProx, self).__init__(
            scale=scale, stretch=stretch, shift=shift, 
            linear=linear, const=const)
    
    @property
    def radius(self):
        return self._radius


###****************************************************************************
### Useful classes for function-prox objects **********************************
###****************************************************************************
class L1Norm(NormFunctionWithGradProx):
    __doc__ = NormFunctionWithGradProx.__doc__ + \
    """
    Function and prox operator for the l1-norm, sum(abs(x)).
    
    fun(x) = sum(abs(x))
    
    The prox operator is soft thresholding:
        st(x[k], lmbda) = { (1 - lmbda/|x[k]|)*x[k]  if |x[k]| > lmbda
                          {           0              otherwise
    
    """
    @property
    def _conjugate_class(self):
        return LInfBallInd
    fun = staticmethod(l1norm)
    prox = staticmethod(prox_l1)
    
class L1L2Norm(NormFunctionWithGradProx):
    __doc__ = NormFunctionWithGradProx.__doc__ + \
    """
    Function and prox operator for the combined l1- and l2-norm.
    
    The axis argument defines the axes over which to take the l2-norm
    (axis=None specifies all axes and is equivalent to L2Norm).
    
    fun(x) = l1norm(l2norm(x, axis))
    
    The prox operator is block soft thresholding (for 'k' NOT along axis):
      bst(x[k, :], lmbda) = {(1 - lmbda/||x[k]||_2)*x[k]  if ||x[k]||_2 > lmbda
                            {          0                  otherwise
    
    """
    @prepend_docstring(NormFunctionWithGradProx.__init__)
    def __init__(self, axis=-1, scale=None, stretch=None, shift=None, 
                 linear=None, const=None):
        """The axis argument defines the axes over which to take the l2-norm
        (axis=None specifies all axes and is equivalent to L2Norm).
        
        """
        self._axis = axis
        
        super(L1L2Norm, self).__init__(
            scale=scale, stretch=stretch, shift=shift, 
            linear=linear, const=const)
    
    @property
    def axis(self):
        return self._axis
    
    def fun(self, x):
        """Combined l1- and l2-norm with l2-norm taken over axis=self.axis."""
        return l1l2norm(x, self._axis)
        
    def prox(self, x, lmbda=1):
        """Prox operator of combined l1- and l2-norm (l2 over axis=self.axis)."""
        return prox_l1l2(x, lmbda=lmbda, axis=self._axis)

class L2Norm(NormFunctionWithGradProx):
    __doc__ = NormFunctionWithGradProx.__doc__ + \
    """
    Function and prox operator for the l2-norm, sqrt(sum(abs(x)**2)).
    
    fun(x) = sqrt(sum(abs(x)**2))
    
    The prox operator is the block thresholding function for block=(all of x):
        bst(x, lmbda) = { (1 - lmbda/||x||_2)*x  if ||x||_2 > lmbda
                        {           0            otherwise
    
    """
    @property
    def _conjugate_class(self):
        return L2BallInd
    fun = staticmethod(l2norm)
    prox = staticmethod(prox_l2)

class L2NormSqHalf(NormSqFunctionWithGradProx):
    __doc__ = NormSqFunctionWithGradProx.__doc__ + \
    """
    Function with gradient and prox operator for half the squared l2-norm.
    
    fun(x) = 0.5*sum(abs(x)**2))
    
    The prox operator is the shrinkage function:
        shrink(x, lmbda) = x/(1 + lmbda)
    
    """
    @property
    def _conjugate_class(self):
        return L2NormSqHalf
    fun = staticmethod(l2normsqhalf)
    grad = staticmethod(grad_l2sqhalf)
    prox = staticmethod(prox_l2sqhalf)

class L2BallInd(NormBallWithGradProx):
    __doc__ = NormBallWithGradProx.__doc__ + \
    """
    Function and prox operator for the indicator of the l2-ball.
    
    The indicator function is zero for vectors inside the ball, infinity for 
    vectors outside the ball.
    
    The prox operator is Euclidean projection onto the l2-ball.
    
    """
    @property
    def _conjugate_class(self):
        return L2Norm
    
    def fun(self, x):
        """Indicator function for the l2-ball with radius=self.radius."""
        nrm = l2norm(x)
        eps = np.finfo(nrm.dtype).eps
        rad = self.radius
        if nrm <= rad*(1 + 10*eps):
            return 0
        else:
            return np.inf
    
    def prox(self, x, lmbda=1):
        """Projection onto the l2-ball with radius=self.radius."""
        return proj_l2(x, radius=self.radius)

class LInfBallInd(NormBallWithGradProx):
    __doc__ = NormBallWithGradProx.__doc__ + \
    """
    Function and prox operator for the indicator of the linf-ball.
    
    The indicator function is zero for vectors inside the ball, infinity for 
    vectors outside the ball.
    
    The prox operator is Euclidean projection onto the linf-ball.
    
    """
    @property
    def _conjugate_class(self):
        return L1Norm
    
    def fun(self, x):
        """Indicator function for the linf-ball with radius=self.radius."""
        nrm = linfnorm(x)
        eps = np.finfo(nrm.dtype).eps
        rad = self.radius
        if nrm <= rad*(1 + 10*eps):
            return 0
        else:
            return np.inf
    
    def prox(self, x, lmbda=1):
        """Projection onto the linf-ball with radius=self.radius."""
        return proj_linf(x, radius=self.radius)

class ZerosInd(IndicatorWithGradProx):
    __doc__ = IndicatorWithGradProx.__doc__ + \
    """
    Function and prox operator for the indicator of zero elements.
    
    The indicator function is zero for vectors with only zeros in the 
    specified places, infinity if any of the required zero entries are nonzero.
    
    The prox operator is Euclidean projection onto the set with 
    specified zeros (x[z] is set to 0).
    
    """
    @prepend_docstring(IndicatorWithGradProx.__init__)
    def __init__(self, z=None, scale=None, stretch=None, shift=None, 
                 linear=None, const=None):
        """The parameter 'z' must be a boolean array giving the zero locations.
        
        """
        if z is None:
            raise ValueError('Must specify zero locations (z)!')
        
        # stretch can be eliminated by bringing into shift
        # (since multiplying does not change zero locations)
        if stretch is not None and shift is not None:
            shift = shift/stretch
        stretch = None
        
        self._z = z
        
        # we can also eliminate scaling, 
        # but this is taken care of by parent class
        super(ZerosInd, self).__init__(
            scale=scale, stretch=stretch, shift=shift, 
            linear=linear, const=const)
    
    @property
    def _conjugate_class(self):
        return ZerosInd
    
    @property
    def _conjugate_args(self):
        # The conjugate of the zero indicator is the indicator for the
        # complimentary set of zeros.
        z = ~self._z
        kwargs = super(ZerosInd, self)._conjugate_args
        kwargs.update(z=z)
        return kwargs
    
    @property
    def z(self):
        return self._z
    
    def fun(self, x):
        """Indicator function for zero elements z."""
        if np.any(x[self._z] != 0):
            return np.inf
        else:
            return 0
        
    def prox(self, x, lmbda=1):
        """Projection onto the set with specified zeros (x[z] is set to 0)"""
        return proj_zeros(x, z=self._z)