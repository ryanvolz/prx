# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Linear operator classes.

.. currentmodule:: prx.operator_class

.. autosummary::
    :toctree:

    LinearOperator
    DiagLinop
    MatrixLinop
    FixedSizeLinop

"""

import numpy as np

__all__ = [
    'LinearOperator', 'DiagLinop', 'MatrixLinop', 'FixedSizeLinop'
]

class LinearOperator(object):
    """Linear operator A.

    This class defines a linear operator through its forward and adjoint
    operations. The forward operation is called using the `forward` method,
    while the adjoint operation is called using the `adjoint` method.

    For specifying a particular operator, initialize this class with the
    forward and adjoint functions as arguments. Alternatively, you can inherit
    from this class and override the `_forward` and `_adjoint` methods.

    .. automethod:: __init__


    Attributes
    ----------

    Adjoint : LinearOperator
        Object corresponding to the adjoint operator (i.e., the forward and
        adjoint functions are swapped).

    """
    def __init__(self, forward=None, adjoint=None):
        """Create a linear operator with the given forward/adjoint functions.


        Parameters
        ----------

        forward : function
            A(x), forward operator taking a single argument. If an
            optional `out` keyword argument is accepted, it is used to specify
            the output array.

        adjoint : function
            Astar(y), adjoint operator taking a single argument. If an
            optional `out` keyword argument is accepted, it is used to specify
            the output array.

        """
        if forward is not None:
            self._forward = forward
        if adjoint is not None:
            self._adjoint = adjoint

    def _forward(self, x):
        raise NotImplementedError

    def _adjoint(self, y):
        raise NotImplementedError

    def forward(self, x, out=None):
        """Calculate the forward operation, A(x).


        Parameters
        ----------

        x : ndarray
            Input to forward operation A.

        out : ndarray
            Optional pre-allocated array for storing the output.


        Returns
        -------

        out : ndarray
            Output of the forward operation A.


        Notes
        -----

        Supplying an `out` argument is only useful for avoiding output memory
        allocation when the specified forward operator itself accepts an
        `out` argument.

        """
        if out is not None:
            try:
                out = self._forward(x, out=out)
            except TypeError:
                out[...] = self._forward(x)
            return out
        return self._forward(x)

    __call__ = forward

    def adjoint(self, y, out=None):
        """Calculate the adjoint operation, A*(y).


        Parameters
        ----------

        y : ndarray
            Input to adjoint operation A*.

        out : ndarray
            Optional pre-allocated array for storing the output.


        Returns
        -------

        out : ndarray
            Output of the adjoint operation A*.


        Notes
        -----

        Supplying an `out` argument is only useful for avoiding output memory
        allocation when the specified adjoint operator itself accepts an
        `out` argument.

        """
        if out is not None:
            try:
                out = self._adjoint(y, out=out)
            except TypeError:
                out[...] = self._adjoint(y)
            return out
        return self._adjoint(y)

    # short aliases
    A = forward
    As = adjoint
    H = adjoint

    @property
    def Adjoint(self):
        return LinearOperator(self._adjoint, self._forward)

class DiagLinop(LinearOperator):
    """Diagonal linear operator A(x) = s*x (element-wise multiplication).

    .. automethod:: __init__

    """
    def __init__(self, s):
        """Create a diagonal linear operator A(x) = s*x.


        Parameters
        ----------

        s : float, int, or ndarray
            Scalar or array defining the diagonal linear operator.

        """
        self._s = s
        self._sconj = np.conj(s)

        super(DiagLinop, self).__init__()

    def _forward(self, x, out=None):
        return np.multiply(self._s, x, out)

    def _adjoint(self, y, out=None):
        return np.multiply(self._sconj, y, out)

class MatrixLinop(LinearOperator):
    """Matrix linear operator A(x) = np.dot(M, x).

    .. automethod:: __init__

    """
    def __init__(self, M):
        """Create a matrix linear operator A(x) = np.dot(M, x).


        Parameters
        ----------

        M : matrix or ndarray
            Matrix defining the linear operator.

        """
        self._M = np.ascontiguousarray(M)
        self._Ms = np.ascontiguousarray(np.conj(np.transpose(M)))

        super(MatrixLinop, self).__init__()

    def _forward(self, x, out=None):
        return np.dot(self._M, x, out)

    def _adjoint(self, y, out=None):
        return np.dot(self._Ms, y, out)

class FixedSizeLinop(LinearOperator):
    """Fixed size linear operator A.

    This class defines a linear operator through its forward and adjoint
    operations. The forward operation is called using the `forward` method,
    while the adjoint operation is called using the `adjoint` method.

    For specifying a particular operator, inherit from this class and override
    the `_forward` and `_adjoint` methods.

    .. automethod:: __init__


    Attributes
    ----------

    inshape, outshape : tuple
        Tuples giving the shape of the input and output arrays of the forward
        operation (output and input arrays of the adjoint operation),
        respectively.

    indtype, outdtype : dtype
        Dtypes of the input and output arrays of the forward operation
        (output and input arrays of the adjoint operation), respectively.

    """
    def __init__(self, inshape, indtype, outshape, outdtype):
        """Create a linear operator for the given input and output specs.


        Parameters
        ----------

        inshape, outshape : tuple
            Tuples giving the shape of the input and output arrays of the
            forward operation (output and input arrays of the adjoint
            operation), respectively.

        indtype, outdtype : dtype
            Dtypes of the input and output arrays of the forward operation
            (output and input arrays of the adjoint operation), respectively.

        """
        self.inshape = inshape
        self.indtype = indtype
        self.outshape = outshape
        self.outdtype = outdtype

        super(FixedSizeLinop, self).__init__()

    def _forward(self, x, out):
        raise NotImplementedError

    def _adjoint(self, y, out):
        raise NotImplementedError

    def forward(self, x, out=None):
        """Calculate the forward operation, A(x).


        Parameters
        ----------

        x : ndarray of shape `inshape` and dtype `indtype`
            Input to forward operation A.

        out : ndarray of shape `outshape` and dtype `outdtype`
            Optional pre-allocated array for storing the output.


        Returns
        -------

        out : ndarray of shape `outshape` and dtype `outdtype`
            Output of the forward operation A.

        """
        if out is None:
            out = np.empty(self.outshape, self.outdtype)
        return self._forward(x, out)

    def adjoint(self, y, out=None):
        """Calculate the adjoint operation, A*(y).


        Parameters
        ----------

        y : ndarray of shape `outshape` and dtype `outdtype`
            Input to adjoint operation A*.

        out : ndarray of shape `outshape` and dtype `outdtype`
            Optional pre-allocated array for storing the output.


        Returns
        -------

        out : ndarray of shape `inshape` and dtype `indtype`
            Output of the adjoint operation A*.

        """
        if out is None:
            out = np.empty(self.inshape, self.indtype)
        return self._adjoint(y, out)
