# ----------------------------------------------------------------------------
# Copyright (c) 2015, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

"""Linear operator classes."""

try:
    from inspect import getfullargspec as _getfullargspec
except ImportError:
    from inspect import getargspec as _getfullargspec

import functools

import numpy as np

from six import with_metaclass

from .docstring_helpers import DocstringSubstituteMeta

__all__ = (
    'DiagLinop', 'LinearOperator', 'MatrixLinop',
)


class BaseLinearOperator(with_metaclass(DocstringSubstituteMeta, object)):
    """{summary}

    {description}


    Attributes
    ----------

    {attributes}

    """

    _doc_summary = 'Linear operator A.'

    _doc_description = """
    This class defines a linear operator through its forward and adjoint
    operations. The forward operation is called using the :meth:`forward`
    method, while the adjoint operation is called using the :meth:`adjoint`
    method.

    """

    _doc_attributes = """
    H : :class:`.LinearOperator`
        Object corresponding to the adjoint operator (i.e., the forward and
        adjoint functions are swapped).

    inshape : tuple
        Tuple giving the shape of the input array for the forward operation
        and output array for the adjoint operation. May be `None` if unknown or
        shape is not fixed.

    indtype : dtype
        Data type of the input array for the forward operation and output array
        for the adjoint operation. May be `None` if unknown or dtype is not
        fixed.

    outshape : tuple
        Tuple giving the shape of the output array for the forward operation
        and input array for the adjoint operation. May be `None` if unknown or
        shape is not fixed.

    outdtype : dtype
        Data type of the output array for the forward operation and input array
        for the adjoint operation. May be `None` if unknown or dtype is not
        fixed.

    """

    _doc_parameters = """
    inshape : tuple
        Tuple giving the shape of the input array for the forward operation
        and output array for the adjoint operation. May be `None` if unknown or
        shape is not fixed.

    indtype : dtype
        Data type of the input array for the forward operation and output array
        for the adjoint operation. May be `None` if unknown or dtype is not
        fixed.

    outshape : tuple
        Tuple giving the shape of the output array for the forward operation
        and input array for the adjoint operation. May be `None` if unknown or
        shape is not fixed.

    outdtype : dtype
        Data type of the output array for the forward operation and input array
        for the adjoint operation. May be `None` if unknown or dtype is not
        fixed.

    """

    def __init__(
        self, inshape=None, indtype=None, outshape=None, outdtype=None,
        **kwargs
    ):
        """Initialize a linear operator object.

        Parameters
        ----------

        {parameters}

        """
        self._inshape = inshape
        self._indtype = indtype
        self._outshape = outshape
        self._outdtype = outdtype
        if kwargs:
            errstr = 'Unrecognized argument to __init__: {0}.'
            raise TypeError(errstr.format(kwargs))

    _doc_forward_doc = """Calculate the forward operation, ``A(x)``.

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

    Depending on the implementation of the specific operator, supplying an
    `out` argument may not always avoid output memory allocation.

    """

    def forward(self, x, out=None):
        """{forward_doc}"""
        raise NotImplementedError

    def A(self, x, out=None):
        """{forward_doc}"""
        return self.forward(x, out)

    def __call__(self, x, out=None):
        """{forward_doc}"""
        return self.forward(x, out)

    _doc_adjoint_doc = """Calculate the adjoint operation, ``A*(y)``.

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

    Depending on the implementation of the specific operator, supplying an
    `out` argument may not always avoid output memory allocation.

    """

    def adjoint(self, y, out=None):
        """{adjoint_doc}"""
        raise NotImplementedError

    def As(self, y, out=None):
        """{adjoint_doc}"""
        return self.adjoint(y, out)

    @property
    def H(self):
        """Object corresponding to the adjoint operator."""
        return LinearOperator(
            forward=self.adjoint, adjoint=self.forward,
            inshape=self.outshape, indtype=self.outdtype,
            outshape=self.inshape, outdtype=self.indtype,
        )

    @property
    def inshape(self):
        """Tuple giving the shape of the input array for the forward op."""
        return self._inshape

    @property
    def indtype(self):
        """Data type of the input array for the forward op."""
        return self._indtype

    @property
    def outshape(self):
        """Tuple giving the shape of the output array for the forward op."""
        return self._outshape

    @property
    def outdtype(self):
        """Data type of the output array for the forward op."""
        return self._outdtype


class DiagLinop(BaseLinearOperator):
    """."""

    _doc_summary = \
        'Diagonal linear operator ``A(x) = s*x`` (element-wise mult.).'

    _doc_attributes = """
    s : float | int | ndarray
        Scalar or array defining the diagonal linear operator.

    {attributes}

    """

    _doc_parameters = """
    s : float | int | ndarray
        Scalar or array defining the diagonal linear operator.

    {parameters}

    """

    def __init__(self, s, **kwargs):
        """."""
        super(DiagLinop, self).__init__(**kwargs)
        test_in = np.empty(self.inshape, self.indtype)
        test_out = np.empty(self.outshape, self.outdtype)
        try:
            np.broadcast(s, test_in)
            np.broadcast(s, test_out)
        except ValueError:
            errstr = (
                's must have shape that is broadcastable with inshape and'
                ' outshape (which must be the same)'
            )
            raise ValueError(errstr)
        self._s = s
        self._sconj = np.conj(s)

    def forward(self, x, out=None):
        """."""
        return np.multiply(self._s, x, out)

    def adjoint(self, y, out=None):
        """."""
        return np.multiply(self._sconj, y, out)

    @property
    def s(self):
        """Scalar or array defining the diagonal linear operator."""
        return self._s


class LinearOperator(BaseLinearOperator):
    """."""

    _doc_summary = \
        'Linear operator built from supplied forward and adjoint functions.'

    _doc_parameters = """
    forward : function
        A(x), forward operator taking a single argument. If an
        optional `out` keyword argument is accepted, it is used to specify
        the output array.

    adjoint : function
        Astar(y), adjoint operator taking a single argument. If an
        optional `out` keyword argument is accepted, it is used to specify
        the output array.

    {parameters}

    """

    def __init__(self, forward, adjoint, **kwargs):
        """."""
        super(LinearOperator, self).__init__(**kwargs)
        self._forward = self._normalize_op_function(
            forward, self._outshape, self._outdtype,
        )
        self._adjoint = self._normalize_op_function(
            adjoint, self._inshape, self._indtype,
        )

    @staticmethod
    def _normalize_op_function(fun, shape, dtype):
        """Return function based on fun that has signature f(x, out=None)."""
        argspec = _getfullargspec(fun)
        if 'self' in argspec.args and argspec.args[0] == 'self':
            argspec.args.pop(0)
        nargs = len(argspec.args)
        if nargs == 0 or nargs > 2:
            errstr = 'Operator function {0} must accept 1 or 2 arguments.'
            raise ValueError(errstr.format(fun))
        elif nargs == 1:
            # function does not have an out argument, wrap so it does
            @functools.wraps(fun)
            def fun_with_out(x, out=None):
                if out is None:
                    return fun(x)
                else:
                    out[...] = fun(x)
                    return out
            return fun_with_out

        # now we have nargs=2, determine if second is required or not
        if argspec.defaults is not None:
            ndefaults = len(argspec.defaults)
        else:
            ndefaults = 0
        npargs = nargs - ndefaults

        if npargs <= 1:
            # second argument is not required, fun is good as is
            return fun

        # fun requires a second argument, wrap so it doesn't
        if shape is None or dtype is None:
            errstr = (
                'Either the operator function {0} must have an optional second'
                ' (output) argument or the shape and dtype of its output must'
                ' be specified.'
            )
            raise ValueError(errstr.format(fun))

        @functools.wraps(fun)
        def fun_with_optional_out(x, out=None):
            if out is None:
                out = np.empty(shape, dtype)
            return fun(x, out)
        return fun_with_optional_out

    def forward(self, x, out=None):
        """."""
        return self._forward(x, out)

    def adjoint(self, y, out=None):
        """."""
        return self._adjoint(y, out)


class MatrixLinop(BaseLinearOperator):
    """."""

    _doc_summary = 'Matrix linear operator ``A(x) = np.matmul(M, x)``.'

    _doc_parameters = """
    M : matrix | ndarray
        Matrix defining the linear operator.

    """

    def __init__(self, M):
        """."""
        self._M = np.ascontiguousarray(M)
        self._Ms = np.ascontiguousarray(np.conj(np.transpose(M)))
        Mshape = M.shape
        if len(Mshape) == 1:
            Mshape = (1,) + Mshape
        inshape = Mshape[:-2] + Mshape[-1:]
        outshape = Mshape[:-2] + Mshape[-2:-1]
        super(MatrixLinop, self).__init__(
            inshape=inshape, indtype=M.dtype,
            outshape=outshape, outdtype=M.dtype,
        )

    def forward(self, x, out=None):
        """."""
        return np.matmul(self._M, x, out)

    def adjoint(self, y, out=None):
        """."""
        return np.matmul(self._Ms, y, out)

    @property
    def M(self):
        """Matrix defining the linear operator."""
        return self._M
