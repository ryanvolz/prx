# ----------------------------------------------------------------------------
# Copyright (c) 2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Base optimizer algorithm classes."""

from six import with_metaclass

from ..docstring_helpers import DocstringSubstituteMeta
from ..fun.norms import linfnorm

__all__ = ('BaseOptAlgorithm', 'BaseIterativeAlgorithm')


class BaseOptAlgorithm(with_metaclass(DocstringSubstituteMeta, object)):
    """Base class for optimization algorithms.

    Subclasses should implement :meth:`minimize` and add any necessary
    parameters to :meth:`__init__` and the parametrization methods.

    Parameter handling is designed to be compatible with scikit-learn
    estimators when used as part of a :class:`.BaseObjective` object.

    This uses the DocstringSubstituteMeta metaclass so that class attribute
    strings starting with '_doc_' can be used to specify substitution values to
    be used in docstrings. The name following the '_doc_' prefix is used as the
    substitution variable name. In addition, '_doc_' class attribute strings
    are inherited from the most recent value in classes specified by the
    '_docstring_bases' list attribute followed by all parent classes. Any
    inherited values can also be used as substitution strings in the class's
    own '_doc_' attribute strings. String substitution is carried out for all
    class and method docstrings using normal `string.format` rules.

    Finally, the entire pre-substitution docstring of a parent method or
    classe can be inherited by using '.' as the entire docstring.

    """

    _doc_state_argument = """
    state : dict
        State dictionary, with entries varying by objective. The state
        defines a point at which the objective is evaluated.

    """

    _doc_keyword_arguments = """
    kwargs : dict
        Keyword arguments, varying by objective or algorithm. These are often
        associated with data and/or a measurement model.

    """

    _doc_algorithm_objective_parameter = """
    objective : :class:`.BaseObjective`
        Objective object that defines the objective function to be minimized,
        including an :meth:`evaluate` method and attributes that may be
        required by a particular algorithm.

    """

    _doc_algorithm_self = ':class:`.BaseOptAlgorithm`'
    _doc_algorithm_description = ''
    _doc_algorithm_notes = ''
    _doc_algorithm_references = ''
    _doc_algorithm_see_also = ''
    _doc_algorithm_parameters = ''
    _doc_algorithm_attributes = ''
    _doc_algorithm_objective_attributes = ''

    def __init__(self, objective, **kwargs):
        """Initialize optimization algorithm.

        Parameters
        ----------

        {algorithm_objective_parameter}

        """
        self.objective = objective
        if kwargs:
            errstr = 'Unrecognized argument to __init__: {0}.'
            raise TypeError(errstr.format(kwargs))

    def validate_params(self):
        """Validate the current set of parameters.

        Returns
        -------

        self : :class:`BaseOptAlgorithm`

        """
        return self

    def get_params(self, deep=True):
        """Get the current set of parameters.

        Parameters
        ----------

        deep : boolean, optional
            No effect. Included for compatibility with scikit-learn estimators.


        Returns
        -------

        params : dict
            Parameter names mapped to their values.

        """
        params = {}
        return params

    def set_params(self, **params):
        """Set parameters post-initialization.

        Returns
        -------

        self : :class:`BaseOptAlgorithm`

        """
        valid_params = self.get_params()
        for parameter, value in params.items():
            if parameter not in valid_params:
                errstr = 'Invalid parameter name {0} for Algorithm {1}.'
                raise ValueError(errstr.format(parameter, self))
            setattr(self, parameter, value)
        return self

    def minimize(self, state, **kwargs):
        """Minimize the objective function given an initial state.

        Parameters
        ----------

        {state_argument}

        {keyword_arguments}

        Returns
        -------

        self : :class:`BaseOptAlgorithm`

        """
        raise NotImplementedError

    def evaluate(self, state):
        """Return the objective value given a state.

        Parameters
        ----------

        {state_argument}


        Returns
        -------

        val : float
            The objective value.

        """
        return self.objective.evaluate(state)


class BaseIterativeAlgorithm(BaseOptAlgorithm):
    """Base class for iterative optimization algorithms.

    Subclasses can rely on the initialization parameters included in this class
    and include them in docstrings by using the {{algorithm_parameters}}
    substitution variable.

    Subclasses should implement the :meth:`iterate` method, override
    :meth:`validate_params` and :meth:`get_params`, and add additional
    parameters and set the :attr:`print_str` attribute in :meth:`__init__`.

    """

    _doc_algorithm_self = ':class:`.BaseIterativeAlgorithm`'

    _doc_algorithm_parameters = """
    rel_tol : float, optional
        Relative tolerance used on the residual for testing convergence.
        The optimization stops when both tolerances are satisfied.

    abs_tol : float, optional
        Absolute tolerance used on the residual for testing convergence.
        The optimization stops when both tolerances are satisfied.

    tol_norm : callable, optional
        Function evaluating a norm that is to be used on the residual for
        comparison with the tolerance parameters.

    max_iter : int, optional
        Maximum number of iterations to take before stopping regardless of
        convergence.

    print_period : int | ``None``, optional
        Number of iterations between printed status updates. If ``None``,
        do not print algorithm status or final summary.

    {algorithm_parameters}

    """

    _doc_algorithm_attributes = """
    print_str : string
        String, including format-style keyword substitutions taken from the
        iteration's `state` dictionary, that is printed as the status update
        every `print_period` iterations.

    {algorithm_attributes}

    """

    def __init__(
        self, objective, rel_tol=1e-6, abs_tol=1e-10, tol_norm=linfnorm,
        max_iter=10000, print_period=100, **kwargs
    ):
        """Initialize iterative optimization algorithm.

        Parameters
        ----------

        {algorithm_objective_parameter}

        {algorithm_parameters}

        """
        super(BaseIterativeAlgorithm, self).__init__(objective, **kwargs)
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.tol_norm = tol_norm
        self.max_iter = max_iter
        self.print_period = print_period

        self.print_str = '{_iter}: val={_val:.5}'

    def validate_params(self):
        """."""
        self.rel_tol = float(self.rel_tol)
        if self.rel_tol < 0:
            raise ValueError('rel_tol must be >= 0')

        self.abs_tol = float(self.abs_tol)
        if self.abs_tol < 0:
            raise ValueError('abs_tol must be >= 0')

        self.max_iter = int(self.max_iter)
        if self.max_iter <= 0:
            raise ValueError('max_iter must be a positive integer')

        if self.print_period is not None:
            self.print_period = int(self.print_period)
            if self.print_period <= 0:
                errstr = 'print_period must be a positive integer or None'
                raise ValueError(errstr)

        return super(BaseIterativeAlgorithm, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(BaseIterativeAlgorithm, self).get_params(deep=deep)
        params.update(
            rel_tol=self.rel_tol, abs_tol=self.abs_tol, max_iter=self.max_iter,
            print_period=self.print_period,
        )
        return params

    def minimize(self, state, **kwargs):
        """."""
        # we only validate parameters just before minimizing to allow for
        # efficiency in initialization and multiple set_params calls
        self.objective.validate_params()

        # basic minimization: just loop through the iteration steps and print
        # a status update at a given period
        for state in self.iterate(state, **kwargs):
            if (self.print_period is not None
                    and (state['_iter'] % self.print_period) == 0):
                self.evaluate(state)
                print(self.print_str.format(**state))

        if self.print_period is not None:
            if state['_iter'] >= self.max_iter:
                msg = 'Failed to converge'
            else:
                msg = 'Converged'
            msg += ' after {0} iterations'.format(state['_iter'])
            if getattr(self, 'backtrack', None) is not None:
                msg += ' (and {0} backtracks)'.format(state['_backtracks'])
            print(msg)

        # return the minimized objective with state_ attribute set
        return self.objective

    def iterate(self, state, **kwargs):
        """Create generator that yields the `state` dictionary after each step.

        You can inspect the `state` dictionary for useful information to
        customize each iteration. Variables in the `state` dictionary that
        are not prefixed with an underscore can be modified and will affect the
        subsequent iteration step.

        """
        raise NotImplementedError
