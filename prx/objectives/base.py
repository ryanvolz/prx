# ----------------------------------------------------------------------------
# Copyright (c) 2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Base optimization objective classes."""

from six import with_metaclass

from ..algorithms import base as _algbase
from ..docstring_helpers import DocstringSubstituteMeta

__all__ = ('ObjectiveAlgorithmMeta', 'BaseObjective')


class ObjectiveAlgorithmMeta(DocstringSubstituteMeta):
    """Metaclass for setting the Algorithm associated with an Objective.

    Objective classes are associated with a particular optimization Algorithm
    class that is used to implement its `minimize` method. The associated
    Algorithm class is set with the `_alg_cls` class attribute. This metaclass
    initializes the `_alg_cls` attribute, if not present, to `BaseOptAlgorithm`
    and causes docstrings substitutions from `_alg_cls` to be inherited.
    Additionally, this provides :meth:`with_alg` and :meth:`__add__`
    metaclass methods for copying an existing Objective class but with
    `_alg_cls` set to a new Algorithm class.

    """

    def __new__(cls, name, bases, dct):
        """Create a new class of type ObjectiveAlgorithmMeta."""
        # save original dct for recreating class with new alg
        if '_orig_args' not in dct:
            orig_args = (name, bases, dct.copy())
            dct['_orig_args'] = orig_args

        # get _alg_cls attribute or default and ensure it is set
        alg_cls = dct.get('_alg_cls', None)
        if alg_cls is None:
            for base in bases:
                alg_cls = getattr(base, '_alg_cls', None)
                if alg_cls is not None:
                    break
            else:
                alg_cls = _algbase.BaseOptAlgorithm
            dct['_alg_cls'] = alg_cls
        # add alg_cls to _docstring_bases for docstring substitution
        dct.setdefault('_docstring_bases', []).append(alg_cls)

        return super(ObjectiveAlgorithmMeta, cls).__new__(
            cls, name, bases, dct,
        )

    def __init__(self, name, bases, dct):
        """Initialize a class of type ObjectiveAlgorithmMeta."""
        orig_name, orig_bases, orig_dct = self._orig_args
        if '_orig_args' not in orig_dct:
            # update orig_bases to include self, which is the original class
            # this allows super() to work by making all recreated classes
            # a subclass of the original class
            orig_bases = (self,) + orig_bases
            # set the updated orig_args to orig_dct and self
            orig_args = (orig_name, orig_bases, orig_dct)
            orig_dct['_orig_args'] = orig_args
            self._orig_args = orig_args
        super(ObjectiveAlgorithmMeta, self).__init__(name, bases, dct)

    def __add__(self, other):
        """Return a duplicate Objective subclass, but using `other` Algorithm.

        Parameters
        ----------

        other : :class:`BaseOptAlgorithm`
            Algorithm class to associate with the duplicate Objective subclass.


        Returns
        -------

        cls : type
            New subclass that duplicates the current class except for a
            different associated Algorithm class.

        """
        if issubclass(other, _algbase.BaseOptAlgorithm):
            return self.with_alg(other)
        else:
            return NotImplemented

    __radd__ = __add__

    def with_alg(self, alg_cls, name=None):
        """Return a duplicate Objective subclass, but using `alg_cls` instead.

        Parameters
        ----------

        alg_cls : :class:`BaseOptAlgorithm`
            Algorithm class to associate with the duplicate Objective subclass.

        name : str, optional
            Name to assign to the new subclass. If None, the Algorithm's name
            will be appended to the Objective's name after two underscores to
            get the new subclass name.


        Returns
        -------

        cls : type
            New subclass that duplicates the current class except for a
            different associated Algorithm class.

        """
        orig_name, orig_bases, orig_dct = self._orig_args
        dct = orig_dct.copy()
        dct['_alg_cls'] = alg_cls
        if name is None:
            name = '{obj_name}__{alg_name}'.format(
                obj_name=orig_name, alg_name=alg_cls.__name__,
            )
        # need to subclass from original class for super() calls to work
        return type(self)(name, orig_bases, dct)


class BaseObjective(with_metaclass(ObjectiveAlgorithmMeta, object)):
    """Base class for optimization objectives.

    {objective_with_algorithm_description}

    Subclasses should define an :meth:`__init__` with any desired parameters
    and an :meth:`evaluate` method for returning the value of the objective.


    Attributes
    ----------

    {objective_attributes}


    Notes
    -----

    {objective_notes}

    """

    _doc_objective_self = ':class:`.BaseObjective`'

    _doc_objective_with_algorithm_description = """
    This Objective uses the {{algorithm_self}} algorithm class to
    instantiate self.alg.

    {{algorithm_description}}

    """

    _doc_objective_attributes = """
    alg : {{algorithm_self}}
        Algorithm object that implements the :meth:`minimize` method for this
        objective. For more details or to learn how to customize the algorithm,
        consult this object's class documentation.

    state_ : dict
        Dictionary containing the final values of the state variables, set
        after minimization has converged through :meth:`minimize`.

    {{algorithm_objective_attributes}}

    """

    _doc_objective_notes = """
    Objectives are associated with a particular optimization Algorithm class
    that is used to implement the :meth:`minimize` method. The associated
    Algorithm class is given by the :attr:`_alg_cls` class attribute, which can
    be reset by adding this class to a desired Algorithm class, e.g.
    ``NewObj = Objective + Algorithm``. A given optimization Algorithm class
    will often require a specific subclass of :class:`.BaseObjective` that
    defines a particular set of parameters.

    """

    _doc_evaluate_state_argument = """
    state : dict
        State dictionary, with entries varying by objective and algorithm. The
        state defines a point at which the objective is evaluated.

    """

    _doc_objective_parameters = ''

    _alg_cls = None

    def __init__(self, **alg_kwargs):
        """Initialize the Objective and its associated Algorithm.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}

        """
        if self._alg_cls is None:
            errstr = (
                'Must set `_alg_cls` class attribute before instantiating a'
                ' BaseAlgorithm class.'
            )
            raise ValueError(errstr)

        self.alg = self._alg_cls(objective=self, **alg_kwargs)

    def validate_params(self):
        """Validate the current set of parameters.

        Returns
        -------

        self : {objective_self}

        """
        self.alg.validate_params()
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
        params = self.alg.get_params(deep=deep)
        return params

    def _assign_params(self, **params):
        """Assign non-None keyword arguments to instance attributes."""
        for k, v in params.items():
            if v is not None:
                setattr(self, k, v)

    def set_params(self, **alg_params):
        """Set parameters post-initialization.

        Parameters
        ----------

        {objective_parameters}


        Other Parameters
        ----------------

        {algorithm_parameters}


        Returns
        -------

        self : {objective_self}

        """
        self.alg.set_params(**alg_params)
        return self

    def __call__(self, state):
        """Evaluate the objective. See :meth:`evaluate`.

        Parameters
        ----------

        {evaluate_state_argument}


        Returns
        -------

        val : float
            The objective value.

        """
        return self.evaluate(state)

    def evaluate(self, state):
        """Evaluate the objective.

        This also sets ``state['_val']`` to the objective value.


        Parameters
        ----------

        {evaluate_state_argument}


        Returns
        -------

        val : float
            The objective value.

        """
        raise NotImplementedError

    def minimize(self, state, **kwargs):
        """Minimize the objective using the associated algorithm.

        See ``self.alg.minimize`` for details pertaining to the specific
        algorithm in use.


        Parameters
        ----------

        {initial_state_argument}

        {keyword_arguments}


        Returns
        -------

        self : {objective_self}

        """
        return self.alg.minimize(state, **kwargs)
