# ----------------------------------------------------------------------------
# Copyright (c) 2018, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""Split optimization objectives."""

from . import base as _base
from .. import algorithms as _alg

__all__ = ('SplitObjective', 'SplitObjectiveAffine')


class SplitObjective(_base.BaseObjective):
    """Class for split objectives of the form ``F(x) + G(z)``.

    The first part of the objective, ``F(x)``, is a function of the state
    variable ``x``. This function is associated with a prox operator,
    ``proxF(v, lmbda)`` which gives the solution ``x`` to the problem::

        minimize    F(x) + l2normsqhalf(x - v)/lmbda.

    The second part of the objective, ``G(z)``, is evaluated using a different
    state variable, ``z``, that is dependent on the primary state variable
    ``x``. Typical cases are ``z = x`` or ``z = A(x) - b`` for a linear
    operator ``A`` and constant ``b``. This part of the objective is associated
    with either a gradient function or prox operator as a function of ``z``,
    depending on the needs of the optimization algorithm.

    {objective_with_algorithm_description}


    Attributes
    ----------

    {objective_attributes}


    Notes
    -----

    {objective_notes}

    {algorithm_description}

    """

    _doc_objective_self = ':class:`.SplitObjective`'

    _doc_evaluate_state_argument = """
    state : dict
        State dictionary containing:

        x : array_like
            Point at which the first part of the objective, `F`, is evaluated.

        z : array_like
            Point at which the second part of the objective, `G`, is evaluated.

    """

    _doc_objective_parameters = """
    F : callable
        ``F(x)``, the possibly non-smooth part of the objective function,
        evaluated in terms of ``x`` directly.

    proxF : callable
        Prox operator ``proxF(v, s)`` corresponding to `F`. It returns the
        ``x`` that minimizes ``F(x) + l2normsqhalf(x - v) / s``.

    G : callable
        ``G(z)``, the second part of the objective function, evaluated at
        ``z`` which is some function of ``x``.

    gradG : callable, sometimes optional
        Gradient function ``gradG(z)`` corresponding to `G` with respect to
        ``z``. Note that this is not the gradient of `G` as a function of
        ``x``, which is different unless ``z = x``. For example, the full
        gradient with respect to ``x`` when ``z = A(x) - b`` would be:
        ``gradGx(x) = Astar(gradG(A(x) - b))``.

    proxG : callable, sometimes optional
        Prox operator ``proxF(v, s)`` corresponding to `G`. It returns the
        ``z`` which minimizes ``G(z) + l2normsqhalf(z - v) / s``.

    """

    def __init__(
        self, F=None, proxF=None, G=None, gradG=None, proxG=None, **alg_kwargs
    ):
        """."""
        self.F = F
        self.proxF = proxF
        self.G = G
        self.gradG = gradG
        self.proxG = proxG
        super(SplitObjective, self).__init__(**alg_kwargs)

    def validate_params(self):
        """."""
        # existence of F, proxF, G, (gradG or proxG) is checked by the
        # Algorithm so that the Algorithm itself can ensure that it has a
        # compatible Objective
        return super(SplitObjective, self).validate_params()

    def get_params(self, deep=True):
        """."""
        params = super(SplitObjective, self).get_params(deep=deep)
        params.update(
            F=self.F, proxF=self.proxF, G=self.G, gradG=self.gradG,
            proxG=self.proxG,
        )
        return params

    def set_params(
        self, F=None, proxF=None, G=None, gradG=None, proxG=None, **alg_params
    ):
        """."""
        self._assign_params(F=F, proxF=proxF, G=G, gradG=gradG, proxG=proxG)
        return super(SplitObjective, self).set_params(**alg_params)

    def evaluate(self, state):
        """Evaluate the objective: ``F(x) + G(z)``.

        This also sets ``state['_val']`` to the objective value and
        ``state['_valF']`` and ``state['_valG']`` to the partial split
        objective values.


        Parameters
        ----------

        {evaluate_state_argument}


        Returns
        -------

        val : float
            The objective value.

        """
        valF = self.F(state['x'])
        state['_valF'] = valF
        valG = self.G(state['z'])
        state['_valG'] = valG
        val = valF + valG
        state['_val'] = val
        return val


SplitObjectiveAffine = SplitObjective.with_alg(
    _alg.ProxGradAccel, 'SplitObjectiveAffine',
)
