"""Policy module — production-safe wrapper for Bayesian inference decisions.

Provides :class:`BayesianPolicy` for loading validated models and making
fail-closed decisions, :func:`select_action` for deterministic action
selection, and :class:`PolicyDecision` for traceable decision results.
"""

from bayesian_engine.policy.wrapper import BayesianPolicy, PolicyDecision, select_action

__all__ = [
    "BayesianPolicy",
    "PolicyDecision",
    "select_action",
]
