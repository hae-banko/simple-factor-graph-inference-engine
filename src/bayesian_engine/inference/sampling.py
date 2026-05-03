"""Rejection sampling — approximate inference for cyclic factor graphs."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from bayesian_engine.core.factor import Factor
from bayesian_engine.core.variable import Variable
from bayesian_engine.utils.types import ContinuousDomain, DiscreteDomain


def rejection_sampling(
    variables: dict[str, Variable],
    factors: list[Factor],
    target: str,
    evidence: set[str],
    n_samples: int = 10_000,
) -> dict[str, Any]:
    """Compute P(target | evidence) using rejection sampling.

    Samples random values for non-evidence variables, evaluates the joint weight
    (product of all factor weights), and builds a weighted distribution over the
    target variable.

    Args:
        variables: All relevant variables.
        factors: All relevant factors.
        target: Query variable name.
        evidence: Evidence variable names.
        n_samples: Number of samples to draw.

    Returns:
        Dict with ``value`` and ``distribution``.
    """
    target_var = variables[target]
    non_evidence = [v for v in variables if v not in evidence]
    rng = random.Random(42)

    weights: list[float] = []
    target_values: list[Any] = []

    for _ in range(n_samples):
        sample = {}
        for name in non_evidence:
            var = variables[name]
            sample[name] = _random_value(var.domain, rng)
        for name in evidence:
            sample[name] = variables[name].value

        weight = 1.0
        for factor in factors:
            input_vals = [sample[inp] for inp in factor.inputs]
            output_val = sample[factor.output]
            factor_args = [*input_vals, output_val]
            try:
                w = factor.weight_function(*factor_args)
            except Exception:
                w = 0.0
            weight *= float(w)

        weights.append(weight)
        target_values.append(sample[target])

    return _build_weighted_distribution(target_var, target_values, weights)


def _random_value(domain: DiscreteDomain | ContinuousDomain, rng: random.Random) -> Any:
    """Draw a random value from a domain."""
    if isinstance(domain, DiscreteDomain):
        return rng.choice(domain.values)
    return rng.uniform(domain.lo, domain.hi)


def _build_weighted_distribution(
    var: Variable,
    values: list[Any],
    weights: list[float],
) -> dict[str, Any]:
    """Build a probability distribution from weighted samples."""
    if isinstance(var.domain, DiscreteDomain):
        domain_values = var.domain.values
        probs = np.zeros(len(domain_values))
        for val, w in zip(values, weights):
            idx = domain_values.index(val)
            probs[idx] += w
    else:
        domain_values = var.domain.bin_centers()
        probs = np.zeros(len(domain_values))
        for val, w in zip(values, weights):
            idx = var.domain.discretize(val)
            probs[idx] += w

    total = np.sum(probs)
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones_like(probs) / len(probs)

    map_idx = int(np.argmax(probs))
    distribution = [
        {"value": domain_values[i], "probability": float(probs[i])}
        for i in range(len(domain_values))
    ]

    return {
        "value": domain_values[map_idx],
        "distribution": distribution,
    }
