"""Variable elimination — exact inference for acyclic factor graphs."""

from __future__ import annotations

from typing import Any

import numpy as np

from bayesian_engine.core.factor import Factor
from bayesian_engine.core.variable import Variable
from bayesian_engine.inference.base import Potential
from bayesian_engine.utils.types import ContinuousDomain, DiscreteDomain


def variable_elimination(
    variables: dict[str, Variable],
    factors: list[Factor],
    target: str,
    evidence: set[str],
) -> dict[str, Any]:
    """Compute P(target | evidence) using variable elimination.

    Args:
        variables: All relevant variables (name → Variable).
        factors: All relevant factors.
        target: Name of the query variable.
        evidence: Names of evidence (observed) variables.

    Returns:
        Dict with ``value`` (MAP estimate) and ``distribution`` (probability over domain).
    """
    target_var = variables[target]
    potentials = [_build_potential(f, variables) for f in factors]

    potentials = _apply_evidence(potentials, evidence, variables)
    elim_order = _elimination_order(variables, potentials, target, evidence)

    for var_name in elim_order:
        relevant = [p for p in potentials if var_name in p.vars]
        if not relevant:
            continue
        others = [p for p in potentials if var_name not in p.vars]
        product = _factor_product(relevant, variables)
        marginalized = _sum_out(product, var_name, variables[var_name])
        potentials = others + [marginalized]

    if potentials:
        final = _factor_product(potentials, variables)
    else:
        final = Potential(vars=[target], array=np.ones(_domain_size(target_var.domain)))

    probs = _normalize(final.array)

    return _format_result(target_var, probs, final.vars)


def _build_potential(factor: Factor, variables: dict[str, Variable]) -> Potential:
    """Build a factor potential array over all variables (inputs + output)."""
    var_names = [*factor.inputs, factor.output]
    domains = [_get_domain_values(variables[name].domain) for name in var_names]
    sizes = [len(d) for d in domains]
    array = np.zeros(sizes)

    for idx in np.ndindex(*sizes):
        values = [domains[i][idx[i]] for i in range(len(var_names))]
        array[idx] = factor.weight_function(*values)

    return Potential(vars=var_names, array=array)


def _get_domain_values(domain: DiscreteDomain | ContinuousDomain) -> list:
    """Return the enumerable values of a domain."""
    if isinstance(domain, DiscreteDomain):
        return domain.values
    return [round(v, 6) for v in domain.bin_centers()]


def _domain_size(domain: DiscreteDomain | ContinuousDomain) -> int:
    if isinstance(domain, DiscreteDomain):
        return len(domain.values)
    return domain.bins


def _value_index(value: Any, domain: DiscreteDomain | ContinuousDomain) -> int:
    """Map a value to its index in the domain."""
    if isinstance(domain, DiscreteDomain):
        return domain.values.index(value)
    return domain.discretize(value)


def _apply_evidence(
    potentials: list[Potential],
    evidence: set[str],
    variables: dict[str, Variable],
) -> list[Potential]:
    """Fix evidence variables to their observed values in each potential."""
    result = []
    for pot in potentials:
        for ev_name in evidence:
            if ev_name in pot.vars:
                axis = pot.vars.index(ev_name)
                ev_idx = _value_index(variables[ev_name].value, variables[ev_name].domain)
                pot = Potential(
                    vars=[v for v in pot.vars if v != ev_name],
                    array=np.take(pot.array, ev_idx, axis=axis),
                )
        result.append(pot)
    return [p for p in result if p.vars]


def _elimination_order(
    variables: dict[str, Variable],
    potentials: list[Potential],
    target: str,
    evidence: set[str],
) -> list[str]:
    """Compute variable elimination order using min-degree heuristic.

    Variables with the fewest neighbors (in the factor graph) are eliminated first.
    """
    skip = {target} | evidence
    candidates = [v for v in variables if v not in skip]

    neighbor_count: dict[str, int] = {v: 0 for v in candidates}
    for pot in potentials:
        for v in pot.vars:
            if v in neighbor_count:
                neighbor_count[v] += len(pot.vars) - 1

    return sorted(candidates, key=lambda v: (neighbor_count.get(v, 0), v))


def _factor_product(
    potentials: list[Potential],
    variables: dict[str, Variable],
) -> Potential:
    """Multiply a list of potentials into a single potential."""
    if not potentials:
        raise ValueError("Cannot multiply empty list of potentials")
    if len(potentials) == 1:
        return potentials[0]

    all_vars = list(dict.fromkeys(v for p in potentials for v in p.vars))
    sizes = {v: _domain_size(variables[v].domain) for v in all_vars}

    result = np.ones([sizes[v] for v in all_vars])

    for pot in potentials:
        expanded = pot.array.reshape(
            [sizes[v] if v in pot.vars else 1 for v in all_vars]
        )
        result = result * expanded

    return Potential(vars=all_vars, array=result)


def _sum_out(
    potential: Potential,
    var_name: str,
    variable: Variable,
) -> Potential:
    """Sum out (marginalize) a variable from a potential."""
    axis = potential.vars.index(var_name)
    summed = np.sum(potential.array, axis=axis)
    new_vars = [v for v in potential.vars if v != var_name]
    return Potential(vars=new_vars, array=summed)


def _normalize(array: np.ndarray) -> np.ndarray:
    """Normalize an array to sum to 1 (probability distribution)."""
    total = np.sum(array)
    if total == 0:
        return np.ones_like(array) / array.size
    return array / total


def _format_result(
    var: Variable,
    probs: np.ndarray,
    var_order: list[str],
) -> dict[str, Any]:
    """Format inference result as a dict with value and distribution."""
    probs_flat = probs.flatten()

    if isinstance(var.domain, DiscreteDomain):
        values = var.domain.values
    else:
        values = [round(v, 4) for v in var.domain.bin_centers()]

    if len(values) != len(probs_flat):
        values = values[: len(probs_flat)]
        probs_flat = probs_flat[: len(values)]

    map_idx = int(np.argmax(probs_flat))
    distribution = [
        {"value": values[i], "probability": float(probs_flat[i])}
        for i in range(len(values))
    ]

    return {
        "value": values[map_idx],
        "distribution": distribution,
    }
