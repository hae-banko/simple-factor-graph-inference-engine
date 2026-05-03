"""Factor graph — adjacency list construction, cycle detection, subgraph extraction."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bayesian_engine.core.factor import Factor
    from bayesian_engine.core.variable import Variable


def build_factor_graph(
    variables: dict[str, Variable], factors: list[Factor]
) -> dict[str, Variable]:
    """Populate each variable's incoming/outgoing factor lists.

    Returns the variables dict (mutated in place) for chaining.
    """
    for var in variables.values():
        var.incoming_factors.clear()
        var.outgoing_factors.clear()

    for factor in factors:
        target = variables.get(factor.output)
        if target is None:
            raise KeyError(f"Factor output variable '{factor.output}' not found")
        target.incoming_factors.append(factor)
        for input_name in factor.inputs:
            source = variables.get(input_name)
            if source is None:
                raise KeyError(f"Factor input variable '{input_name}' not found")
            source.outgoing_factors.append(factor)

    return variables


def detect_cycles(variables: dict[str, Variable]) -> list[list[str]]:
    """Return a list of cycles found in the factor graph.

    A cycle is a sequence of variable names where each variable influences the next
    through a chain of factors. Detection is advisory — cycles do not prevent inference.
    """
    adj: dict[str, list[str]] = {name: [] for name in variables}
    for name, var in variables.items():
        for factor in var.outgoing_factors:
            target = factor.output
            if target in adj and target != name:
                adj[name].append(target)

    unvisited, visiting, done = 0, 1, 2
    color: dict[str, int] = {name: unvisited for name in variables}
    parent: dict[str, str | None] = {name: None for name in variables}
    cycles: list[list[str]] = []

    def dfs(node: str):
        color[node] = visiting
        for neighbor in adj.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == visiting:
                cycle = [neighbor, node]
                curr = node
                while parent.get(curr) and parent[curr] != neighbor:
                    curr = parent[curr]
                    cycle.append(curr)
                cycle.append(neighbor)
                cycle.reverse()
                cycles.append(cycle)
            elif color[neighbor] == unvisited:
                parent[neighbor] = node
                dfs(neighbor)
        color[node] = done

    for name in variables:
        if color[name] == unvisited:
            dfs(name)

    return cycles


def extract_subgraph(
    variables: dict[str, Variable],
    target: str,
    evidence: set[str],
) -> tuple[dict[str, Variable], list[Factor]]:
    """Extract the subgraph relevant to a query.

    Returns (relevant_variables, relevant_factors) — all variables and factors that
    are ancestors of the target (reachable by walking backward through factor inputs).
    """
    if target not in variables:
        raise KeyError(f"Target variable '{target}' not found")

    relevant_vars: dict[str, Variable] = {}
    relevant_factors: list[Factor] = []
    visited: set[str] = set()
    queue = deque([target])

    while queue:
        var_name = queue.popleft()
        if var_name in visited:
            continue
        visited.add(var_name)
        var = variables[var_name]
        relevant_vars[var_name] = var

        for factor in var.incoming_factors:
            if factor not in relevant_factors:
                relevant_factors.append(factor)
            for input_name in factor.inputs:
                if input_name not in visited and input_name not in evidence:
                    queue.append(input_name)

    return relevant_vars, relevant_factors
