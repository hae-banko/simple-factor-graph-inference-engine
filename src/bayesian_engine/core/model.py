"""Model — named container for variables and factors with strict isolation."""

from __future__ import annotations

from typing import Any

from bayesian_engine.core.factor import Factor
from bayesian_engine.core.graph import build_factor_graph, detect_cycles, extract_subgraph
from bayesian_engine.core.variable import Variable


class Model:
    """A named collection of variables and factors forming an independent probability graph.

    Models enforce strict isolation — factors in one model cannot reference variables
    in another model.

    Args:
        name: Unique identifier for this model.
        metadata: Optional dict with model metadata
            (e.g. ``model_version``, ``description``).
    """

    def __init__(self, name: str, metadata: dict[str, str] | None = None) -> None:
        self.name = name
        self.metadata = metadata or {}
        self._variables: dict[str, Variable] = {}
        self._factors: list[Factor] = []

    @property
    def variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def factors(self) -> list[Factor]:
        return self._factors

    def add_variable(self, variable: Variable) -> None:
        """Register a variable in this model.

        Raises ValueError if a variable with the same name already exists.
        """
        if variable.name in self._variables:
            raise ValueError(
                f"Variable '{variable.name}' already exists in model '{self.name}'"
            )
        self._variables[variable.name] = variable

    def add_factor(self, factor: Factor) -> None:
        """Register a factor in this model.

        All input and output variables must already be registered in this model.

        Raises KeyError if any referenced variable is not found.
        """
        if factor.output not in self._variables:
            raise KeyError(
                f"Factor output variable '{factor.output}' not found in model '{self.name}'"
            )
        for input_name in factor.inputs:
            if input_name not in self._variables:
                raise KeyError(
                    f"Factor input variable '{input_name}' not found in model '{self.name}'"
                )
        self._factors.append(factor)

    def set_evidence(self, evidence: dict[str, Any]) -> None:
        """Set observed values on variables.

        Args:
            evidence: Mapping of variable names to their observed values.

        Raises KeyError if a variable is not found in this model.
        """
        for name, value in evidence.items():
            if name not in self._variables:
                raise KeyError(f"Variable '{name}' not found in model '{self.name}'")
            self._variables[name].set_value(value)

    def query(
        self, target: str, evidence: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute P(target | evidence).

        Args:
            target: Name of the target variable.
            evidence: Mapping of variable names to observed values.

        Returns:
            A dict with ``target`` mapping to ``{"value": ..., "distribution": ...}``.
        """
        from bayesian_engine.inference.elimination import variable_elimination
        from bayesian_engine.inference.sampling import rejection_sampling

        if target not in self._variables:
            raise KeyError(f"Target variable '{target}' not found in model '{self.name}'")

        if evidence:
            self.set_evidence(evidence)

        evidence_names = set(evidence.keys()) if evidence else set()
        evidence_names.discard(target)

        graph_vars = build_factor_graph(self._variables, self._factors)
        relevant_vars, relevant_factors = extract_subgraph(
            graph_vars, target, evidence_names
        )

        # Inference still needs domain info for evidence variables
        # (they're excluded from relevant_vars but their domains are
        # required for constructing factor potentials).
        inference_vars = dict(relevant_vars)
        for ev_name in evidence_names:
            if ev_name in self._variables and ev_name not in inference_vars:
                inference_vars[ev_name] = self._variables[ev_name]

        cycles = detect_cycles(inference_vars)

        if cycles:
            result = rejection_sampling(
                inference_vars, relevant_factors, target, evidence_names
            )
        else:
            result = variable_elimination(
                inference_vars, relevant_factors, target, evidence_names
            )

        return {target: result}
