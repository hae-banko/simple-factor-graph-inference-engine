"""Variable — a named node with a domain, value, and optional latency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bayesian_engine.utils.types import Domain, normalize_domain

if TYPE_CHECKING:
    from bayesian_engine.core.factor import Factor


@dataclass
class Variable:
    """A named variable in a Bayesian model.

    Args:
        name: Unique identifier within the model.
        domain: The set of possible values. Accepts a list of strings for discrete
            variables or a ``("continuous", lo, hi)`` tuple for continuous variables.
        value: The current value. Must be in the domain.
        latent: If True, this variable is inferred (not directly observed).
    """

    name: str
    domain: list[str] | tuple | Domain
    value: Any = None
    latent: bool = False

    # Graph connectivity — populated by FactorGraph builder.
    incoming_factors: list[Factor] = field(default_factory=list, init=False, repr=False)
    outgoing_factors: list[Factor] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.domain, Domain):
            self.domain = normalize_domain(self.domain)
        if self.value is not None and self.value not in self.domain:
            raise ValueError(
                f"Value {self.value!r} not in domain of variable '{self.name}'"
            )

    def set_value(self, value: Any) -> None:
        if value not in self.domain:
            raise ValueError(
                f"Value {value!r} not in domain of variable '{self.name}'"
            )
        self.value = value
