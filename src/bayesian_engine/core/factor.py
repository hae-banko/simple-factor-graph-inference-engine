"""Factor — a weighted edge connecting input variables to an output variable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Factor:
    """A factor connecting N input variables to 1 output variable via a weight function.

    Args:
        inputs: Names of parent (input) variables, in the order passed to the weight
            function.
        output: Name of the child (output) variable.
        weight_function: A callable ``f(*parent_values) -> float`` that returns a
            real-valued weight for a given configuration of parent values.
    """

    inputs: list[str]
    output: str
    weight_function: Callable[..., float]

    def __post_init__(self):
        if not self.inputs:
            raise ValueError("Factor must have at least one input variable")
        if not self.output:
            raise ValueError("Factor must have an output variable")
        if not callable(self.weight_function):
            raise TypeError("weight_function must be callable")

    def __call__(self, *parent_values: Any) -> float:
        """Evaluate the weight function on the given parent values."""
        return float(self.weight_function(*parent_values))
