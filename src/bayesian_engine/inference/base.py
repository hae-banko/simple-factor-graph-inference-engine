"""Inference engine protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Potential:
    """A factor potential: an array over a set of variables."""

    vars: list[str]
    array: np.ndarray


class InferenceEngine(Protocol):
    """Protocol for inference engines."""

    def __call__(
        self,
        variables: dict,
        factors: list,
        target: str,
        evidence: set[str],
    ) -> dict: ...
