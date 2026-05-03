"""Domain types for Bayesian variables."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscreteDomain:
    """A finite set of named values."""

    values: list[str]

    def __contains__(self, value: str) -> bool:
        return value in self.values

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class ContinuousDomain:
    """A real-valued interval, discretized into bins for inference."""

    lo: float
    hi: float
    bins: int = 20

    def __post_init__(self):
        if self.lo >= self.hi:
            raise ValueError(f"lo ({self.lo}) must be less than hi ({self.hi})")
        if self.bins < 2:
            raise ValueError(f"bins must be at least 2, got {self.bins}")

    @property
    def step(self) -> float:
        return (self.hi - self.lo) / self.bins

    def bin_centers(self) -> list[float]:
        half = self.step / 2
        return [self.lo + half + i * self.step for i in range(self.bins)]

    def discretize(self, value: float) -> int:
        """Return the bin index for a continuous value."""
        clamped = max(self.lo, min(self.hi - 1e-9, value))
        return int((clamped - self.lo) / self.step)

    def __contains__(self, value: float) -> bool:
        return self.lo <= value <= self.hi


Domain = DiscreteDomain | ContinuousDomain


def normalize_domain(raw: list[str] | tuple) -> Domain:
    """Normalize a user-facing domain spec into a Domain object.

    >>> normalize_domain(["a", "b", "c"])
    DiscreteDomain(values=['a', 'b', 'c'])

    >>> normalize_domain(("continuous", 0, 24))
    ContinuousDomain(lo=0, hi=24, bins=20)
    """
    if isinstance(raw, list):
        return DiscreteDomain(values=raw)
    if isinstance(raw, tuple) and raw[0] == "continuous":
        lo, hi = raw[1], raw[2]
        bins = raw[3] if len(raw) > 3 else 20
        return ContinuousDomain(lo=lo, hi=hi, bins=bins)
    raise TypeError(f"Unsupported domain spec: {raw!r}")
