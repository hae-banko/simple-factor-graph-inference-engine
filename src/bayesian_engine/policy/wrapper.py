"""Production policy wrapper — safe model loading, query, and decision selection.

Converts raw Bayesian inference results into safe, traceable policy decisions
suitable for autonomous agent use. All failure modes default to a configurable
safe action.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from bayesian_engine.core.model import Model
from bayesian_engine.io.import_ import model_from_dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyDecision:
    """A production-safe policy decision with full traceability.

    Attributes:
        action: The selected action label.
        reason: Why this action was chosen (e.g. ``"argmax_above_threshold"``).
        confidence: The probability of the selected action (0.0–1.0).
        trace: Full decision trace, JSON-serializable.
    """

    action: str
    reason: str
    confidence: float
    trace: dict[str, Any]


def select_action(
    distribution: list[dict[str, Any]],
    threshold: float = 0.35,
    fallback_action: str = "SELF_MAINTAIN",
) -> tuple[str, float, str]:
    """Deterministic argmax-with-threshold action selector.

    Args:
        distribution: List of ``{"value": str, "probability": float}`` dicts.
        threshold: Minimum probability required to select the argmax action.
            If the argmax probability is below this threshold, the fallback
            action is returned instead.
        fallback_action: Safe default action returned when the argmax is below
            threshold.

    Returns:
        A ``(action, confidence, reason)`` tuple.
    """
    ranked = sorted(distribution, key=lambda d: d["probability"], reverse=True)
    top = ranked[0]
    if top["probability"] < threshold:
        return fallback_action, top["probability"], "below_threshold"
    return top["value"], top["probability"], "argmax_above_threshold"


class BayesianPolicy:
    """Safe production wrapper around a Bayesian model.

    Loads a serialized model file, validates it, and exposes a single
    :meth:`decide` method that never raises on normal production failure
    modes — it always returns a :class:`PolicyDecision`.
    """

    def __init__(self, model: Model | None, model_path: str, error: str | None = None):
        self._model = model
        self._model_path = model_path
        self._error = error

    def _base_trace(self, target: str, evidence: dict) -> dict:
        trace: dict[str, Any] = {
            "model_path": self._model_path,
            "target": target,
            "evidence": evidence,
        }
        if self._model is not None and self._model.metadata:
            trace["metadata"] = dict(self._model.metadata)
        return trace

    @classmethod
    def load(cls, path: str) -> BayesianPolicy:
        """Load a policy from a serialized model JSON file.

        Always returns a :class:`BayesianPolicy` — on failure the returned
        policy will default every :meth:`decide` call to the safe fallback.

        Args:
            path: Filesystem path to a JSON model file.
        """
        try:
            with open(path) as f:
                data = json.load(f)
            model = model_from_dict(data)
            logging.getLogger(__name__).info("Loaded policy model from %s", path)
            return cls(model=model, model_path=path)
        except FileNotFoundError:
            msg = f"Model file not found: {path}"
            logger.error(msg)
            return cls(model=None, model_path=path, error=msg)
        except Exception as exc:
            msg = f"Failed to load model from {path}: {exc}"
            logger.error(msg)
            return cls(model=None, model_path=path, error=msg)

    def decide(
        self,
        target: str,
        evidence: dict[str, Any] | None = None,
        threshold: float = 0.35,
        fallback_action: str = "SELF_MAINTAIN",
    ) -> PolicyDecision:
        """Query the model and return a safe policy decision.

        All failure modes (missing model, invalid target, inference error,
        empty/un-normalized distribution, low confidence) fall back to a safe
        action with a traceable reason.

        Args:
            target: Name of the target variable to query.
            evidence: Mapping of variable names to observed values.
            threshold: Minimum probability for argmax selection.
            fallback_action: Safe default when argmax is below threshold.

        Returns:
            A :class:`PolicyDecision` with action, reason, confidence, and full trace.
        """
        evidence = evidence or {}

        if self._model is None:
            trace = self._base_trace(target, evidence)
            trace["error"] = self._error or "Model not loaded"
            return PolicyDecision(
                action=fallback_action,
                reason="model_load_error",
                confidence=1.0,
                trace=trace,
            )

        try:
            result = self._model.query(target, evidence)
            query_result = result.get(target, {})
            distribution = query_result.get("distribution", [])
        except KeyError:
            trace = self._base_trace(target, evidence)
            trace["error"] = f"Target variable '{target}' not found in model"
            return PolicyDecision(
                action=fallback_action,
                reason="query_error",
                confidence=1.0,
                trace=trace,
            )
        except Exception as exc:
            trace = self._base_trace(target, evidence)
            trace["error"] = str(exc)
            return PolicyDecision(
                action=fallback_action,
                reason="inference_error",
                confidence=1.0,
                trace=trace,
            )

        if not distribution:
            trace = self._base_trace(target, evidence)
            trace["result"] = query_result
            trace["error"] = "Distribution is empty"
            return PolicyDecision(
                action=fallback_action,
                reason="empty_distribution",
                confidence=1.0,
                trace=trace,
            )

        total = sum(d["probability"] for d in distribution)
        if total < 0.99 or total > 1.01:
            trace = self._base_trace(target, evidence)
            trace["result"] = query_result
            trace["error"] = f"Distribution sums to {total}, expected ~1.0"
            return PolicyDecision(
                action=fallback_action,
                reason="invalid_distribution",
                confidence=1.0,
                trace=trace,
            )

        action, confidence, reason = select_action(
            distribution, threshold=threshold, fallback_action=fallback_action
        )

        trace = self._base_trace(target, evidence)
        trace["result"] = query_result
        trace["selection"] = {"action": action, "reason": reason, "confidence": confidence}
        return PolicyDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            trace=trace,
        )
