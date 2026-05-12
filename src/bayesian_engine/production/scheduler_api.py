"""BayesianScheduler — unified composition API for the production Bayesian engine.

This is the single public entry point for the copresence skill and any future
Hermes cron job. It composes::

    ModelRegistry   ← loads policy models from ~/.hermes/bayesian_scheduler/models/
    StreamingEngine ← online belief updating (seed=42, persisted)
    DecisionLogger  ← dual-write audit: SQLite + JSONL
    EvidenceRecorder ← outcome recording and evidence history

Usage::

    scheduler = BayesianScheduler(model_name="copresence")
    decision = scheduler.infer("action_class", {"day_type": "weekday", "hour_block": "morning", ...})
    scheduler.record_outcome(task_id="...", action=decision.action, success=True)
    beliefs = scheduler.get_beliefs("contact_window")
    scheduler.reset()   # clear all belief state
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bayesian_engine.policy import BayesianPolicy, PolicyDecision
from bayesian_engine.production.decision_logger import DecisionLogger
from bayesian_engine.production.evidence_recorder import EvidenceRecorder
from bayesian_engine.production.model_registry import ModelRegistry
from bayesian_engine.production.scheduler_db import SchedulerDB, get_state_dir
from bayesian_engine.production.streaming_engine import StreamingEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "copresence"
DEFAULT_THRESHOLD = 0.35
DEFAULT_N_PARTICLES = 200
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# BayesianScheduler
# ---------------------------------------------------------------------------

class BayesianScheduler:
    """Unified interface to the production Bayesian engine.

    Attributes
    ----------
    model_name : str
        Name of the policy model to use (looked up in the ModelRegistry).
    db : SchedulerDB
        SQLite persistence layer.
    evidence : EvidenceRecorder
        Records and queries outcome history for evidence injection.
    decisions : DecisionLogger
        Logs every policy decision to SQLite and JSONL.
    streaming : StreamingEngine
        Maintains online belief state across inference calls.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        state_dir: Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        n_particles: int = DEFAULT_N_PARTICLES,
        seed: int = DEFAULT_SEED,
        learning_rate: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str, default "copresence"
            Name of the registered policy model to load.
        state_dir : Path, optional
            Override the default state directory
            (``~/.hermes/bayesian_scheduler``). If None, uses the default.
        threshold : float, default 0.35
            Minimum confidence for a non-default action.
            Passed through to each ``infer()`` call.
        n_particles : int, default 200
            Number of particles for the streaming engine.
        seed : int, default 42
            RNG seed for deterministic particle resampling.
        learning_rate : float, default 1.0
            Dirichlet learning rate. Values < 1.0 slow adaptation, preventing
            the belief state from collapsing to a deterministic posterior.
            Recommended for copresence: 0.1.
        """
        self.model_name = model_name
        self._threshold = threshold
        self._n_particles = n_particles
        self._seed = seed
        self._learning_rate = learning_rate
        self._state_dir = state_dir or get_state_dir()

        # Core persistence — SchedulerDB takes db_path, not state_dir
        db_path = self._state_dir / "scheduler.db"
        self.db = SchedulerDB(db_path=db_path)

        # Model registry — takes state_dir
        self._registry = ModelRegistry(state_dir=self._state_dir)

        # Load the policy model via the registry
        self._policy: BayesianPolicy = self._registry.load_model(model_name)

        # Streaming engine — needs the absolute model path directly from the registry
        # (registry resolves manifest paths relative to state_dir; StreamingEngine
        # needs the absolute path so it can load the JSON file independently)
        model_path = self._registry._resolve_path(model_name)
        self.streaming = StreamingEngine(
            model_path=model_path,
            n_particles=n_particles,
            seed=seed,
            learning_rate=learning_rate,
        )

        # Audit trail — DecisionLogger needs (db, log_file_path)
        self.decisions = DecisionLogger(
            db=self.db,
            log_file_path=self._state_dir / "logs" / "decisions.log",
        )

        # Outcome recorder
        self.evidence = EvidenceRecorder(db=self.db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(
        self,
        target: str,
        evidence: dict[str, Any],
        threshold: float | None = None,
        session_id: str | None = None,
        cron_job_id: str | None = None,
    ) -> PolicyDecision:
        """Run streaming inference and log the decision.

        This is the primary method the copresence skill calls on each cron tick.

        1. StreamingEngine.update_and_query() — updates beliefs and returns the
           posterior for ``target`` using the accumulated belief state as prior.
        2. BayesianPolicy.decide() — converts the posterior into a concrete
           PolicyDecision (action + reason + confidence).
        3. DecisionLogger.log() — persists the decision to SQLite + JSONL.

        Parameters
        ----------
        target : str
            Variable name to query (e.g. ``"action_class"``).
        evidence : dict[str, Any]
            Newly observed evidence for this tick
            (e.g. ``{"day_type": "weekday", "user_presence": "active_recently"}``).
        threshold : float, optional
            Override the instance-level threshold for this call only.
        session_id : str, optional
            Hermes session ID (for audit correlation).
        cron_job_id : str, optional
            Cron job ID that triggered this inference (for audit correlation).

        Returns
        -------
        PolicyDecision
            The policy decision with action, reason, confidence, and posterior trace.
        """
        thr = threshold if threshold is not None else self._threshold

        # Atomic belief update + query via the streaming engine
        posterior = self.streaming.update_and_query(target, evidence)

        # Extract distribution from the streaming posterior
        posterior_entry = posterior.get(target, {})
        distribution = posterior_entry.get("distribution", [])

        # Map to a PolicyDecision using select_action (same logic as BayesianPolicy.decide)
        # BayesianPolicy.decide() runs fresh VE internally — we use the streaming posterior instead
        from bayesian_engine.policy.wrapper import select_action
        action, confidence, reason = select_action(
            distribution=distribution,
            threshold=thr,
            fallback_action="SELF_MAINTAIN",
        )

        decision = PolicyDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            trace={
                "posterior": posterior,
                "streaming": True,
                "model": self.model_name,
            },
        )

        # Audit log — SQLite + JSONL
        self.decisions.log(
            policy_decision=decision,
            model_name=self.model_name,
            target=target,
            evidence=evidence,
            session_id=session_id,
            cron_job_id=cron_job_id,
        )

        return decision

    def record_outcome(
        self,
        task_id: str,
        action: str,
        success: bool | None,
        latency_ms: float | None = None,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> int:
        """Record the observed outcome of a previously logged decision.

        Called by the copresence skill when Master's next turn reveals whether
        the CHECK_IN succeeded.

        Parameters
        ----------
        task_id : str
            Identifier correlating this outcome to the inference that produced it.
        action : str
            The action that was taken (from ``PolicyDecision.action``).
        success : bool or None
            Whether the action achieved its intended effect.
            ``None`` means unknown / not yet resolved.
        latency_ms : float, optional
            Time from decision to observed outcome.
        error : str, optional
            Error message if the action failed.
        context : dict, optional
            Arbitrary additional context (e.g. ``{"channel": "discord", "reply_to": "12345"}``).

        Returns
        -------
        int
            The outcome database row ID.
        """
        return self.evidence.record_outcome(
            inferred_at=datetime.now(timezone.utc).isoformat(),
            observed_at=datetime.now(timezone.utc).isoformat(),
            task_id=task_id,
            action=action,
            success=success,
            latency_ms=latency_ms,
            error=error,
            context=context,
        )

    def list_decisions(
        self,
        limit: int = 20,
        model_name: str | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent policy decisions.

        Parameters
        ----------
        limit : int, default 20
            Maximum number of decisions to return.
        model_name : str, optional
            Filter by model name. If None, returns all models.
        since : datetime, optional
            Only return decisions after this timestamp.

        Returns
        -------
        list[dict]
            Each dict contains the fields from the ``inferences`` SQLite table.
        """
        return self.db.get_inferences(
            model_name=model_name or self.model_name,
            since=since,
            limit=limit,
        )

    def get_beliefs(self, variable: str) -> dict[str, Any]:
        """Return the current belief state for a variable.

        Parameters
        ----------
        variable : str
            Name of the variable (e.g. ``"contact_window"``).

        Returns
        -------
        dict
            For particle-represented (latent) variables::

                {
                    "type": "particles",
                    "weights": [...],
                    "distribution": {<value>: <probability>, ...}
                }

            For CPT (table-factor) variables::

                {
                    "type": "cpt",
                    "pseudo_counts": {...},
                    "probabilities": {...}
                }
        """
        return self.streaming.get_current_beliefs(variable)

    def reset(self) -> None:
        """Clear all belief state and delete persisted beliefs.

        Use this when the scheduler should forget accumulated evidence
        (e.g. after a major context shift or manual reset).
        """
        self.streaming.reset_beliefs()
        logger.info("BayesianScheduler belief state reset.")

    def list_models(self) -> list[dict[str, Any]]:
        """Return all registered policy models."""
        return self._registry.list_models()

    def get_evidence_for_inference(
        self,
        task_id: str,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Return evidence history relevant to a task_id.

        Parameters
        ----------
        task_id : str
            The task to get evidence for.
        since : datetime, optional
            Only consider outcomes after this timestamp.

        Returns
        -------
        dict
            ``EvidenceRecorder.get_evidence_for_inference()`` result.
        """
        return self.evidence.get_evidence_for_inference(task_id, since=since)
