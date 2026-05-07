"""DecisionLogger — dual-output audit trail for Bayesian policy decisions.

Logs every policy decision to both:
1. The SQLite ``SchedulerDB`` (inferences table)
2. A JSONL-formatted file at ``~/.hermes/bayesian_scheduler/logs/decisions.log``

This gives both structured queryable persistence (SQLite) and a human-readable
audit log suitable for grep/tail inspection.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bayesian_engine.policy import BayesianPolicy, PolicyDecision
from bayesian_engine.production.scheduler_db import SchedulerDB, get_state_dir

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_LOG_DIR = get_state_dir() / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "decisions.log"

# -----------------------------------------------------------------------------
# DecisionLogger
# -----------------------------------------------------------------------------


class DecisionLogger:
    """Dual-output audit logger for Bayesian policy decisions.

    Args:
        db: A :class:`SchedulerDB` instance used for structured persistence.
        log_file_path: Path to the JSONL log file. If ``None``, defaults to
            ``~/.hermes/bayesian_scheduler/logs/decisions.log``. The directory
            is created automatically on first write.
    """

    def __init__(
        self,
        db: SchedulerDB,
        log_file_path: Path | None = None,
    ) -> None:
        self._db = db
        self._log_file_path: Path = log_file_path if log_file_path is not None else DEFAULT_LOG_FILE

    # ------------------------------------------------------------------:
    # Public API
    # ------------------------------------------------------------------:

    def log(
        self,
        policy_decision: PolicyDecision,
        model_name: str,
        target: str,
        evidence: dict[str, Any],
        latency_ms: float | None = None,
        cron_job_id: str | None = None,
        session_id: str | None = None,
        observed_at: str | None = None,
    ) -> int:
        """Log a policy decision to both SchedulerDB and the decisions log file.

        Args:
            policy_decision: The :class:`PolicyDecision` returned by
                :meth:`BayesianPolicy.decide`.
            model_name: Name of the model used for inference.
            target: Target variable name queried.
            evidence: Evidence dictionary passed to the model.
            latency_ms: Optional inference latency in milliseconds.
            cron_job_id: Optional identifier of the originating cron job.
            session_id: Optional session identifier.
            observed_at: ISO-8601 timestamp; defaults to the current UTC time.

        Returns:
            The inference record ID (rowid) from the SchedulerDB.
        """
        observed_at = observed_at or datetime.now(timezone.utc).isoformat()

        # Persist to SQLite.
        inference_id = self._db.insert_inference(
            run_at=observed_at,
            model_name=model_name,
            target=target,
            evidence=evidence,
            action=policy_decision.action,
            reason=policy_decision.reason,
            confidence=policy_decision.confidence,
            trace=policy_decision.trace,
            latency_ms=latency_ms,
            cron_job_id=cron_job_id,
            session_id=session_id,
        )

        # Append to JSONL audit file.
        self._write_jsonl(
            inference_id=inference_id,
            model_name=model_name,
            target=target,
            evidence=evidence,
            decision=policy_decision,
            latency_ms=latency_ms,
            cron_job_id=cron_job_id,
            session_id=session_id,
            observed_at=observed_at,
        )

        return inference_id

    def log_and_decide(
        self,
        policy: BayesianPolicy,
        target: str,
        evidence: dict[str, Any],
        model_name: str,
        threshold: float = 0.35,
        fallback_action: str = "SELF_MAINTAIN",
        cron_job_id: str | None = None,
        session_id: str | None = None,
    ) -> PolicyDecision:
        """Convenience method: run BayesianPolicy.decide(), then log the result.

        This is equivalent to calling ``policy.decide(...)`` followed by
        :meth:`log`, but wraps both steps so that ``latency_ms`` captures the
        wall-clock time of the inference call accurately.

        Args:
            policy: A :class:`BayesianPolicy` instance.
            target: Target variable name.
            evidence: Evidence dictionary.
            model_name: Model identifier string (used in log records).
            threshold: Decision threshold passed to ``policy.decide``.
            fallback_action: Fallback action passed to ``policy.decide``.
            cron_job_id: Optional cron job identifier.
            session_id: Optional session identifier.

        Returns:
            The :class:`PolicyDecision` returned by ``policy.decide``.
        """
        start = time.perf_counter()
        policy_decision = policy.decide(
            target=target,
            evidence=evidence,
            threshold=threshold,
            fallback_action=fallback_action,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        self.log(
            policy_decision=policy_decision,
            model_name=model_name,
            target=target,
            evidence=evidence,
            latency_ms=latency_ms,
            cron_job_id=cron_job_id,
            session_id=session_id,
        )

        return policy_decision

    # ------------------------------------------------------------------:
    # Private helpers
    # ------------------------------------------------------------------:

    def _write_jsonl(
        self,
        inference_id: int,
        model_name: str,
        target: str,
        evidence: dict[str, Any],
        decision: PolicyDecision,
        latency_ms: float | None,
        cron_job_id: str | None,
        session_id: str | None,
        observed_at: str,
    ) -> None:
        """Append a JSON line to the audit log file.

        The log directory is created automatically on first write. Errors
        during file I/O are logged to stderr but never propagate, ensuring
        that a failed file write cannot crash or derail a policy decision.
        """
        entry = {
            "timestamp": observed_at,
            "level": "INFO",
            "inference_id": inference_id,
            "model_name": model_name,
            "target": target,
            "evidence": evidence,
            "decision": {
                "action": decision.action,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
            "latency_ms": latency_ms,
            "cron_job_id": cron_job_id,
            "session_id": session_id,
        }

        try:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_file_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
                fh.flush()
        except Exception as exc:  # pragma: no cover — defensive logging
            logger.warning("Failed to write to decision log %s: %s", self._log_file_path, exc)
            print(
                f"WARNING: DecisionLogger failed to write JSONL entry: {exc}",
                file=sys.stderr,
            )
