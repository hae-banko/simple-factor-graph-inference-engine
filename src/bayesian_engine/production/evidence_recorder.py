"""Evidence recorder for Bayesian inference feedback loop."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .scheduler_db import SchedulerDB


class EvidenceRecorder:
    """Records observed outcomes and builds evidence dictionaries for Bayesian inference.

    This class closes the feedback loop: decisions made by BayesianPolicy are
    recorded here when their outcomes are observed, and historical evidence is
    queried here when making the next inference.

    Args:
        db: A SchedulerDB instance used for persistence.
    """

    def __init__(self, db: SchedulerDB) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Timestamp helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_now() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Record outcomes
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        task_id: str,
        action: str,
        inferred_at: str | None = None,
        observed_at: str | None = None,
        success: bool | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
        context: dict | None = None,
    ) -> int:
        """Record an observed outcome for a task/action pair.

        Args:
            task_id: Identifier of the task whose outcome is being recorded.
            action: Action string that was executed.
            inferred_at: ISO-8601 timestamp of when the decision was made.
                Defaults to the current UTC time.
            observed_at: ISO-8601 timestamp of when the outcome was observed.
                Defaults to the current UTC time.
            success: ``True`` if successful, ``False`` if failed,
                ``None`` if outcome is still pending.
            latency_ms: Optional execution latency in milliseconds.
            error: Optional error message string.
            context: Optional arbitrary context dictionary.

        Returns:
            The integer outcome record ID from SchedulerDB.
        """
        inferred_at = inferred_at if inferred_at is not None else self._utc_now()
        observed_at = observed_at if observed_at is not None else self._utc_now()

        return self._db.insert_outcome(
            inferred_at=inferred_at,
            observed_at=observed_at,
            task_id=task_id,
            action=action,
            success=success,
            latency_ms=latency_ms,
            error=error,
            context=context,
        )

    def mark_outcome_success(
        self, outcome_id: int, observed_at: str | None = None
    ) -> None:
        """Mark an outcome (by DB ID) as successful.

        Used when a previously-pending outcome resolves to success.

        Args:
            outcome_id: The integer outcome record ID.
            observed_at: Optional override for the observed_at timestamp.
        """
        conn = self._db.get_connection()
        cursor = conn.cursor()
        if observed_at is None:
            observed_at = self._utc_now()
        cursor.execute(
            """
            UPDATE outcomes
            SET success = 1, observed_at = ?
            WHERE id = ?;
            """,
            (observed_at, outcome_id),
        )
        conn.commit()

    def mark_outcome_failure(
        self,
        outcome_id: int,
        error: str,
        observed_at: str | None = None,
    ) -> None:
        """Mark an outcome (by DB ID) as failed with an error message.

        Used when a previously-pending outcome resolves to failure.

        Args:
            outcome_id: The integer outcome record ID.
            error: Error message string describing the failure.
            observed_at: Optional override for the observed_at timestamp.
        """
        conn = self._db.get_connection()
        cursor = conn.cursor()
        if observed_at is None:
            observed_at = self._utc_now()
        cursor.execute(
            """
            UPDATE outcomes
            SET success = 0, observed_at = ?, error = ?
            WHERE id = ?;
            """,
            (observed_at, error, outcome_id),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Query history
    # ------------------------------------------------------------------

    def get_outcome_history(
        self,
        task_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get outcome history for a task, most recent first.

        Args:
            task_id: The task identifier to query.
            limit: Maximum number of records to return (default 100).

        Returns:
            A list of outcome dictionaries, ordered by observed_at descending.
        """
        records = self._db.get_outcomes(task_id=task_id, limit=limit)
        return [
            {
                "id": r.id,
                "inferred_at": r.inferred_at,
                "observed_at": r.observed_at,
                "task_id": r.task_id,
                "action": r.action,
                "success": r.success,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "context": r.context,
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Evidence for inference
    # ------------------------------------------------------------------

    def get_evidence_for_inference(
        self,
        task_id: str,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Return an evidence dict shaped for passing to BayesianPolicy.decide().

        Analyses the most recent outcome history for ``task_id`` and returns
        a dictionary of derived statistics. If no history exists, a sensible
        default is returned with zero counts and ``None`` values.

        Args:
            task_id: The task identifier to build evidence for.
            since: If provided, only consider outcomes with observed_at >= since.

        Returns:
            A dictionary containing:
            - ``recent_outcome_count``: Number of outcomes in the window.
            - ``outcome_success_rate``: Fraction of outcomes where success is True.
              ``None`` if no completed outcomes exist.
            - ``last_outcome``: One of "success", "failure", "pending", "none".
            - ``avg_latency_ms``: Mean latency of completed outcomes.
              ``None`` if no completed outcomes exist.
            - ``consecutive_failures``: Count of trailing failures at the end
              of the history (for escalation signals).
        """
        since_str: str | None = since.isoformat() if since is not None else None
        records = self._db.get_outcomes(task_id=task_id, since=since_str, limit=100)

        if not records:
            return {
                "recent_outcome_count": 0,
                "outcome_success_rate": None,
                "last_outcome": "none",
                "avg_latency_ms": None,
                "consecutive_failures": 0,
            }

        # Count outcomes and successes (only completed ones count for rate)
        completed = [r for r in records if r.success is not None]
        success_count = sum(1 for r in completed if r.success is True)

        recent_outcome_count = len(records)

        if completed:
            outcome_success_rate = success_count / len(completed)
            latencies = [r.latency_ms for r in completed if r.latency_ms is not None]
            avg_latency_ms = sum(latencies) / len(latencies) if latencies else None
        else:
            outcome_success_rate = None
            avg_latency_ms = None

        # last_outcome is based on the most recent record (records are desc)
        last_record = records[0]
        if last_record.success is None:
            last_outcome = "pending"
        elif last_record.success is True:
            last_outcome = "success"
        else:
            last_outcome = "failure"

        # consecutive_failures: walk backwards from most recent
        consecutive_failures = 0
        for r in records:
            if r.success is False:
                consecutive_failures += 1
            else:
                break

        return {
            "recent_outcome_count": recent_outcome_count,
            "outcome_success_rate": outcome_success_rate,
            "last_outcome": last_outcome,
            "avg_latency_ms": avg_latency_ms,
            "consecutive_failures": consecutive_failures,
        }
