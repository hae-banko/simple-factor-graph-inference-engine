"""Tests for EvidenceRecorder."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from bayesian_engine.production.evidence_recorder import EvidenceRecorder
from bayesian_engine.production.scheduler_db import SchedulerDB


@pytest.fixture
def db(tmp_path: Path) -> SchedulerDB:
    """Provide a temporary SchedulerDB for each test."""
    return SchedulerDB(db_path=tmp_path / "test.db")


@pytest.fixture
def recorder(db: SchedulerDB) -> EvidenceRecorder:
    """Provide an EvidenceRecorder backed by the temporary DB."""
    return EvidenceRecorder(db)


class TestRecordOutcome:
    def test_record_outcome_round_trip(
        self, recorder: EvidenceRecorder, db: SchedulerDB
    ) -> None:
        """Record an outcome with all fields, then query it back and verify."""
        outcome_id = recorder.record_outcome(
            task_id="task-1",
            action="process",
            inferred_at="2026-01-01T10:00:00+00:00",
            observed_at="2026-01-01T10:01:00+00:00",
            success=True,
            latency_ms=1500.0,
            error=None,
            context={"worker": "node-1"},
        )

        history = recorder.get_outcome_history("task-1")
        assert len(history) == 1

        rec = history[0]
        assert rec["id"] == outcome_id
        assert rec["task_id"] == "task-1"
        assert rec["action"] == "process"
        assert rec["inferred_at"] == "2026-01-01T10:00:00+00:00"
        assert rec["observed_at"] == "2026-01-01T10:01:00+00:00"
        assert rec["success"] is True
        assert rec["latency_ms"] == 1500.0
        assert rec["error"] is None
        assert rec["context"] == {"worker": "node-1"}

    def test_record_outcome_defaults_timestamp(
        self, recorder: EvidenceRecorder
    ) -> None:
        """Pass no timestamps; both inferred_at and observed_at should be set."""
        before = datetime.now(timezone.utc).isoformat()
        outcome_id = recorder.record_outcome(
            task_id="task-2",
            action="scan",
            success=False,
        )
        after = datetime.now(timezone.utc).isoformat()

        assert outcome_id > 0
        history = recorder.get_outcome_history("task-2")
        assert len(history) == 1

        rec = history[0]
        assert rec["inferred_at"] >= before
        assert rec["inferred_at"] <= after
        assert rec["observed_at"] >= before
        assert rec["observed_at"] <= after


class TestGetOutcomeHistory:
    def test_get_outcome_history_empty(self, recorder: EvidenceRecorder) -> None:
        """Query an unknown task_id; should return an empty list."""
        history = recorder.get_outcome_history("nonexistent-task")
        assert history == []

    def test_get_outcome_history_most_recent_first(
        self, recorder: EvidenceRecorder
    ) -> None:
        """Insert 3 outcomes; verify they come back most recent first."""
        # Insert in chronological order
        recorder.record_outcome(
            task_id="task-3",
            action="a",
            inferred_at="2026-01-01T10:00:00+00:00",
            observed_at="2026-01-01T10:01:00+00:00",
            success=True,
        )
        recorder.record_outcome(
            task_id="task-3",
            action="b",
            inferred_at="2026-01-01T11:00:00+00:00",
            observed_at="2026-01-01T11:01:00+00:00",
            success=True,
        )
        recorder.record_outcome(
            task_id="task-3",
            action="c",
            inferred_at="2026-01-01T12:00:00+00:00",
            observed_at="2026-01-01T12:01:00+00:00",
            success=False,
        )

        history = recorder.get_outcome_history("task-3")
        assert len(history) == 3
        # Most recent first
        assert history[0]["action"] == "c"
        assert history[1]["action"] == "b"
        assert history[2]["action"] == "a"


class TestGetEvidenceForInference:
    def test_get_evidence_for_inference_all_success(
        self, recorder: EvidenceRecorder
    ) -> None:
        """4 successful outcomes → success_rate = 1.0."""
        for i in range(4):
            recorder.record_outcome(
                task_id="task-4",
                action=f"action-{i}",
                inferred_at=f"2026-01-01T{10+i:02d}:00:00+00:00",
                observed_at=f"2026-01-01T{10+i:02d}:01:00+00:00",
                success=True,
                latency_ms=1000.0 + i * 100,
            )

        evidence = recorder.get_evidence_for_inference("task-4")
        assert evidence["recent_outcome_count"] == 4
        assert evidence["outcome_success_rate"] == 1.0
        assert evidence["last_outcome"] == "success"
        assert evidence["avg_latency_ms"] == 1150.0  # (1000+1100+1200+1300)/4
        assert evidence["consecutive_failures"] == 0

    def test_get_evidence_for_inference_mixed(
        self, recorder: EvidenceRecorder
    ) -> None:
        """2 success, 2 failure → success_rate = 0.5."""
        # Insert in chronological order (oldest first)
        recorder.record_outcome(
            task_id="task-5",
            action="a",
            inferred_at="2026-01-01T10:00:00+00:00",
            observed_at="2026-01-01T10:01:00+00:00",
            success=True,
            latency_ms=1000.0,
        )
        recorder.record_outcome(
            task_id="task-5",
            action="b",
            inferred_at="2026-01-01T11:00:00+00:00",
            observed_at="2026-01-01T11:01:00+00:00",
            success=False,
            latency_ms=900.0,
        )
        recorder.record_outcome(
            task_id="task-5",
            action="c",
            inferred_at="2026-01-01T12:00:00+00:00",
            observed_at="2026-01-01T12:01:00+00:00",
            success=True,
            latency_ms=1100.0,
        )
        recorder.record_outcome(
            task_id="task-5",
            action="d",
            inferred_at="2026-01-01T13:00:00+00:00",
            observed_at="2026-01-01T13:01:00+00:00",
            success=False,
            latency_ms=800.0,
        )

        evidence = recorder.get_evidence_for_inference("task-5")
        assert evidence["recent_outcome_count"] == 4
        assert evidence["outcome_success_rate"] == 0.5
        assert evidence["last_outcome"] == "failure"
        # avg of completed latencies: (1000+900+1100+800)/4 = 950.0
        assert evidence["avg_latency_ms"] == 950.0
        # Most recent is failure, consecutive_failures = 1
        assert evidence["consecutive_failures"] == 1

    def test_get_evidence_for_inference_no_history(
        self, recorder: EvidenceRecorder
    ) -> None:
        """Unknown task_id → default dict with zeros and Nones."""
        evidence = recorder.get_evidence_for_inference("unknown-task")
        assert evidence["recent_outcome_count"] == 0
        assert evidence["outcome_success_rate"] is None
        assert evidence["last_outcome"] == "none"
        assert evidence["avg_latency_ms"] is None
        assert evidence["consecutive_failures"] == 0

    def test_get_evidence_for_inference_consecutive_failures(
        self, recorder: EvidenceRecorder
    ) -> None:
        """3 outcomes, last 2 are failures → consecutive_failures = 2."""
        # success, failure, failure  (most recent = failure)
        recorder.record_outcome(
            task_id="task-6",
            action="a",
            inferred_at="2026-01-01T10:00:00+00:00",
            observed_at="2026-01-01T10:01:00+00:00",
            success=True,
        )
        recorder.record_outcome(
            task_id="task-6",
            action="b",
            inferred_at="2026-01-01T11:00:00+00:00",
            observed_at="2026-01-01T11:01:00+00:00",
            success=False,
        )
        recorder.record_outcome(
            task_id="task-6",
            action="c",
            inferred_at="2026-01-01T12:00:00+00:00",
            observed_at="2026-01-01T12:01:00+00:00",
            success=False,
        )

        evidence = recorder.get_evidence_for_inference("task-6")
        assert evidence["consecutive_failures"] == 2

    def test_get_evidence_for_inference_only_pending(
        self, recorder: EvidenceRecorder
    ) -> None:
        """All pending outcomes → success_rate and avg_latency are None."""
        recorder.record_outcome(
            task_id="task-7",
            action="a",
            inferred_at="2026-01-01T10:00:00+00:00",
            observed_at="2026-01-01T10:01:00+00:00",
            success=None,
        )
        recorder.record_outcome(
            task_id="task-7",
            action="b",
            inferred_at="2026-01-01T11:00:00+00:00",
            observed_at="2026-01-01T11:01:00+00:00",
            success=None,
        )

        evidence = recorder.get_evidence_for_inference("task-7")
        assert evidence["recent_outcome_count"] == 2
        assert evidence["outcome_success_rate"] is None
        assert evidence["last_outcome"] == "pending"
        assert evidence["avg_latency_ms"] is None
        assert evidence["consecutive_failures"] == 0


class TestMarkOutcome:
    def test_mark_outcome_success(
        self, recorder: EvidenceRecorder, db: SchedulerDB
    ) -> None:
        """Record with success=None, mark it True, query back."""
        outcome_id = recorder.record_outcome(
            task_id="task-8",
            action="process",
            inferred_at="2026-01-01T10:00:00+00:00",
            observed_at="2026-01-01T10:01:00+00:00",
            success=None,
            latency_ms=500.0,
        )

        recorder.mark_outcome_success(outcome_id)

        history = recorder.get_outcome_history("task-8")
        assert len(history) == 1
        assert history[0]["success"] is True
        assert history[0]["latency_ms"] == 500.0

    def test_mark_outcome_failure(
        self, recorder: EvidenceRecorder, db: SchedulerDB
    ) -> None:
        """Record with success=None, mark it False with error, query back."""
        outcome_id = recorder.record_outcome(
            task_id="task-9",
            action="process",
            inferred_at="2026-01-01T10:00:00+00:00",
            observed_at="2026-01-01T10:01:00+00:00",
            success=None,
        )

        recorder.mark_outcome_failure(outcome_id, error="Connection refused")

        history = recorder.get_outcome_history("task-9")
        assert len(history) == 1
        assert history[0]["success"] is False
        assert history[0]["error"] == "Connection refused"


class TestRecordOutcomeWithJsonContext:
    def test_record_outcome_with_json_context(
        self, recorder: EvidenceRecorder, db: SchedulerDB
    ) -> None:
        """Context dict is stored as JSON and retrieved correctly."""
        context = {
            "worker": "node-5",
            "attempts": 3,
            "metadata": {"region": "us-east", "tier": "production"},
        }

        recorder.record_outcome(
            task_id="task-10",
            action="deploy",
            inferred_at="2026-01-01T10:00:00+00:00",
            observed_at="2026-01-01T10:05:00+00:00",
            success=True,
            latency_ms=3000.0,
            context=context,
        )

        history = recorder.get_outcome_history("task-10")
        assert len(history) == 1
        assert history[0]["context"] == context
        assert history[0]["context"]["metadata"]["region"] == "us-east"
