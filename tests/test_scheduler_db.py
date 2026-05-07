"""Tests for the production scheduler database module."""

from __future__ import annotations

import json
import sqlite3

import pytest

from bayesian_engine.production.scheduler_db import (
    SchedulerDB,
    InferenceRecord,
    OutcomeRecord,
    PosteriorRecord,
    STATE_DIR,
    get_state_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Place a fresh database file in a temporary directory."""
    return tmp_path / "test_scheduler.db"


@pytest.fixture
def db(db_path):
    """Construct a SchedulerDB instance backed by a temporary file."""
    db = SchedulerDB(db_path=db_path)
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_exists(conn: sqlite3.Connection, table: str, key: str, value: str) -> bool:
    """Return True if a row with (key, value) exists in ``table``."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT 1 FROM {table} WHERE key = ?", (value,))
    return cursor.fetchone() is not None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Return True if ``table`` exists in the database."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitCreatesTables:
    def test_init_creates_tables(self, db_path):
        """SchedulerDB() creates all four tables on a fresh database."""
        db = SchedulerDB(db_path=db_path)
        try:
            conn = db.get_connection()
            for table in ("meta", "inferences", "outcomes", "posteriors"):
                assert _table_exists(conn, table), f"Table '{table}' was not created"
        finally:
            db.close()

    def test_meta_version_set_on_init(self, db_path):
        """meta.version is set to '1' on first initialisation."""
        db = SchedulerDB(db_path=db_path)
        try:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key = 'version';")
            row = cursor.fetchone()
            assert row is not None, "meta.version row not found"
            assert row["value"] == "1"
        finally:
            db.close()


class TestInsertInferenceRoundTrip:
    def test_insert_inference_round_trip(self, db):
        """Insert an inference and retrieve it with all fields matching."""
        inference_id = db.insert_inference(
            run_at="2026-01-01T10:00:00Z",
            model_name="test_model",
            target="action",
            evidence={"day": "weekend"},
            action="GREET",
            reason="argmax_above_threshold",
            confidence=0.85,
            trace={"step": 1, "details": "ok"},
            latency_ms=12.5,
            cron_job_id="cron-42",
            session_id="session-abc",
        )
        assert inference_id > 0

        records = db.get_inferences(model_name="test_model", limit=10)
        assert len(records) == 1

        rec = records[0]
        assert rec.id == inference_id
        assert rec.run_at == "2026-01-01T10:00:00Z"
        assert rec.model_name == "test_model"
        assert rec.target == "action"
        assert rec.evidence == {"day": "weekend"}
        assert rec.action == "GREET"
        assert rec.reason == "argmax_above_threshold"
        assert rec.confidence == 0.85
        assert rec.trace == {"step": 1, "details": "ok"}
        assert rec.latency_ms == 12.5
        assert rec.cron_job_id == "cron-42"
        assert rec.session_id == "session-abc"


class TestInsertOutcomeRoundTrip:
    def test_insert_outcome_round_trip(self, db):
        """Insert an outcome with JSON context and retrieve it with all fields matching."""
        outcome_id = db.insert_outcome(
            inferred_at="2026-01-01T10:00:00Z",
            observed_at="2026-01-01T10:01:00Z",
            task_id="task-001",
            action="GREET",
            success=True,
            latency_ms=150.0,
            error=None,
            context={"user_id": "u123", "region": "us-east"},
        )
        assert outcome_id > 0

        records = db.get_outcomes(task_id="task-001", limit=10)
        assert len(records) == 1

        rec = records[0]
        assert rec.id == outcome_id
        assert rec.inferred_at == "2026-01-01T10:00:00Z"
        assert rec.observed_at == "2026-01-01T10:01:00Z"
        assert rec.task_id == "task-001"
        assert rec.action == "GREET"
        assert rec.success is True
        assert rec.latency_ms == 150.0
        assert rec.error is None
        assert rec.context == {"user_id": "u123", "region": "us-east"}


class TestInsertPosteriorRoundTrip:
    def test_insert_posterior_round_trip(self, db):
        """Insert a posterior and retrieve it with all fields matching."""
        posterior_id = db.insert_posterior(
            snapshot_at="2026-01-01T10:00:00Z",
            model_name="test_model",
            variable_name="action",
            value="GREET",
            probability=0.75,
            evidence_count=10,
        )
        assert posterior_id > 0

        records = db.get_posteriors(model_name="test_model", limit=10)
        assert len(records) == 1

        rec = records[0]
        assert rec.id == posterior_id
        assert rec.snapshot_at == "2026-01-01T10:00:00Z"
        assert rec.model_name == "test_model"
        assert rec.variable_name == "action"
        assert rec.value == "GREET"
        assert rec.probability == 0.75
        assert rec.evidence_count == 10


class TestGetInferencesWithFilters:
    def test_get_inferences_with_filters(self, db):
        """Filter by model_name and by since timestamp."""
        # Insert two records at different times.
        db.insert_inference(
            run_at="2026-01-01T09:00:00Z",
            model_name="model_a",
            target="action",
            evidence={},
            action="ACT",
            reason="test",
            confidence=0.5,
            trace={},
        )
        db.insert_inference(
            run_at="2026-01-01T11:00:00Z",
            model_name="model_b",
            target="action",
            evidence={},
            action="WAIT",
            reason="test",
            confidence=0.6,
            trace={},
        )
        db.insert_inference(
            run_at="2026-01-01T12:00:00Z",
            model_name="model_a",
            target="action",
            evidence={},
            action="ACT2",
            reason="test",
            confidence=0.7,
            trace={},
        )

        # Filter by model_name = model_a should return 2 records.
        records = db.get_inferences(model_name="model_a", limit=10)
        assert len(records) == 2
        assert all(r.model_name == "model_a" for r in records)

        # Filter by since = 2026-01-01T10:30:00Z should return 2 newer records.
        records = db.get_inferences(since="2026-01-01T10:30:00Z", limit=10)
        assert len(records) == 2
        assert all(r.run_at >= "2026-01-01T10:30:00Z" for r in records)

        # Combined filter.
        records = db.get_inferences(
            model_name="model_a", since="2026-01-01T11:30:00Z", limit=10
        )
        assert len(records) == 1
        assert records[0].model_name == "model_a"
        assert records[0].action == "ACT2"


class TestGetOutcomesFilterByTaskId:
    def test_get_outcomes_filter_by_task_id(self, db):
        """Only outcomes for the specified task_id are returned."""
        db.insert_outcome(
            inferred_at="2026-01-01T10:00:00Z",
            observed_at="2026-01-01T10:01:00Z",
            task_id="task-A",
            action="ACT",
            success=True,
            context=None,
        )
        db.insert_outcome(
            inferred_at="2026-01-01T10:02:00Z",
            observed_at="2026-01-01T10:03:00Z",
            task_id="task-B",
            action="WAIT",
            success=False,
            context=None,
        )

        records = db.get_outcomes(task_id="task-A", limit=10)
        assert len(records) == 1
        assert records[0].task_id == "task-A"
        assert records[0].action == "ACT"

        records = db.get_outcomes(task_id="task-B", limit=10)
        assert len(records) == 1
        assert records[0].task_id == "task-B"
        assert records[0].action == "WAIT"

        # Unknown task_id returns empty list.
        records = db.get_outcomes(task_id="nonexistent", limit=10)
        assert records == []


class TestClose:
    def test_close_no_error(self, db_path):
        """close() must not raise even when called multiple times."""
        db = SchedulerDB(db_path=db_path)
        db.close()  # First close
        db.close()  # Second close — must not raise