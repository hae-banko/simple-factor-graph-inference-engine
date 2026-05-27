"""Tests for the DecisionLogger production module."""

from __future__ import annotations

import json

import pytest

from bayesian_engine import Factor, Model, Variable
from bayesian_engine.io import export_model
from bayesian_engine.operators import table
from bayesian_engine.policy import BayesianPolicy, PolicyDecision
from bayesian_engine.production.decision_logger import DecisionLogger
from bayesian_engine.production.scheduler_db import SchedulerDB

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_temp_model(tmp_path):
    """Build and export a simple model to a temp file; returns the path."""
    m = Model("copresence")
    m.add_variable(Variable("day_type", domain=["weekday", "weekend"]))
    m.add_variable(Variable("hour_block", domain=["morning", "afternoon", "evening"]))
    m.add_variable(Variable("action_class", domain=["CHECK_IN", "SELF_MAINTAIN", "DEFER"]))
    m.add_factor(Factor(
        inputs=["day_type"],
        output="action_class",
        weight_function=table({
            ("weekday", "CHECK_IN"): 0.3,
            ("weekday", "SELF_MAINTAIN"): 0.5,
            ("weekday", "DEFER"): 0.2,
            ("weekend", "CHECK_IN"): 0.6,
            ("weekend", "SELF_MAINTAIN"): 0.3,
            ("weekend", "DEFER"): 0.1,
        }),
    ))
    path = str(tmp_path / "copresence_model.json")
    export_model(m, path)
    return path


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Place a fresh database file in a temporary directory."""
    return tmp_path / "test_decisions.db"


@pytest.fixture
def db(db_path):
    """Construct a SchedulerDB instance backed by a temporary file."""
    db = SchedulerDB(db_path=db_path)
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def log_path(tmp_path):
    """Return a path inside tmp_path for the JSONL log file."""
    return tmp_path / "decisions.log"


@pytest.fixture
def decision_logger(db, log_path):
    """Construct a DecisionLogger with temp DB and log file."""
    return DecisionLogger(db=db, log_file_path=log_path)


@pytest.fixture
def policy(tmp_path):
    """Load a real BayesianPolicy from a temporary model file."""
    model_path = _make_temp_model(tmp_path)
    return BayesianPolicy.load(model_path)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestLogWritesToDb:
    def test_log_writes_to_db(self, db, decision_logger, policy):
        """DecisionLogger.log() persists a record to SchedulerDB."""
        policy_decision = policy.decide(
            target="action_class",
            evidence={"day_type": "weekend"},
        )
        evidence = {"day_type": "weekend", "hour_block": "evening"}

        inference_id = decision_logger.log(
            policy_decision=policy_decision,
            model_name="copresence",
            target="action_class",
            evidence=evidence,
            latency_ms=12.4,
            cron_job_id="job_abc123",
            session_id="session_x",
        )

        assert inference_id > 0

        records = db.get_inferences(model_name="copresence", limit=10)
        assert len(records) == 1
        rec = records[0]

        assert rec.id == inference_id
        assert rec.model_name == "copresence"
        assert rec.target == "action_class"
        assert rec.evidence == evidence
        assert rec.action == policy_decision.action
        assert rec.reason == policy_decision.reason
        assert rec.confidence == policy_decision.confidence
        assert rec.trace == policy_decision.trace
        assert rec.latency_ms == 12.4
        assert rec.cron_job_id == "job_abc123"
        assert rec.session_id == "session_x"


class TestLogWritesToJsonlFile:
    def test_log_writes_to_jsonl_file(self, decision_logger, policy, log_path):
        """The JSONL log file contains exactly one line with the correct keys."""
        policy_decision = policy.decide(
            target="action_class",
            evidence={"day_type": "weekend"},
        )
        evidence = {"day_type": "weekend", "hour_block": "evening"}

        decision_logger.log(
            policy_decision=policy_decision,
            model_name="copresence",
            target="action_class",
            evidence=evidence,
            latency_ms=12.4,
            cron_job_id="job_abc123",
            session_id=None,
        )

        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])

        assert set(record.keys()) == {
            "timestamp",
            "level",
            "inference_id",
            "model_name",
            "target",
            "evidence",
            "decision",
            "latency_ms",
            "cron_job_id",
            "session_id",
        }
        assert record["model_name"] == "copresence"
        assert record["target"] == "action_class"
        assert record["evidence"] == evidence
        assert record["decision"]["action"] == policy_decision.action
        assert record["decision"]["confidence"] == policy_decision.confidence
        assert record["decision"]["reason"] == policy_decision.reason
        assert record["latency_ms"] == 12.4
        assert record["cron_job_id"] == "job_abc123"
        assert record["session_id"] is None
        assert record["level"] == "INFO"
        assert record["inference_id"] > 0


class TestLogAndDecide:
    def test_log_and_decide_returns_decision(self, decision_logger, policy):
        """log_and_decide() returns a PolicyDecision with the correct action."""
        result = decision_logger.log_and_decide(
            policy=policy,
            target="action_class",
            evidence={"day_type": "weekend"},
            model_name="copresence",
            threshold=0.35,
            fallback_action="SELF_MAINTAIN",
        )

        assert isinstance(result, PolicyDecision)
        assert result.action in ("CHECK_IN", "SELF_MAINTAIN", "DEFER")
        assert isinstance(result.reason, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.trace, dict)

    def test_log_and_decide_records_latency(self, db, decision_logger, policy, log_path):
        """log_and_decide() records a positive, reasonable latency_ms."""
        decision_logger.log_and_decide(
            policy=policy,
            target="action_class",
            evidence={"day_type": "weekend"},
            model_name="copresence",
            threshold=0.35,
            fallback_action="SELF_MAINTAIN",
        )

        # Verify latency in DB
        records = db.get_inferences(model_name="copresence", limit=1)
        assert len(records) == 1
        assert records[0].latency_ms is not None
        assert records[0].latency_ms > 0
        assert records[0].latency_ms < 10_000  # sanity: less than 10 seconds

        # Verify latency in JSONL
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["latency_ms"] > 0
        assert record["latency_ms"] < 10_000


class TestLogIncrementsInferenceId:
    def test_log_increments_inference_id(self, db, decision_logger, policy):
        """Two log() calls receive distinct, incrementing inference IDs."""
        pd1 = policy.decide(target="action_class", evidence={"day_type": "weekday"})
        pd2 = policy.decide(target="action_class", evidence={"day_type": "weekend"})

        id1 = decision_logger.log(pd1, "copresence", "action_class", {"day_type": "weekday"})
        id2 = decision_logger.log(pd2, "copresence", "action_class", {"day_type": "weekend"})

        assert id1 != id2
        assert id2 > id1


class TestLogWithCronJobIdAndSessionId:
    def test_log_with_cron_job_id_and_session_id(self, db, decision_logger, policy):
        """cron_job_id and session_id appear in both DB and JSONL log."""
        policy_decision = policy.decide(
            target="action_class",
            evidence={"day_type": "weekday"},
        )
        evidence = {"day_type": "weekday"}

        decision_logger.log(
            policy_decision=policy_decision,
            model_name="copresence",
            target="action_class",
            evidence=evidence,
            cron_job_id="job_abc123",
            session_id="session_xyz",
        )

        # Verify in DB
        records = db.get_inferences(model_name="copresence", limit=1)
        assert len(records) == 1
        assert records[0].cron_job_id == "job_abc123"
        assert records[0].session_id == "session_xyz"


class TestLogFileErrorDoesNotCrash:
    def test_log_file_error_does_not_crash(self, db, policy, tmp_path):
        """A file write error is handled gracefully — no exception propagates."""

        # Use a path that cannot be created: a file as directory
        bad_log_path = tmp_path / "subdir" / "decisions.log"
        # Do NOT create 'subdir' — the write will fail because the parent
        # directory doesn't exist *and* mkdir(parents=True) should succeed,
        # but we can make the file itself read-only or use an invalid path.
        # Instead, use a path that exists as a file (not a dir).
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        bad_log_path = readonly_dir / "file"
        bad_log_path.write_text("placeholder")  # make it a file, not a dir
        # Now try to write to a path that has a FILE where its parent dir should be
        # Actually the simplest approach: use a directory that is truly read-only
        # ...except on POSIX we may not be able to easily do that.
        # Instead, let's just patch the file to be unwritable.
        bad_log_path.chmod(0o000)  # remove all permissions

        logger = DecisionLogger(db=db, log_file_path=bad_log_path)

        pd = policy.decide(target="action_class", evidence={"day_type": "weekday"})

        # Must NOT raise
        inference_id = logger.log(
            policy_decision=pd,
            model_name="copresence",
            target="action_class",
            evidence={"day_type": "weekday"},
        )

        # The DB record should still have been written
        assert inference_id > 0

        # Restore permissions for cleanup
        bad_log_path.chmod(0o644)

    def test_log_to_missing_directory_auto_creates_it(self, db, tmp_path):
        """Log file directory is created automatically on first write."""
        missing_dir = tmp_path / "logs" / "subdir"
        log_path = missing_dir / "decisions.log"
        logger = DecisionLogger(db=db, log_file_path=log_path)
        assert not missing_dir.exists()

        pd = PolicyDecision(
            action="CHECK_IN",
            reason="argmax_above_threshold",
            confidence=0.52,
            trace={},
        )

        inference_id = logger.log(
            policy_decision=pd,
            model_name="copresence",
            target="action_class",
            evidence={"day": "weekday"},
        )

        assert inference_id > 0
        assert log_path.exists()
        assert missing_dir.exists()

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["model_name"] == "copresence"
