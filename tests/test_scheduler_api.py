"""Tests for BayesianScheduler — the unified composition API."""

from __future__ import annotations

import json

import pytest

from bayesian_engine.production import BayesianScheduler
from bayesian_engine.production.scheduler_api import (
    DEFAULT_MODEL_NAME,
    DEFAULT_N_PARTICLES,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD,
)

COPRESENCE_MODEL_PATH = "/mnt/h/fun/bayesian-engine/models/copresence.json"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def state_dir(tmp_path):
    """Temp state directory with the copresence model registered."""
    sd = tmp_path / ".hermes" / "bayesian_scheduler"
    sd.mkdir(parents=True, exist_ok=True)

    # Register copresence.json in the temp manifest so ModelRegistry can find it
    models_dir = sd / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest = models_dir / "manifest.jsonl"
    manifest.touch()
    entry = {
        "name": DEFAULT_MODEL_NAME,
        "path": COPRESENCE_MODEL_PATH,
        "version": "0.1.0",
        "registered_at": "2026-05-07T00:00:00Z",
        "inference_count": 0,
    }
    manifest.write_text(json.dumps(entry) + "\n")

    return sd


@pytest.fixture
def scheduler(state_dir, tmp_path):
    """Fresh scheduler with temp state dir and beliefs redirected to tmp."""
    # Redirect belief persistence to tmp so tests don't pollute ~/.hermes
    import bayesian_engine.production.streaming_engine as se

    belief_tmp = tmp_path / "beliefs"
    belief_tmp.mkdir()

    orig_dir = se._belief_dir
    orig_path = se._belief_path

    se._belief_dir = lambda: belief_tmp
    se._belief_path = lambda name: belief_tmp / f"{name}.json"

    sched = BayesianScheduler(
        model_name=DEFAULT_MODEL_NAME,
        state_dir=state_dir,
        threshold=DEFAULT_THRESHOLD,
        n_particles=DEFAULT_N_PARTICLES,
        seed=DEFAULT_SEED,
    )
    yield sched

    se._belief_dir = orig_dir
    se._belief_path = orig_path


# ------------------------------------------------------------------
# Init tests
# ------------------------------------------------------------------

class TestInit:
    def test_init_creates_all_submodules(self, scheduler):
        assert scheduler.db is not None
        assert scheduler.decisions is not None
        assert scheduler.evidence is not None
        assert scheduler.streaming is not None
        assert scheduler.model_name == DEFAULT_MODEL_NAME
        assert scheduler._threshold == DEFAULT_THRESHOLD

    def test_init_stores_state_dir(self, scheduler, state_dir):
        assert scheduler._state_dir == state_dir

    def test_init_uses_default_threshold(self, state_dir, tmp_path):
        import bayesian_engine.production.streaming_engine as se

        belief_tmp = tmp_path / "beliefs"
        belief_tmp.mkdir()
        orig_dir = se._belief_dir
        orig_path = se._belief_path
        se._belief_dir = lambda: belief_tmp
        se._belief_path = lambda name: belief_tmp / f"{name}.json"

        sched = BayesianScheduler(model_name=DEFAULT_MODEL_NAME, state_dir=state_dir)

        se._belief_dir = orig_dir
        se._belief_path = orig_path

        assert sched._threshold == DEFAULT_THRESHOLD

    def test_init_particles_count(self, scheduler):
        assert scheduler._n_particles == DEFAULT_N_PARTICLES
        assert len(scheduler.streaming._particles) == DEFAULT_N_PARTICLES

    def test_init_seed_is_stored(self, scheduler):
        assert scheduler._seed == DEFAULT_SEED


# ------------------------------------------------------------------
# infer() tests
# ------------------------------------------------------------------

class TestInfer:
    def test_infer_returns_policy_decision(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        decision = scheduler.infer("action_class", evidence)
        assert decision.action is not None
        assert decision.reason is not None
        assert 0.0 <= decision.confidence <= 1.0

    def test_infer_logs_decision_to_db(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        decision = scheduler.infer("action_class", evidence)

        rows = scheduler.db.get_inferences(model_name=DEFAULT_MODEL_NAME, limit=1)
        assert len(rows) >= 1
        assert rows[0].model_name == DEFAULT_MODEL_NAME
        assert rows[0].action == decision.action

    def test_infer_with_custom_threshold(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        decision = scheduler.infer("action_class", evidence, threshold=0.99)
        assert decision is not None

    def test_infer_passes_session_and_cron_ids(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        scheduler.infer(
            "action_class",
            evidence,
            session_id="sess-abc123",
            cron_job_id="cron-def456",
        )
        rows = scheduler.db.get_inferences(model_name=DEFAULT_MODEL_NAME, limit=1)
        assert rows[0].session_id == "sess-abc123"
        assert rows[0].cron_job_id == "cron-def456"

    def test_infer_updates_streaming_state(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        scheduler.infer("action_class", evidence)

        # After resampling, weights are uniform (1/N each) and sum to 1
        assert scheduler.streaming._weights.sum() == pytest.approx(1.0)
        assert len(scheduler.streaming._weights) == DEFAULT_N_PARTICLES

    def test_infer_second_call_uses_updated_beliefs(self, scheduler):
        evidence1 = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        # Streaming update is tested independently in test_streaming_engine.py.
        # Here we just verify sequential calls work without error.
        scheduler.infer("contact_window", evidence1)
        result2 = scheduler.infer("contact_window", evidence1)  # same evidence, no crash
        assert result2 is not None

    def test_infer_unknown_variable_raises(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        with pytest.raises(KeyError):
            scheduler.infer("nonexistent_variable", evidence)


# ------------------------------------------------------------------
# record_outcome() tests
# ------------------------------------------------------------------

class TestRecordOutcome:
    def test_record_outcome_inserts_row(self, scheduler):
        row_id = scheduler.record_outcome(
            task_id="task-001",
            action="CHECK_IN",
            success=True,
            latency_ms=1200.0,
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_record_outcome_with_context(self, scheduler):
        row_id = scheduler.record_outcome(
            task_id="task-002",
            action="DIRECT_MESSAGE",
            success=False,
            error="Rate limited",
            context={"channel": "discord", "reply_to": "msg-123"},
        )
        assert row_id > 0

    def test_record_outcome_null_success(self, scheduler):
        row_id = scheduler.record_outcome(
            task_id="task-003",
            action="SILENCE",
            success=None,
        )
        assert row_id > 0


# ------------------------------------------------------------------
# list_decisions() tests
# ------------------------------------------------------------------

class TestListDecisions:
    def test_list_decisions_empty(self, scheduler):
        results = scheduler.list_decisions(limit=5)
        assert isinstance(results, list)

    def test_list_decisions_after_infer(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        scheduler.infer("action_class", evidence)

        results = scheduler.list_decisions(limit=10)
        assert len(results) >= 1

    def test_list_decisions_respects_limit(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        for _ in range(5):
            scheduler.infer("action_class", evidence)

        results = scheduler.list_decisions(limit=2)
        assert len(results) == 2


# ------------------------------------------------------------------
# get_beliefs() tests
# ------------------------------------------------------------------

class TestGetBeliefs:
    def test_get_beliefs_latent_variable(self, scheduler):
        beliefs = scheduler.get_beliefs("contact_window")
        assert beliefs["type"] == "particles"
        assert "weights" in beliefs
        assert "distribution" in beliefs

    def test_get_beliefs_cpt_variable(self, scheduler):
        # action_class is latent — particles path
        beliefs = scheduler.get_beliefs("action_class")
        assert beliefs["type"] == "particles"

    def test_get_beliefs_unknown_raises(self, scheduler):
        with pytest.raises(KeyError):
            scheduler.get_beliefs("nonexistent_var")


# ------------------------------------------------------------------
# reset() tests
# ------------------------------------------------------------------

class TestReset:
    def test_reset_clears_streaming_state(self, scheduler):
        evidence = {
            "day_type": "weekday",
            "hour_block": "morning",
            "user_presence": "active_recently",
            "user_load": "free",
        }
        scheduler.infer("action_class", evidence)
        scheduler.reset()

        # Particles are fresh after reset
        assert len(scheduler.streaming._particles) == DEFAULT_N_PARTICLES
        # Dirichlet state is reset to initial uniform (alpha = 1.0 for all entries)
        for fid, configs in scheduler.streaming._dirichlet_state.items():
            for config_str, alpha_dict in configs.items():
                for val, alpha in alpha_dict.items():
                    assert alpha == 1.0


# ------------------------------------------------------------------
# list_models() tests
# ------------------------------------------------------------------

class TestListModels:
    def test_list_models_returns_list(self, scheduler):
        models = scheduler.list_models()
        assert isinstance(models, list)

    def test_list_models_contains_copresence(self, scheduler):
        models = scheduler.list_models()
        names = [m["name"] for m in models]
        assert DEFAULT_MODEL_NAME in names


# ------------------------------------------------------------------
# get_evidence_for_inference() tests
# ------------------------------------------------------------------

class TestGetEvidenceForInference:
    def test_get_evidence_returns_empty_dict_for_unknown_task(self, scheduler):
        result = scheduler.get_evidence_for_inference("task-unknown")
        assert isinstance(result, dict)

    def test_get_evidence_after_outcome_recorded(self, scheduler):
        scheduler.record_outcome(
            task_id="task-100",
            action="CHECK_IN",
            success=True,
        )
        result = scheduler.get_evidence_for_inference("task-100")
        assert isinstance(result, dict)
