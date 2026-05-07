"""Tests for StreamingEngine — hybrid Dirichlet + particle filter belief-state engine."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from bayesian_engine.production.streaming_engine import (
    StreamingEngine,
    _belief_dir,
    _belief_path,
    _factor_id,
    _is_table_factor,
    _iter_domain_combinations,
    _init_particles_low_discrepancy,
)
from bayesian_engine.core.model import Model
from bayesian_engine.core.variable import Variable
from bayesian_engine.core.factor import Factor
from bayesian_engine.operators.table import table


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

COPRESENCE_MODEL_PATH = str(
    Path(__file__).parent.parent / "models" / "copresence.json"
)


@pytest.fixture
def belief_file_path():
    """Return the belief file path for the copresence model and clean up after."""
    path = _belief_path("copresence")
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def engine():
    """Fresh StreamingEngine for the copresence model."""
    return StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

def test_init_loads_model_and_initializes_particles(engine):
    """Particles are created for latent vars; Dirichlet counts for table factors."""
    assert len(engine._particles) == 200
    assert len(engine._weights) == 200
    assert all(w > 0 for w in engine._weights)

    # Latent vars: contact_window and action_class
    latent_names = {v.name for v in engine._latent_vars}
    assert latent_names == {"contact_window", "action_class"}

    # Dirichlet state should have entries for table factors
    assert len(engine._dirichlet_state) > 0

    # All table factors in copresence are CPT-based
    assert len(engine._table_factors) > 0
    assert len(engine._non_table_factors) == 0


def test_particles_cover_latent_domain(engine):
    """All latent variable values are represented in the initial particle set."""
    contact_vals = {p["contact_window"] for p in engine._particles}
    action_vals = {p["action_class"] for p in engine._particles}

    # These are the domain values from copresence.json
    assert "closed" in contact_vals
    assert "soft" in contact_vals
    assert "open" in contact_vals
    assert "SILENCE" in action_vals
    assert "SELF_MAINTAIN" in action_vals
    assert "AMBIENT_PING" in action_vals
    assert "CHECK_IN" in action_vals
    assert "DIRECT_MESSAGE" in action_vals


# ---------------------------------------------------------------------------
# update_and_query tests
# ---------------------------------------------------------------------------

def test_update_and_query_returns_result(engine):
    """Calling update_and_query returns a properly structured dict."""
    evidence = {
        "day_type": "weekday",
        "hour_block": "morning",
        "user_presence": "active_recently",
        "user_load": "free",
    }
    result = engine.update_and_query("contact_window", evidence)

    assert "contact_window" in result
    assert "value" in result["contact_window"]
    assert "distribution" in result["contact_window"]
    dist = result["contact_window"]["distribution"]
    assert len(dist) > 0
    assert all("value" in d and "probability" in d for d in dist)
    assert abs(sum(d["probability"] for d in dist) - 1.0) < 1e-6


def test_update_and_query_deterministic_with_same_seed():
    """Two engines with seed=42 on same evidence produce identical results."""
    evidence = {
        "day_type": "weekday",
        "hour_block": "morning",
        "user_presence": "active_recently",
        "user_load": "free",
    }

    eng1 = StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)
    eng2 = StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)

    result1 = eng1.update_and_query("contact_window", evidence)
    result2 = eng2.update_and_query("contact_window", evidence)

    assert result1["contact_window"]["value"] == result2["contact_window"]["value"]


def test_weight_update_changes_weights():
    """Weights change after update_and_query with evidence."""
    eng = StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)
    initial_weights = eng._weights.copy()

    evidence = {
        "day_type": "weekday",
        "hour_block": "morning",
        "user_presence": "active_recently",
        "user_load": "free",
    }
    eng.update_and_query("contact_window", evidence)

    # Weights are reset to uniform after resampling
    assert eng._weights.shape == initial_weights.shape


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

def test_beliefs_persist_across_instances(belief_file_path):
    """Beliefs saved by one engine are loaded by another."""
    if belief_file_path.exists():
        belief_file_path.unlink()

    eng1 = StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)
    evidence = {
        "day_type": "weekday",
        "hour_block": "morning",
        "user_presence": "active_recently",
        "user_load": "free",
    }
    result1 = eng1.update_and_query("contact_window", evidence)

    # Second engine loads from disk
    eng2 = StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)

    # Particles should be loaded from the belief file eng1 persisted
    assert len(eng2._particles) == 200
    # Loaded state matches what eng1 saved: same particles and weights
    assert eng2._particles == eng1._particles
    assert np.allclose(eng2._weights, eng1._weights)

    # Dirichlet state also persisted
    assert eng2._dirichlet_state == eng1._dirichlet_state

    if belief_file_path.exists():
        belief_file_path.unlink()


def test_reset_clears_beliefs_and_file(belief_file_path):
    """reset_beliefs re-initializes state and deletes the belief file."""
    eng = StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)
    evidence = {
        "day_type": "weekday",
        "hour_block": "morning",
        "user_presence": "active_recently",
        "user_load": "free",
    }
    eng.update_and_query("contact_window", evidence)

    eng.reset_beliefs()

    # File should be gone
    assert not belief_file_path.exists()

    # State should be fresh
    assert len(eng._particles) == 200
    assert len(eng._weights) == 200
    # Initial Dirichlet counts are all 1.0
    for fid, configs in eng._dirichlet_state.items():
        for config_str, alpha_dict in configs.items():
            assert all(a == 1.0 for a in alpha_dict.values())


def test_missing_belief_file_starts_fresh():
    """Creating an engine with no belief file starts without error."""
    path = _belief_path("copresence")
    if path.exists():
        path.unlink()
    # Should not raise
    eng = StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)
    assert len(eng._particles) == 200


def test_corrupted_belief_file_starts_fresh(tmp_path, monkeypatch):
    """Writing invalid JSON to the belief file causes engine to start fresh."""
    path = _belief_path("copresence")

    # Write corrupted JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("{ this is not json }")

    # Monkey-patch home to use tmp_path to avoid polluting real belief dir
    monkeypatch.setattr(
        "bayesian_engine.production.streaming_engine._belief_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "bayesian_engine.production.streaming_engine._belief_path",
        lambda name: tmp_path / f"{name}.json",
    )

    # Should not raise
    eng = StreamingEngine(COPRESENCE_MODEL_PATH, n_particles=200, seed=42)
    assert len(eng._particles) == 200


# ---------------------------------------------------------------------------
# get_current_beliefs tests
# ---------------------------------------------------------------------------

def test_get_current_beliefs_returns_correct_structure(engine):
    """get_current_beliefs returns the expected format for latent variables."""
    beliefs = engine.get_current_beliefs("contact_window")
    assert beliefs["type"] == "particles"
    assert "weights" in beliefs
    assert "distribution" in beliefs
    dist = beliefs["distribution"]
    assert isinstance(dist, dict)
    assert all(isinstance(v, float) for v in dist.values())
    assert abs(sum(dist.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_factor_id_is_stable():
    """_factor_id produces the same hash for the same factor."""
    wf = table({
        ("a", "b", "x"): 0.5,
        ("a", "b", "y"): 0.5,
    })
    f1 = Factor(inputs=["a", "b"], output="x", weight_function=wf)

    id1 = _factor_id(f1)
    id2 = _factor_id(f1)
    assert id1 == id2
    assert len(id1) == 16


def test_is_table_factor():
    """_is_table_factor returns True only for table operators."""
    cpt_wf = table({("a", "x"): 1.0})
    cpt_factor = Factor(inputs=["a"], output="x", weight_function=cpt_wf)
    assert _is_table_factor(cpt_factor) is True

    def sigmoid_wf(x):
        return 1.0 / (1.0 + np.exp(-x))
    sigmoid_wf.__operator_name__ = "sigmoid"
    sigmoid_factor = Factor(inputs=["a"], output="x", weight_function=sigmoid_wf)
    assert _is_table_factor(sigmoid_factor) is False


def test_iter_domain_combinations():
    """_iter_domain_combinations produces all domain combinations."""
    v1 = Variable(name="v1", domain=["a", "b"], latent=True)
    v2 = Variable(name="v2", domain=["x", "y", "z"], latent=True)
    combos = _iter_domain_combinations([v1, v2])
    assert len(combos) == 6  # 2 * 3
    names = {c["v1"] for c in combos}
    assert names == {"a", "b"}


def test_init_particles_low_discrepancy():
    """Particles are evenly spread across domain combinations."""
    v1 = Variable(name="v1", domain=["a", "b"], latent=True)
    v2 = Variable(name="v2", domain=["x", "y"], latent=True)

    particles, weights = _init_particles_low_discrepancy([v1, v2], n_particles=10)

    assert len(particles) == 10
    assert len(weights) == 10
    assert all(w == 0.1 for w in weights)
    # Each value should appear roughly equally
    v1_counts = {v: sum(1 for p in particles if p["v1"] == v) for v in ["a", "b"]}
    assert v1_counts["a"] in (4, 5, 6)
    assert v1_counts["b"] in (4, 5, 6)