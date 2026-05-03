"""Tests for the production policy wrapper."""

import json
import os
import tempfile

import pytest

from bayesian_engine import Factor, Model, Variable
from bayesian_engine.io import export_model
from bayesian_engine.operators import table
from bayesian_engine.policy import BayesianPolicy, PolicyDecision, select_action


class TestPolicyDecision:
    def test_fields(self):
        d = PolicyDecision(
            action="SELF_MAINTAIN", reason="test", confidence=1.0, trace={}
        )
        assert d.action == "SELF_MAINTAIN"
        assert d.reason == "test"
        assert d.confidence == 1.0
        assert d.trace == {}

    def test_immutable(self):
        d = PolicyDecision(
            action="ACT", reason="ok", confidence=0.8, trace={"a": 1}
        )
        with pytest.raises(Exception):
            d.action = "other"


class TestSelectAction:
    dist = [
        {"value": "A", "probability": 0.6},
        {"value": "B", "probability": 0.3},
        {"value": "C", "probability": 0.1},
    ]

    def test_argmax_above_threshold(self):
        action, confidence, reason = select_action(self.dist, threshold=0.35)
        assert action == "A"
        assert confidence == 0.6
        assert reason == "argmax_above_threshold"

    def test_below_threshold_fallback(self):
        action, confidence, reason = select_action(self.dist, threshold=0.7)
        assert action == "SELF_MAINTAIN"
        assert confidence == 0.6
        assert reason == "below_threshold"

    def test_custom_fallback(self):
        action, confidence, reason = select_action(
            self.dist, threshold=0.7, fallback_action="WAIT"
        )
        assert action == "WAIT"
        assert confidence == 0.6
        assert reason == "below_threshold"

    def test_exactly_at_threshold(self):
        dist = [{"value": "X", "probability": 0.35}, {"value": "Y", "probability": 0.65}]
        action, confidence, reason = select_action(dist, threshold=0.65)
        assert action == "Y"
        assert confidence == 0.65
        assert reason == "argmax_above_threshold"


def _make_temp_model():
    """Build and export a simple model to a temp file."""
    m = Model("test")
    m.add_variable(Variable("day", domain=["weekday", "weekend"]))
    m.add_variable(Variable("action", domain=["GREET", "SILENT"]))
    m.add_factor(Factor(
        inputs=["day"], output="action",
        weight_function=table({
            ("weekday", "GREET"): 0.3, ("weekday", "SILENT"): 0.7,
            ("weekend", "GREET"): 0.6, ("weekend", "SILENT"): 0.4,
        }),
    ))
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    path = tmp.name
    tmp.close()
    export_model(m, path)
    return path


def _make_temp_model_with_metadata():
    """Build and export a model with metadata to a temp file."""
    m = Model("test", metadata={"model_version": "0.1.0", "description": "test model"})
    m.add_variable(Variable("day", domain=["weekday", "weekend"]))
    m.add_variable(Variable("action", domain=["GREET", "SILENT"]))
    m.add_factor(Factor(
        inputs=["day"], output="action",
        weight_function=table({
            ("weekday", "GREET"): 0.3, ("weekday", "SILENT"): 0.7,
            ("weekend", "GREET"): 0.6, ("weekend", "SILENT"): 0.4,
        }),
    ))
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    path = tmp.name
    tmp.close()
    export_model(m, path)
    return path


class TestBayesianPolicyLoad:
    def test_load_valid_model(self):
        path = _make_temp_model()
        try:
            policy = BayesianPolicy.load(path)
            assert policy._model is not None
            assert policy._error is None
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        policy = BayesianPolicy.load("/nonexistent/model.json")
        assert policy._model is None
        assert policy._error is not None

    def test_load_invalid_json(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("not json")
        path = tmp.name
        tmp.close()
        try:
            policy = BayesianPolicy.load(path)
            assert policy._model is None
        finally:
            os.unlink(path)

    def test_load_invalid_schema(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump({"bad": "data"}, tmp)
        path = tmp.name
        tmp.close()
        try:
            policy = BayesianPolicy.load(path)
            assert policy._model is None
        finally:
            os.unlink(path)


class TestBayesianPolicyDecide:
    def test_decide_successful(self):
        path = _make_temp_model()
        try:
            policy = BayesianPolicy.load(path)
            decision = policy.decide(
                target="action", evidence={"day": "weekend"}, threshold=0.5
            )
            assert decision.action == "GREET"
            assert decision.reason == "argmax_above_threshold"
            assert decision.confidence == 0.6
            assert "trace" in decision.trace or isinstance(decision.trace, dict)
        finally:
            os.unlink(path)

    def test_decide_low_confidence_fallback(self):
        path = _make_temp_model()
        try:
            policy = BayesianPolicy.load(path)
            decision = policy.decide(
                target="action", evidence={"day": "weekday"}, threshold=0.8
            )
            assert decision.action == "SELF_MAINTAIN"
            assert decision.reason == "below_threshold"
        finally:
            os.unlink(path)

    def test_decide_model_not_loaded(self):
        policy = BayesianPolicy.load("/nonexistent/model.json")
        decision = policy.decide(target="action", evidence={})
        assert decision.action == "SELF_MAINTAIN"
        assert decision.reason == "model_load_error"
        assert decision.confidence == 1.0
        assert "error" in decision.trace

    def test_decide_missing_target(self):
        path = _make_temp_model()
        try:
            policy = BayesianPolicy.load(path)
            decision = policy.decide(target="nonexistent", evidence={})
            assert decision.action == "SELF_MAINTAIN"
            assert decision.reason == "query_error"
            assert decision.confidence == 1.0
        finally:
            os.unlink(path)

    def test_decide_invalid_evidence(self):
        path = _make_temp_model()
        try:
            policy = BayesianPolicy.load(path)
            decision = policy.decide(
                target="action", evidence={"day": "invalid_value"}
            )
            assert decision.action == "SELF_MAINTAIN"
            assert decision.reason == "inference_error"
        finally:
            os.unlink(path)

    def test_decide_trace_is_json_serializable(self):
        path = _make_temp_model()
        try:
            policy = BayesianPolicy.load(path)
            decision = policy.decide(target="action", evidence={"day": "weekend"})
            json.dumps(decision.trace)
        finally:
            os.unlink(path)

    def test_decide_trace_includes_metadata(self):
        path = _make_temp_model_with_metadata()
        try:
            policy = BayesianPolicy.load(path)
            decision = policy.decide(target="action", evidence={"day": "weekend"})
            assert "metadata" in decision.trace
            assert decision.trace["metadata"] == {
                "model_version": "0.1.0",
                "description": "test model",
            }
        finally:
            os.unlink(path)

    def test_decide_fallback_action_never_raises(self):
        """Policy decisions must never propagate exceptions."""
        policy = BayesianPolicy.load("/nonexistent/model.json")
        decision = policy.decide(target="x", evidence={"bad": object()})
        assert decision.action == "SELF_MAINTAIN"
        assert decision.reason == "model_load_error"
