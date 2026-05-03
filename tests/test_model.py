"""Tests for Model container."""

import pytest

from bayesian_engine import Factor, Model, Variable
from bayesian_engine.operators import table


class TestModel:
    def test_add_variable(self):
        m = Model("test")
        v = Variable(name="x", domain=["a", "b"])
        m.add_variable(v)
        assert "x" in m.variables
        assert m.variables["x"] is v

    def test_add_duplicate_variable_raises(self):
        m = Model("test")
        m.add_variable(Variable(name="x", domain=["a", "b"]))
        with pytest.raises(ValueError):
            m.add_variable(Variable(name="x", domain=["a", "b"]))

    def test_add_factor(self):
        m = Model("test")
        m.add_variable(Variable(name="x", domain=["a", "b"], value="a"))
        m.add_variable(Variable(name="y", domain=["c", "d"]))
        f = Factor(
            inputs=["x"], output="y", weight_function=table({("a", "c"): 0.8})
        )
        m.add_factor(f)
        assert len(m.factors) == 1

    def test_add_factor_missing_input_raises(self):
        m = Model("test")
        m.add_variable(Variable(name="y", domain=["c", "d"]))
        f = Factor(inputs=["x"], output="y", weight_function=table({}))
        with pytest.raises(KeyError):
            m.add_factor(f)

    def test_add_factor_missing_output_raises(self):
        m = Model("test")
        m.add_variable(Variable(name="x", domain=["a", "b"]))
        f = Factor(inputs=["x"], output="y", weight_function=table({}))
        with pytest.raises(KeyError):
            m.add_factor(f)

    def test_set_evidence(self):
        m = Model("test")
        m.add_variable(Variable(name="x", domain=["a", "b"], value="a"))
        m.set_evidence({"x": "b"})
        assert m.variables["x"].value == "b"

    def test_set_evidence_missing_variable_raises(self):
        m = Model("test")
        with pytest.raises(KeyError):
            m.set_evidence({"nonexistent": "val"})

    def test_query_missing_target_raises(self):
        m = Model("test")
        with pytest.raises(KeyError):
            m.query("nonexistent")

    def test_model_isolation(self):
        """Factors cannot cross model boundaries."""
        m1 = Model("m1")
        m2 = Model("m2")
        m1.add_variable(Variable(name="x", domain=["a", "b"]))
        m2.add_variable(Variable(name="y", domain=["c", "d"]))
        f = Factor(inputs=["x"], output="y", weight_function=table({}))
        # x is in m1, y is in m2 — should fail when adding to m1
        with pytest.raises(KeyError):
            m1.add_factor(f)
