"""Tests for factor graph construction and operations."""

import pytest

from bayesian_engine.core.factor import Factor
from bayesian_engine.core.graph import build_factor_graph, detect_cycles, extract_subgraph
from bayesian_engine.core.variable import Variable
from bayesian_engine.operators import table


def make_vars_and_factors():
    """Create a simple A → B → C chain."""
    va = Variable(name="a", domain=["on", "off"], value="on")
    vb = Variable(name="b", domain=["high", "low"])
    vc = Variable(name="c", domain=["yes", "no"])
    variables = {"a": va, "b": vb, "c": vc}

    f1 = Factor(
        inputs=["a"], output="b", weight_function=table({("on", "high"): 0.9})
    )
    f2 = Factor(
        inputs=["b"], output="c", weight_function=table({("high", "yes"): 0.7})
    )
    factors = [f1, f2]
    return variables, factors


class TestBuildFactorGraph:
    def test_populates_adjacency(self):
        variables, factors = make_vars_and_factors()
        build_factor_graph(variables, factors)

        va = variables["a"]
        vb = variables["b"]
        vc = variables["c"]

        assert len(va.incoming_factors) == 0
        assert len(va.outgoing_factors) == 1
        assert va.outgoing_factors[0].output == "b"

        assert len(vb.incoming_factors) == 1
        assert len(vb.outgoing_factors) == 1
        assert vb.incoming_factors[0].output == "b"
        assert vb.outgoing_factors[0].output == "c"

        assert len(vc.incoming_factors) == 1
        assert len(vc.outgoing_factors) == 0

    def test_missing_output_variable_raises(self):
        va = Variable(name="a", domain=["on", "off"])
        variables = {"a": va}
        f = Factor(inputs=["a"], output="b", weight_function=table({}))
        with pytest.raises(KeyError):
            build_factor_graph(variables, [f])

    def test_missing_input_variable_raises(self):
        vb = Variable(name="b", domain=["high", "low"])
        variables = {"b": vb}
        f = Factor(inputs=["a"], output="b", weight_function=table({}))
        with pytest.raises(KeyError):
            build_factor_graph(variables, [f])


class TestDetectCycles:
    def test_acyclic_graph(self):
        variables, factors = make_vars_and_factors()
        build_factor_graph(variables, factors)
        cycles = detect_cycles(variables)
        assert cycles == []

    def test_self_loop(self):
        """A factor where a variable is both input and output of same/different factors
        creating a cycle."""
        va = Variable(name="a", domain=["on", "off"])
        vb = Variable(name="b", domain=["high", "low"])
        variables = {"a": va, "b": vb}
        f1 = Factor(inputs=["a"], output="b", weight_function=table({}))
        f2 = Factor(inputs=["b"], output="a", weight_function=table({}))
        build_factor_graph(variables, [f1, f2])
        cycles = detect_cycles(variables)
        assert len(cycles) > 0


class TestExtractSubgraph:
    def test_reachable_variables(self):
        variables, factors = make_vars_and_factors()
        build_factor_graph(variables, factors)

        rel_vars, rel_factors = extract_subgraph(variables, "c", set())
        assert "a" in rel_vars
        assert "b" in rel_vars
        assert "c" in rel_vars
        assert len(rel_factors) == 2

    def test_evidence_variables_excluded(self):
        variables, factors = make_vars_and_factors()
        build_factor_graph(variables, factors)

        rel_vars, rel_factors = extract_subgraph(variables, "c", {"a"})
        assert "a" not in rel_vars
        assert "b" in rel_vars
        assert "c" in rel_vars

    def test_target_not_found_raises(self):
        variables, factors = make_vars_and_factors()
        build_factor_graph(variables, factors)
        with pytest.raises(KeyError):
            extract_subgraph(variables, "nonexistent", set())
