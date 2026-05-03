"""Tests for Factor."""

import numpy as np
import pytest

from bayesian_engine.core.factor import Factor
from bayesian_engine.operators import binary, product, sigmoid, table
from bayesian_engine.operators.combinators import max as c_max
from bayesian_engine.operators.combinators import sum as c_sum


class TestFactor:
    def test_construction(self):
        f = Factor(inputs=["a"], output="b", weight_function=table({("x", "y"): 1.0}))
        assert f.inputs == ["a"]
        assert f.output == "b"

    def test_empty_inputs_raises(self):
        with pytest.raises(ValueError):
            Factor(inputs=[], output="b", weight_function=lambda: 1.0)

    def test_empty_output_raises(self):
        with pytest.raises(ValueError):
            Factor(inputs=["a"], output="", weight_function=lambda: 1.0)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            Factor(inputs=["a"], output="b", weight_function="not_callable")  # type: ignore

    def test_call_delegates_to_weight_function(self):
        f = Factor(inputs=["a"], output="b", weight_function=lambda *vs: sum(float(v) for v in vs))
        assert f(1.0, 2.0, 3.0) == 6.0


class TestTableOperator:
    def test_cpt_lookup(self):
        cpt = {("a", "x"): 0.8, ("a", "y"): 0.2, ("b", "x"): 0.3, ("b", "y"): 0.7}
        f = table(cpt)
        assert f("a", "x") == 0.8
        assert f("b", "y") == 0.7
        assert f("z", "w") == 0.0

    def test_metadata(self):
        cpt = {("a",): 1.0}
        f = table(cpt)
        assert f.__operator_name__ == "table"
        assert f.__operator_params__ == {"cpt": cpt}


class TestSigmoidOperator:
    def test_output(self):
        f = sigmoid(w=np.array([5.0]), sigma=0.1)
        result = f(0.7, 0.97)
        assert 0.8 < result <= 1.0

    def test_mismatched_inputs_raises(self):
        f = sigmoid(w=np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            f(1.0, 2.0)  # only 1 input + 1 output = 2 values, but w expects 2 inputs

    def test_metadata(self):
        f = sigmoid(w=np.array([1.0]), sigma=0.2)
        assert f.__operator_name__ == "sigmoid"
        assert "w" in f.__operator_params__
        assert f.__operator_params__["sigma"] == 0.2


class TestBinaryOperator:
    def test_match(self):
        f = binary()
        assert f("yes", "yes") == 1.0

    def test_mismatch(self):
        f = binary()
        assert f("yes", "no", "x") == 0.0


class TestCombinators:
    def test_product(self):
        f = product()
        assert f(2.0, 3.0, 4.0) == 24.0

    def test_c_sum(self):
        f = c_sum()
        assert f(1.0, 2.0, 3.0) == 6.0

    def test_c_max(self):
        f = c_max()
        assert f(1.0, 5.0, 3.0) == 5.0

    def test_c_max_empty(self):
        f = c_max()
        assert f() == 0.0
