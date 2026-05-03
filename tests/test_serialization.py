"""Tests for model serialization round-trip."""

import os
import tempfile

import numpy as np
import pytest

from bayesian_engine import Factor, Model, Variable
from bayesian_engine.io import (
    SCHEMA_VERSION,
    ModelSchemaError,
    export_model,
    import_model,
    model_from_dict,
    model_from_json,
    model_to_dict,
    model_to_json,
)
from bayesian_engine.operators import (
    binary,
    gaussian,
    product,
    sigmoid,
    threshold,
)
from bayesian_engine.operators import (
    max as max_w,
)
from bayesian_engine.operators import (
    sum as sum_w,
)
from tests.fixtures.initiative_model import make_chain_model, make_weather_model


class TestDictRoundTrip:
    def test_weather_model(self):
        m = make_weather_model()
        r1 = m.query("rain", evidence={"cloudy": "yes"})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("rain", evidence={"cloudy": "yes"})
        assert r1 == r2

    def test_chain_model(self):
        m = make_chain_model()
        r1 = m.query("c", evidence={"a": "on"})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("c", evidence={"a": "on"})
        assert r1 == r2

    def test_schema_version(self):
        m = make_weather_model()
        data = model_to_dict(m)
        assert "schema_version" in data
        assert data["schema_version"] == SCHEMA_VERSION


class TestMetadata:
    def test_metadata_round_trip(self):
        m = Model("test", metadata={"model_version": "0.1.0", "description": "test"})
        m.add_variable(Variable("x", domain=["a", "b"]))
        m.add_variable(Variable("y", domain=["a", "b"]))
        m.add_factor(Factor(inputs=["x"], output="y", weight_function=binary()))
        data = model_to_dict(m)
        assert data["metadata"] == {"model_version": "0.1.0", "description": "test"}
        m2 = model_from_dict(data)
        assert m2.metadata == {"model_version": "0.1.0", "description": "test"}

    def test_no_metadata_still_works(self):
        m = Model("test")
        m.add_variable(Variable("x", domain=["a", "b"]))
        data = model_to_dict(m)
        assert "metadata" not in data
        m2 = model_from_dict(data)
        assert m2.metadata == {}


class TestSchemaValidation:
    base = {"schema_version": "0.1", "name": "test", "variables": [], "factors": []}

    def test_valid_minimal(self):
        model_from_dict(self.base)

    def test_missing_schema_version(self):
        with pytest.raises(ModelSchemaError, match="schema_version"):
            model_from_dict({"name": "x", "variables": [], "factors": []})

    def test_unsupported_schema_version(self):
        with pytest.raises(ModelSchemaError, match="Unsupported schema version"):
            model_from_dict({**self.base, "schema_version": "999"})

    def test_missing_name(self):
        with pytest.raises(ModelSchemaError, match="Missing required field: name"):
            model_from_dict({"schema_version": "0.1", "variables": [], "factors": []})

    def test_name_not_string(self):
        with pytest.raises(ModelSchemaError, match="'name' must be a string"):
            model_from_dict({**self.base, "name": 42})

    def test_missing_variables(self):
        with pytest.raises(ModelSchemaError, match="Missing required field: variables"):
            model_from_dict({"schema_version": "0.1", "name": "x", "factors": []})

    def test_missing_factors(self):
        with pytest.raises(ModelSchemaError, match="Missing required field: factors"):
            model_from_dict({"schema_version": "0.1", "name": "x", "variables": []})

    def test_variable_missing_name(self):
        with pytest.raises(ModelSchemaError, match=r"variables\[0\]: missing 'name'"):
            model_from_dict({**self.base, "variables": [{"domain": ["a"]}]})

    def test_variable_missing_domain(self):
        with pytest.raises(ModelSchemaError, match=r"variables\[0\]: missing 'domain'"):
            model_from_dict({**self.base, "variables": [{"name": "v"}]})

    def test_factor_missing_inputs(self):
        with pytest.raises(ModelSchemaError, match=r"factors\[0\]: missing 'inputs'"):
            model_from_dict({**self.base, "factors": [{"output": "v", "operator": "binary"}]})

    def test_factor_missing_output(self):
        with pytest.raises(ModelSchemaError, match=r"factors\[0\]: missing 'output'"):
            model_from_dict({**self.base, "factors": [{"inputs": ["v"], "operator": "binary"}]})

    def test_factor_missing_operator(self):
        with pytest.raises(ModelSchemaError, match=r"factors\[0\]: missing 'operator'"):
            model_from_dict({**self.base, "factors": [{"inputs": ["v"], "output": "v"}]})


class TestJsonRoundTrip:
    def test_weather_model(self):
        m = make_weather_model()
        r1 = m.query("rain", evidence={"cloudy": "yes"})
        m2 = model_from_json(model_to_json(m))
        r2 = m2.query("rain", evidence={"cloudy": "yes"})
        assert r1 == r2


class TestFileRoundTrip:
    def test_weather_model(self):
        m = make_weather_model()
        r1 = m.query("rain", evidence={"cloudy": "yes"})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp = f.name
        try:
            export_model(m, tmp)
            m2 = import_model(tmp)
            r2 = m2.query("rain", evidence={"cloudy": "yes"})
            assert r1 == r2
        finally:
            os.unlink(tmp)


class TestOperatorRoundTrip:
    def test_table(self):
        m = make_weather_model()
        r1 = m.query("rain", evidence={"cloudy": "yes"})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("rain", evidence={"cloudy": "yes"})
        assert r1 == r2

    def test_sigmoid(self):
        m = Model("test")
        m.add_variable(Variable(name="x", domain=("continuous", 0, 1), value=0.7))
        m.add_variable(Variable(name="y", domain=("continuous", 0, 1)))
        m.add_factor(Factor(inputs=["x"], output="y",
                          weight_function=sigmoid(w=np.array([5.0]), sigma=0.1)))
        r1 = m.query("y", evidence={"x": 0.7})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("y", evidence={"x": 0.7})
        assert r1 == r2

    def test_binary(self):
        m = Model("test")
        m.add_variable(Variable("a", domain=["x", "y"]))
        m.add_variable(Variable("b", domain=["x", "y"]))
        m.add_variable(Variable("match", domain=["x", "y"]))
        m.add_factor(Factor(inputs=["a", "b"], output="match", weight_function=binary()))
        r1 = m.query("match", evidence={"a": "x", "b": "x"})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("match", evidence={"a": "x", "b": "x"})
        assert r1 == r2

    def test_gaussian(self):
        m = Model("test")
        m.add_variable(Variable("x", domain=("continuous", 0, 1)))
        m.add_variable(Variable("score", domain=("continuous", 0, 1)))
        m.add_factor(Factor(inputs=["x"], output="score",
                          weight_function=gaussian(mu=0.5, sigma=0.1)))
        r1 = m.query("score", evidence={"x": 0.5})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("score", evidence={"x": 0.5})
        assert r1 == r2

    def test_threshold(self):
        m = Model("test")
        m.add_variable(Variable("x", domain=("continuous", 0, 1)))
        m.add_variable(Variable("alert", domain=["yes", "no"]))
        m.add_factor(Factor(inputs=["x"], output="alert", weight_function=threshold(0.5)))
        r1 = m.query("alert", evidence={"x": 0.8})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("alert", evidence={"x": 0.8})
        assert r1 == r2

    def test_product(self):
        m = Model("test")
        m.add_variable(Variable("x", domain=("continuous", 0, 1)))
        m.add_variable(Variable("y", domain=("continuous", 0, 1)))
        m.add_variable(Variable("z", domain=("continuous", 0, 1)))
        m.add_factor(Factor(inputs=["x", "y"], output="z", weight_function=product()))
        r1 = m.query("z", evidence={"x": 0.3, "y": 0.4})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("z", evidence={"x": 0.3, "y": 0.4})
        assert r1 == r2

    def test_sum(self):
        m = Model("test")
        m.add_variable(Variable("x", domain=("continuous", 0, 1)))
        m.add_variable(Variable("y", domain=("continuous", 0, 1)))
        m.add_variable(Variable("z", domain=("continuous", 0, 2)))
        m.add_factor(Factor(inputs=["x", "y"], output="z", weight_function=sum_w()))
        r1 = m.query("z", evidence={"x": 0.3, "y": 0.4})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("z", evidence={"x": 0.3, "y": 0.4})
        assert r1 == r2

    def test_max(self):
        m = Model("test")
        m.add_variable(Variable("x", domain=("continuous", 0, 1)))
        m.add_variable(Variable("y", domain=("continuous", 0, 1)))
        m.add_variable(Variable("z", domain=("continuous", 0, 1)))
        m.add_factor(Factor(inputs=["x", "y"], output="z", weight_function=max_w()))
        r1 = m.query("z", evidence={"x": 0.3, "y": 0.5})
        m2 = model_from_dict(model_to_dict(m))
        r2 = m2.query("z", evidence={"x": 0.3, "y": 0.5})
        assert r1 == r2
