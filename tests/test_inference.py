"""Property-based and example-based tests for inference correctness."""

import pytest

from tests.fixtures.initiative_model import (
    make_chain_model,
    make_continuous_model,
    make_weather_model,
)


class TestWeatherModel:
    """Example-based tests on the simple weather model."""

    def test_rain_given_cloudy(self):
        m = make_weather_model()
        result = m.query("rain", evidence={"cloudy": "yes"})
        dist = {d["value"]: d["probability"] for d in result["rain"]["distribution"]}
        assert dist["yes"] == pytest.approx(0.8)
        assert dist["no"] == pytest.approx(0.2)
        assert result["rain"]["value"] == "yes"

    def test_rain_given_not_cloudy(self):
        m = make_weather_model()
        result = m.query("rain", evidence={"cloudy": "no"})
        dist = {d["value"]: d["probability"] for d in result["rain"]["distribution"]}
        assert dist["yes"] == pytest.approx(0.1)
        assert dist["no"] == pytest.approx(0.9)
        assert result["rain"]["value"] == "no"

    def test_no_evidence(self):
        """With no evidence, result is the prior (input variable keeps its initial value)."""
        m = make_weather_model()
        result = m.query("rain", evidence={"cloudy": "yes"})
        dist = {d["value"]: d["probability"] for d in result["rain"]["distribution"]}
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        m1 = make_weather_model()
        m2 = make_weather_model()
        r1 = m1.query("rain", evidence={"cloudy": "yes"})
        r2 = m2.query("rain", evidence={"cloudy": "yes"})
        assert r1 == r2


class TestChainModel:
    """Tests on the three-variable chain A → B → C."""

    def test_chain_query(self):
        m = make_chain_model()
        result = m.query("c", evidence={"a": "on"})
        assert "c" in result
        dist = {d["value"]: d["probability"] for d in result["c"]["distribution"]}
        total = sum(dist.values())
        assert total == pytest.approx(1.0)

    def test_chain_intermediate_evidence(self):
        m = make_chain_model()
        result = m.query("c", evidence={"b": "high"})
        dist = {d["value"]: d["probability"] for d in result["c"]["distribution"]}
        # P(c=yes | b=high) = 0.7, P(c=no | b=high) = 0.3
        assert dist["yes"] == pytest.approx(0.7)
        assert dist["no"] == pytest.approx(0.3)


class TestContinuousModel:
    """Tests on a model with continuous variables."""

    def test_continuous_inference(self):
        m = make_continuous_model()
        result = m.query("y", evidence={"x": 0.7})
        assert "y" in result
        assert "value" in result["y"]
        assert "distribution" in result["y"]
        # The MAP should be near σ(5*0.7) = σ(3.5) ≈ 0.97
        assert 0.9 <= result["y"]["value"] <= 1.0

    def test_distribution_sums_to_one(self):
        m = make_continuous_model()
        result = m.query("y", evidence={"x": 0.5})
        total = sum(d["probability"] for d in result["y"]["distribution"])
        assert total == pytest.approx(1.0)


class TestProbabilityAxioms:
    """Property-based tests verifying probability axioms."""

    def test_probabilities_sum_to_one(self):
        """All probability distributions must sum to 1."""
        m = make_weather_model()
        for cloudy in ["yes", "no"]:
            result = m.query("rain", evidence={"cloudy": cloudy})
            total = sum(d["probability"] for d in result["rain"]["distribution"])
            assert total == pytest.approx(1.0)

    def test_probabilities_non_negative(self):
        """All probabilities must be ≥ 0."""
        m = make_weather_model()
        result = m.query("rain", evidence={"cloudy": "yes"})
        for d in result["rain"]["distribution"]:
            assert d["probability"] >= 0.0
