"""Tests for Variable and domain types."""

import pytest

from bayesian_engine.core.variable import Variable
from bayesian_engine.utils.types import ContinuousDomain, DiscreteDomain


class TestDiscreteDomain:
    def test_contains(self):
        d = DiscreteDomain(values=["a", "b", "c"])
        assert "a" in d
        assert "z" not in d
        assert len(d) == 3


class TestContinuousDomain:
    def test_properties(self):
        d = ContinuousDomain(lo=0, hi=1, bins=10)
        assert d.step == 0.1
        assert len(d.bin_centers()) == 10

    def test_lo_must_be_less_than_hi(self):
        with pytest.raises(ValueError):
            ContinuousDomain(lo=1, hi=0)

    def test_bins_at_least_2(self):
        with pytest.raises(ValueError):
            ContinuousDomain(lo=0, hi=1, bins=1)

    def test_discretize(self):
        d = ContinuousDomain(lo=0, hi=1, bins=10)
        assert d.discretize(0.05) == 0
        assert d.discretize(0.95) == 9


class TestVariable:
    def test_discrete_variable(self):
        v = Variable(name="x", domain=["a", "b", "c"], value="a")
        assert v.name == "x"
        assert isinstance(v.domain, DiscreteDomain)
        assert v.value == "a"
        assert v.latent is False

    def test_continuous_variable(self):
        v = Variable(name="y", domain=("continuous", 0, 24), value=13.0)
        assert isinstance(v.domain, ContinuousDomain)
        assert v.domain.lo == 0
        assert v.domain.hi == 24

    def test_latent_variable(self):
        v = Variable(name="z", domain=["a", "b"], latent=True)
        assert v.latent is True

    def test_value_not_in_domain_raises(self):
        with pytest.raises(ValueError):
            Variable(name="x", domain=["a", "b"], value="z")

    def test_set_value(self):
        v = Variable(name="x", domain=["a", "b"], value="a")
        v.set_value("b")
        assert v.value == "b"

    def test_set_value_not_in_domain_raises(self):
        v = Variable(name="x", domain=["a", "b"], value="a")
        with pytest.raises(ValueError):
            v.set_value("z")

    def test_default_value_is_none(self):
        v = Variable(name="x", domain=["a", "b"])
        assert v.value is None
