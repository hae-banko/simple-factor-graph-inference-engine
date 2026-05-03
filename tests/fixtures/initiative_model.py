"""Reusable test models for the Bayesian engine."""

import numpy as np

from bayesian_engine import Factor, Model, Variable
from bayesian_engine.operators import sigmoid, table


def make_weather_model() -> Model:
    """Simple two-variable weather model: P(rain | cloudy)."""
    m = Model("weather")
    m.add_variable(Variable(name="cloudy", domain=["yes", "no"], value="yes"))
    m.add_variable(Variable(name="rain", domain=["yes", "no"]))
    m.add_factor(
        Factor(
            inputs=["cloudy"],
            output="rain",
            weight_function=table(
                {
                    ("yes", "yes"): 0.8,
                    ("yes", "no"): 0.2,
                    ("no", "yes"): 0.1,
                    ("no", "no"): 0.9,
                }
            ),
        )
    )
    return m


def make_chain_model() -> Model:
    """Three-variable chain: A → B → C."""
    m = Model("chain")
    m.add_variable(Variable(name="a", domain=["on", "off"], value="on"))
    m.add_variable(Variable(name="b", domain=["high", "low"]))
    m.add_variable(Variable(name="c", domain=["yes", "no"]))

    m.add_factor(
        Factor(
            inputs=["a"],
            output="b",
            weight_function=table(
                {
                    ("on", "high"): 0.9,
                    ("on", "low"): 0.1,
                    ("off", "high"): 0.2,
                    ("off", "low"): 0.8,
                }
            ),
        )
    )
    m.add_factor(
        Factor(
            inputs=["b"],
            output="c",
            weight_function=table(
                {
                    ("high", "yes"): 0.7,
                    ("high", "no"): 0.3,
                    ("low", "yes"): 0.4,
                    ("low", "no"): 0.6,
                }
            ),
        )
    )
    return m


def make_continuous_model() -> Model:
    """Model with a continuous output variable."""
    m = Model("continuous")
    m.add_variable(Variable(name="x", domain=("continuous", 0, 1), value=0.7))
    m.add_variable(Variable(name="y", domain=("continuous", 0, 1)))
    m.add_factor(
        Factor(
            inputs=["x"],
            output="y",
            weight_function=sigmoid(w=np.array([5.0]), sigma=0.1),
        )
    )
    return m
