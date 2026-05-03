"""Bayesian Engine — general-purpose probabilistic inference over factor graphs.

.. code-block:: python

    from bayesian_engine import Model, Variable, Factor
    from bayesian_engine.operators import sigmoid, threshold, table
    from bayesian_engine.io import export_model, import_model

    # Build a model
    model = Model("weather")
    model.add_variable(Variable("cloudy", domain=["yes", "no"]))
    model.add_variable(Variable("rain", domain=["yes", "no"]))
    model.add_factor(Factor(
        inputs=["cloudy"],
        output="rain",
        weight_function=table({
            ("yes", "yes"): 0.8,
            ("yes", "no"): 0.2,
        }),
    ))

    # Query
    result = model.query("rain", evidence={"cloudy": "yes"})

    # Serialize
    export_model(model, "weather.json")
"""

from bayesian_engine.core.factor import Factor
from bayesian_engine.core.model import Model
from bayesian_engine.core.variable import Variable

__all__ = [
    "Factor",
    "Model",
    "Variable",
]
