"""Model import functions — deserializing JSON to Model instances."""

from __future__ import annotations

import json

from bayesian_engine.core.factor import Factor
from bayesian_engine.core.model import Model
from bayesian_engine.core.variable import Variable
from bayesian_engine.io.schema import validate_model_schema
from bayesian_engine.io.serializer import (
    _domain_from_json,
    _params_from_json,
    _resolve_operator,
)


def model_from_dict(data: dict) -> Model:
    """Deserialize a dictionary back into a :class:`Model`.

    Args:
        data: A dictionary with ``schema_version``, ``name``, ``variables``,
            and ``factors`` keys — as produced by :func:`model_to_dict`.

    Returns:
        A reconstructed :class:`Model` instance.

    Raises:
        ValueError: If the schema version is unsupported.
        KeyError: If required fields are missing.
    """
    validate_model_schema(data)

    metadata = data.get("metadata", {})
    model = Model(name=data["name"], metadata=metadata if isinstance(metadata, dict) else None)

    for var_entry in data.get("variables", []):
        domain = _domain_from_json(var_entry["domain"])
        variable = Variable(
            name=var_entry["name"],
            domain=domain,
            latent=var_entry.get("latent", False),
        )
        if "value" in var_entry:
            variable.set_value(var_entry["value"])
        model.add_variable(variable)

    for factor_entry in data.get("factors", []):
        operator_factory = _resolve_operator(factor_entry["operator"])
        params = _params_from_json(factor_entry.get("params", {}))
        weight_function = operator_factory(**params) if params else operator_factory()

        factor = Factor(
            inputs=list(factor_entry["inputs"]),
            output=factor_entry["output"],
            weight_function=weight_function,
        )
        model.add_factor(factor)

    return model


def model_from_json(text: str) -> Model:
    """Deserialize a JSON string back into a :class:`Model`.

    Args:
        text: A JSON string — as produced by :func:`model_to_json`.

    Returns:
        A reconstructed :class:`Model` instance.
    """
    data = json.loads(text)
    return model_from_dict(data)


def import_model(path: str) -> Model:
    """Load a :class:`Model` from a JSON file.

    Args:
        path: Filesystem path to a JSON model file.

    Returns:
        The reconstructed :class:`Model` instance.
    """
    with open(path) as f:
        return model_from_json(f.read())
