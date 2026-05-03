"""Model export functions — serializing Model instances to JSON."""

from __future__ import annotations

import json
from typing import Any

from bayesian_engine.core.model import Model
from bayesian_engine.io.serializer import (
    SUPPORTED_SCHEMA_VERSION,
    _domain_to_json,
    _extract_operator_info,
)


def model_to_dict(model: Model) -> dict:
    """Serialize a :class:`Model` to a JSON-compatible dictionary.

    Args:
        model: The model to export.

    Returns:
        A dictionary with ``schema_version``, ``name``, ``variables``,
        ``factors``, and ``metadata`` keys.

    Raises:
        TypeError: If a factor uses an unsupported operator type.
    """
    variables_json = []
    for var in model.variables.values():
        entry: dict[str, Any] = {
            "name": var.name,
            "domain": _domain_to_json(var.domain),
            "latent": var.latent,
        }
        if var.value is not None:
            entry["value"] = var.value
        variables_json.append(entry)

    factors_json = []
    for factor in model.factors:
        op_info = _extract_operator_info(factor.weight_function)
        factors_json.append(
            {
                "inputs": list(factor.inputs),
                "output": factor.output,
                "operator": op_info["operator"],
                "params": op_info["params"],
            }
        )

    result = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "name": model.name,
        "variables": variables_json,
        "factors": factors_json,
    }
    if model.metadata:
        result["metadata"] = dict(model.metadata)
    return result


def model_to_json(model: Model, indent: int = 2) -> str:
    """Serialize a :class:`Model` to a JSON string.

    Args:
        model: The model to export.
        indent: Pretty-print indentation level (default: 2).

    Returns:
        A JSON string representation of the model.
    """
    return json.dumps(model_to_dict(model), indent=indent)


def export_model(model: Model, path: str) -> None:
    """Export a :class:`Model` to a JSON file.

    Args:
        model: The model to export.
        path: Filesystem path for the output JSON file.
    """
    with open(path, "w") as f:
        f.write(model_to_json(model))
