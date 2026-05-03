"""JSON schema definition and validation for model serialization."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "0.1"


class ModelSchemaError(ValueError):
    """Raised when a serialized model fails schema validation."""


def validate_model_schema(data: dict[str, Any]) -> None:
    """Validate a serialized model dict against the expected schema.

    Raises ModelSchemaError if validation fails.
    """
    if not isinstance(data, dict):
        raise ModelSchemaError("Model data must be a dict")

    if "schema_version" not in data:
        raise ModelSchemaError("Missing required field: schema_version")
    version = data["schema_version"]
    if version != SCHEMA_VERSION:
        raise ModelSchemaError(
            f"Unsupported schema version {version!r}, expected {SCHEMA_VERSION!r}"
        )

    if "name" not in data:
        raise ModelSchemaError("Missing required field: name")
    if not isinstance(data["name"], str):
        raise ModelSchemaError("'name' must be a string")

    variables = data.get("variables")
    if not isinstance(variables, list):
        raise ModelSchemaError("Missing required field: variables")

    for i, var in enumerate(variables):
        _validate_variable(var, i)

    factors = data.get("factors")
    if not isinstance(factors, list):
        raise ModelSchemaError("Missing required field: factors")

    for i, fac in enumerate(factors):
        _validate_factor(fac, i)


def _validate_variable(var: dict[str, Any], idx: int) -> None:
    prefix = f"variables[{idx}]"
    if "name" not in var:
        raise ModelSchemaError(f"{prefix}: missing 'name'")
    if not isinstance(var["name"], str):
        raise ModelSchemaError(f"{prefix}: 'name' must be a string")
    if "domain" not in var:
        raise ModelSchemaError(f"{prefix}: missing 'domain'")
    domain = var["domain"]
    if not isinstance(domain, (list, dict)):
        raise ModelSchemaError(f"{prefix}: 'domain' must be a list or dict")


def _validate_factor(fac: dict[str, Any], idx: int) -> None:
    prefix = f"factors[{idx}]"
    if "inputs" not in fac:
        raise ModelSchemaError(f"{prefix}: missing 'inputs'")
    if not isinstance(fac["inputs"], list):
        raise ModelSchemaError(f"{prefix}: 'inputs' must be a list")
    if "output" not in fac:
        raise ModelSchemaError(f"{prefix}: missing 'output'")
    if not isinstance(fac["output"], str):
        raise ModelSchemaError(f"{prefix}: 'output' must be a string")
    if "operator" not in fac:
        raise ModelSchemaError(f"{prefix}: missing 'operator'")
    if not isinstance(fac["operator"], str):
        raise ModelSchemaError(f"{prefix}: 'operator' must be a string")
