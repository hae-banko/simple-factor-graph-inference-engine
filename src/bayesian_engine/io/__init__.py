"""I/O module — JSON model serialization and deserialization.

Public API:
    - :func:`export_model` / :func:`import_model` — file-based round-trip.
    - :func:`model_to_json` / :func:`model_from_json` — string-based round-trip.
    - :func:`model_to_dict` / :func:`model_from_dict` — dict-based round-trip.
"""

from bayesian_engine.io.export import export_model, model_to_dict, model_to_json
from bayesian_engine.io.import_ import import_model, model_from_dict, model_from_json
from bayesian_engine.io.schema import SCHEMA_VERSION, ModelSchemaError, validate_model_schema

__all__ = [
    "ModelSchemaError",
    "SCHEMA_VERSION",
    "export_model",
    "import_model",
    "model_to_dict",
    "model_from_dict",
    "model_to_json",
    "model_from_json",
    "validate_model_schema",
]
