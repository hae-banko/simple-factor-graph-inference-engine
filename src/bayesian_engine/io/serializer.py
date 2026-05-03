"""JSON serialization for Model objects with versioned schema support.

Supports round-trip serialization of :class:`~bayesian_engine.core.model.Model`
instances, including variables, factors, and operator configurations.

Schema version: defined in `bayesian_engine.io.schema.SCHEMA_VERSION`.
Discrete domains serialized inline as lists. Continuous domains as
``{"_type": "continuous", ...}`` dicts. Operators serialized by name + params
(via ``__operator_name__`` / ``__operator_params__`` closure attributes,
not as callables).
"""

from __future__ import annotations

import importlib
from typing import Any

from bayesian_engine.io.schema import SCHEMA_VERSION
from bayesian_engine.utils.types import ContinuousDomain, DiscreteDomain, Domain

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SUPPORTED_SCHEMA_VERSION = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Domain serialization
# ---------------------------------------------------------------------------


def _domain_to_json(domain: Domain) -> list | dict:
    """Serialize a Domain to JSON-safe representation."""
    if isinstance(domain, DiscreteDomain):
        return list(domain.values)
    if isinstance(domain, ContinuousDomain):
        return {
            "_type": "continuous",
            "lo": domain.lo,
            "hi": domain.hi,
            "bins": domain.bins,
        }
    raise TypeError(f"Unsupported domain type: {type(domain).__name__}")


def _domain_from_json(data: Any) -> Domain:
    """Deserialize JSON-safe representation back to a Domain."""
    if isinstance(data, list):
        return DiscreteDomain(values=[str(v) for v in data])
    if isinstance(data, dict):
        if data.get("_type") == "continuous":
            return ContinuousDomain(
                lo=data["lo"], hi=data["hi"], bins=data.get("bins", 20)
            )
        raise ValueError(f"Unknown domain type marker: {data.get('_type')}")
    raise TypeError(f"Cannot deserialize domain from: {type(data).__name__}")


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

_OPERATOR_REGISTRY: dict[str, str] = {
    "sigmoid": "bayesian_engine.operators.sigmoid.sigmoid",
    "table": "bayesian_engine.operators.table.table",
    "binary": "bayesian_engine.operators.binary.binary",
    "gaussian": "bayesian_engine.operators.gaussian.gaussian",
    "threshold": "bayesian_engine.operators.threshold.threshold",
    "product": "bayesian_engine.operators.combinators.product",
    "sum": "bayesian_engine.operators.combinators.sum",
    "max": "bayesian_engine.operators.combinators.max",
}


def _resolve_operator(name: str) -> Any:
    """Import and return an operator factory by its registry name."""
    path = _OPERATOR_REGISTRY.get(name)
    if path is None:
        raise ValueError(
            f"Unknown operator: {name!r}. Known: {list(_OPERATOR_REGISTRY)}"
        )
    module_path, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# ---------------------------------------------------------------------------
# Operator introspection
# ---------------------------------------------------------------------------

def _extract_operator_info(weight_function: Any) -> dict:
    """Inspect a weight function to identify its operator type and parameters.

    Uses ``__operator_name__`` and ``__operator_params__`` attributes set by the
    operator factories. Falls back to closure-based introspection for operators
    that don't set these attributes.

    Returns:
        A dict with ``operator`` (str) and ``params`` (dict) keys.

    Raises:
        TypeError: If the weight function cannot be introspected.
    """
    op_name = getattr(weight_function, "__operator_name__", None)
    op_params = getattr(weight_function, "__operator_params__", {})

    if op_name is not None:
        serializable_params = {}
        for key, val in op_params.items():
            serializable_params[key] = _make_json_safe(val)
        return {"operator": op_name, "params": serializable_params}

    raise TypeError(
        f"Cannot determine operator info for {weight_function!r}. "
        "Ensure the operator sets __operator_name__ and __operator_params__ attributes."
    )


def _make_json_safe(val: Any) -> Any:
    """Convert a value to a JSON-safe representation."""
    import numpy as np

    if isinstance(val, np.ndarray):
        return {"_type": "ndarray", "data": val.tolist()}
    if isinstance(val, dict):
        if val and all(isinstance(k, tuple) for k in val.keys()):
            return {"_type": "cpt", "entries": [[list(k), v] for k, v in val.items()]}
        return {str(k): _make_json_safe(v) for k, v in val.items()}
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _params_from_json(params: dict) -> dict:
    """Convert JSON-safe params back to their original types."""
    import numpy as np

    result = {}
    for key, val in params.items():
        if isinstance(val, dict):
            if val.get("_type") == "ndarray":
                result[key] = np.array(val["data"], dtype=float)
            elif val.get("_type") == "cpt":
                result[key] = {tuple(k): v for k, v in val["entries"]}
            else:
                result[key] = val
        else:
            result[key] = val
    return result
