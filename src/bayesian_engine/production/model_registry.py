"""Multi-model lifecycle manager for bayesian-engine.

Provides a registry to register, list, load, and delete Bayesian models
using a JSONL manifest file for persistence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bayesian_engine.io.schema import validate_model_schema
from bayesian_engine.policy.wrapper import BayesianPolicy


class ModelRegistry:
    """Registry for managing multiple Bayesian model lifecycle.

    Models are registered with a name, version, and path to a JSON model file.
    The registry maintains a JSONL manifest at ``<state_dir>/models/manifest.jsonl``.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        """Initialize the registry.

        Args:
            state_dir: Optional custom state directory. Defaults to
                ``~/.hermes/bayesian_scheduler``.
        """
        self._state_dir = state_dir

    def get_state_dir(self) -> Path:
        """Return the state directory, creating it if missing.

        Returns:
            Path to the state directory.
        """
        if self._state_dir is None:
            state_dir = Path.home() / ".hermes" / "bayesian_scheduler"
        else:
            state_dir = self._state_dir

        models_dir = state_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = models_dir / "manifest.jsonl"
        manifest_path.touch(exist_ok=True)
        return state_dir

    def get_manifest_path(self) -> Path:
        """Return the path to the manifest JSONL file.

        Returns:
            Path to ``<state_dir>/models/manifest.jsonl``.
        """
        return self.get_state_dir() / "models" / "manifest.jsonl"

    def list_models(self) -> list[dict[str, Any]]:
        """Scan the manifest and return all model entries.

        Returns:
            List of model entry dicts from the manifest.
        """
        manifest_path = self.get_manifest_path()
        if not manifest_path.exists():
            return []

        models = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                models.append(json.loads(line))
        return models

    def load_model(self, name: str) -> BayesianPolicy:
        """Load a model by name from the registry.

        Args:
            name: Name of the model to load.

        Returns:
            A :class:`BayesianPolicy` instance wrapping the loaded model.

        Raises:
            FileNotFoundError: If the model is not found in the registry
                or its file does not exist.
        """
        models = self.list_models()
        for entry in models:
            if entry["name"] == name:
                model_path = self.get_state_dir() / entry["path"]
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                return BayesianPolicy.load(str(model_path))

        raise FileNotFoundError(f"Model '{name}' not found in registry")

    def _resolve_path(self, name: str) -> Path:
        """Return the absolute path for a registered model.

        Manifest paths are stored relative to ``state_dir``; this resolves them
        to absolute paths.

        Raises:
            FileNotFoundError: If the model is not registered or its file is missing.
        """
        entries = self.list_models()
        for entry in entries:
            if entry["name"] == name:
                # Path stored in manifest is relative (e.g. "models/copresence.json")
                relative = Path(entry["path"])
                if relative.is_absolute():
                    return relative
                return self.get_state_dir() / relative
        raise FileNotFoundError(f"Model '{name}' not found in registry")

    def register_model(
        self, model_path: Path | str, name: str, version: str = "0.1.0"
    ) -> dict[str, Any]:
        """Register a new model in the registry.

        Validates the model JSON file against the schema before registering.

        Args:
            model_path: Path to the JSON model file.
            name: Unique name for the model.
            version: Version string for the model. Defaults to "0.1.0".

        Returns:
            The manifest entry dict that was written.

        Raises:
            ValueError: If a model with the given name already exists,
                or if the model file fails schema validation.
        """
        manifest_path = self.get_manifest_path()

        # Check for duplicate name
        existing = self.list_models()
        for entry in existing:
            if entry["name"] == name:
                raise ValueError(f"Model '{name}' already registered")

        # Validate the model file
        model_path = Path(model_path)
        with open(model_path) as f:
            model_dict = json.load(f)
        validate_model_schema(model_dict)

        # Write to manifest (append mode)
        entry = {
            "name": name,
            "path": f"models/{model_path.name}",
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "inference_count": 0,
        }

        with open(manifest_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def delete_model(self, name: str) -> bool:
        """Remove a model from the registry by name.

        Does NOT delete the model file itself.

        Args:
            name: Name of the model to delete.

        Returns:
            True if the model was found and deleted, False otherwise.
        """
        manifest_path = self.get_manifest_path()
        if not manifest_path.exists():
            return False

        # Read all entries
        remaining = []
        found = False
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry["name"] == name:
                    found = True
                else:
                    remaining.append(entry)

        if not found:
            return False

        # Rewrite without the deleted entry
        with open(manifest_path, "w") as f:
            for entry in remaining:
                f.write(json.dumps(entry) + "\n")

        return True

    def increment_inference_count(self, name: str) -> int:
        """Increment the inference count for a named model.

        Args:
            name: Name of the model.

        Returns:
            The new inference count after incrementing.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        manifest_path = self.get_manifest_path()
        if not manifest_path.exists():
            raise KeyError(f"Model '{name}' not found in registry")

        # Read all entries
        entries = []
        found = False
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry["name"] == name:
                    entry["inference_count"] += 1
                    new_count = entry["inference_count"]
                    found = True
                entries.append(entry)

        if not found:
            raise KeyError(f"Model '{name}' not found in registry")

        # Rewrite all entries
        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return new_count
