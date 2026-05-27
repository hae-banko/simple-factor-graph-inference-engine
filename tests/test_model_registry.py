"""Tests for the ModelRegistry class."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bayesian_engine.production.model_registry import ModelRegistry


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    """Create a ModelRegistry with a temporary state directory."""
    return ModelRegistry(state_dir=tmp_path)


@pytest.fixture
def model_file(registry: ModelRegistry, tmp_path: Path) -> Path:
    """Copy the copresence.json model into the registry's models dir and return its path."""
    src = Path("/mnt/h/fun/bayesian-engine/models/copresence.json")
    models_dir = registry.get_state_dir() / "models"
    dest = models_dir / "copresence.json"
    shutil.copy(src, dest)
    return dest


class TestInit:
    def test_init_creates_state_dir(self, tmp_path: Path) -> None:
        """ModelRegistry() creates the state dir and manifest.jsonl."""
        reg = ModelRegistry(state_dir=tmp_path)
        state_dir = reg.get_state_dir()
        manifest = reg.get_manifest_path()

        assert state_dir.exists()
        assert state_dir == tmp_path
        assert manifest.exists()

    def test_manifest_path_resolves_correctly(self, tmp_path: Path) -> None:
        """Manifest path resolves to <state_dir>/models/manifest.jsonl."""
        reg = ModelRegistry(state_dir=tmp_path)
        expected = tmp_path / "models" / "manifest.jsonl"
        assert reg.get_manifest_path() == expected


class TestRegisterModel:
    def test_register_model_round_trip(self, registry: ModelRegistry, model_file: Path) -> None:
        """Register a valid model JSON, list_models returns it."""
        entry = registry.register_model(model_file, name="copresence", version="0.1.0")

        assert entry["name"] == "copresence"
        assert entry["version"] == "0.1.0"
        assert entry["inference_count"] == 0
        assert "registered_at" in entry

        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "copresence"

    def test_register_duplicate_name_raises(
        self, registry: ModelRegistry, model_file: Path
    ) -> None:
        """Register same name twice raises ValueError."""
        registry.register_model(model_file, name="copresence")

        with pytest.raises(ValueError, match="already registered"):
            registry.register_model(model_file, name="copresence")

    def test_register_invalid_model_raises(self, registry: ModelRegistry, tmp_path: Path) -> None:
        """Register a model with invalid schema raises the validation error."""
        invalid_model = tmp_path / "invalid.json"
        invalid_model.write_text('{"schema_version": "99.0"}')

        with pytest.raises(Exception):
            registry.register_model(invalid_model, name="bad_model")


class TestListModels:
    def test_list_models_empty(self, registry: ModelRegistry) -> None:
        """Fresh registry returns empty list."""
        assert registry.list_models() == []


class TestDeleteModel:
    def test_delete_model(self, registry: ModelRegistry, tmp_path: Path) -> None:
        """Register 2 models, delete 1, list_models returns only the other."""
        # Create two model files
        model1_path = tmp_path / "model1.json"
        model2_path = tmp_path / "model2.json"

        src = Path("/mnt/h/fun/bayesian-engine/models/copresence.json")
        shutil.copy(src, model1_path)
        shutil.copy(src, model2_path)

        registry.register_model(model1_path, name="model1")
        registry.register_model(model2_path, name="model2")

        assert len(registry.list_models()) == 2

        result = registry.delete_model("model1")
        assert result is True

        remaining = registry.list_models()
        assert len(remaining) == 1
        assert remaining[0]["name"] == "model2"

    def test_delete_nonexistent_returns_false(self, registry: ModelRegistry) -> None:
        """Deleting a nonexistent model returns False."""
        result = registry.delete_model("nonexistent")
        assert result is False


class TestLoadModel:
    def test_load_model(self, registry: ModelRegistry, model_file: Path) -> None:
        """Register copresence.json, load it, get BayesianPolicy object."""
        from bayesian_engine.policy.wrapper import BayesianPolicy

        registry.register_model(model_file, name="copresence")
        policy = registry.load_model("copresence")

        assert isinstance(policy, BayesianPolicy)
        assert policy._model is not None

    def test_load_nonexistent_raises(self, registry: ModelRegistry) -> None:
        """Loading a nonexistent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            registry.load_model("nonexistent")


class TestIncrementInferenceCount:
    def test_increment_inference_count(self, registry: ModelRegistry, model_file: Path) -> None:
        """Increments from 0 to 1 to 2, returns correct count."""
        registry.register_model(model_file, name="copresence")

        assert registry.increment_inference_count("copresence") == 1
        assert registry.increment_inference_count("copresence") == 2

        # Verify persisted
        models = registry.list_models()
        assert models[0]["inference_count"] == 2

    def test_increment_nonexistent_raises_keyerror(self, registry: ModelRegistry) -> None:
        """Incrementing nonexistent model raises KeyError."""
        with pytest.raises(KeyError):
            registry.increment_inference_count("nonexistent")
