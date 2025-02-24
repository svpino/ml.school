import os
from unittest.mock import Mock

import pytest

from pipelines.inference.model import Model


@pytest.fixture
def context():
    """Return a mock context."""
    mock_context = Mock()
    mock_context.artifacts = {
        "model": "model",
        "features_transformer": "features_transformer",
        "target_transformer": "target_transformer",
    }

    return mock_context


@pytest.fixture
def model(monkeypatch):
    """Return a model instance."""
    model = Model()

    def mock_load(path):
        return Mock(artifact=path)

    monkeypatch.setattr("joblib.load", mock_load)
    monkeypatch.setattr("keras.saving.load_model", lambda _: Mock(artifact="model"))

    return model


def test_load_artifacts_loads_keras_model(model, context):
    model.load_context(context)
    assert model.model.artifact == "model"


def test_load_artifacts_loads_features_transformers(model, context):
    model.load_context(context)
    artifact = model.features_transformer.artifact
    assert artifact == context.artifacts["features_transformer"]


def test_load_artifacts_loads_target_transformer(model, context):
    model.load_context(context)
    artifact = model.target_transformer.artifact
    assert artifact == context.artifacts["target_transformer"]


def test_keras_backend_is_set_to_tensorflow_by_default(model, context, monkeypatch):
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    model.load_context(context)
    assert os.getenv("KERAS_BACKEND") == "tensorflow"


def test_keras_backend_is_unchanged_if_present(model, context, monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "jax")
    model.load_context(context)
    assert os.getenv("KERAS_BACKEND") == "jax"
