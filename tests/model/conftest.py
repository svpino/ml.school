from unittest.mock import Mock

import numpy as np
import pytest

from pipelines.inference.model import Model


@pytest.fixture
def model(monkeypatch):
    """Return a model instance."""
    model = Model()

    context = Mock()
    context.artifacts = {
        "model": "model",
        "features_transformer": "features_transformer",
        "target_transformer": "target_transformer",
    }

    mock_features_transformer = Mock()
    mock_features_transformer.transform = Mock()

    mock_species_transformer = Mock()
    mock_species_transformer.categories_ = [["Adelie", "Chinstrap", "Gentoo"]]

    mock_target_transformer = Mock()
    mock_target_transformer.named_transformers_ = {"species": mock_species_transformer}

    def mock_load(path):
        return (
            mock_features_transformer
            if path == "features_transformer"
            else mock_target_transformer
        )

    monkeypatch.setattr("joblib.load", mock_load)

    mock_model = Mock()
    mock_model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr("keras.saving.load_model", lambda _: mock_model)

    model.load_context(context)

    return model
