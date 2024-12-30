import json
import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pipelines.inference.model import Model


@pytest.fixture
def mock_keras_model(monkeypatch):
    """Return a mock Keras model."""
    mock_model = Mock()
    mock_model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr("keras.saving.load_model", lambda _: mock_model)

    return mock_model


@pytest.fixture
def mock_transformers(monkeypatch):
    """Return mock transformer instances."""
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
    return mock_features_transformer, mock_target_transformer


@pytest.fixture
def model(mock_keras_model, mock_transformers):
    """Return a model instance."""
    model = Model()

    mock_context = Mock()
    mock_context.artifacts = {
        "model": "model",
        "features_transformer": "features_transformer",
        "target_transformer": "target_transformer",
    }

    model.load_context(mock_context)

    assert model.model == mock_keras_model
    assert model.features_transformer == mock_transformers[0]
    assert model.target_transformer == mock_transformers[1]

    return model


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file."""
    config_file = tmp_path / "config.json"
    config_data = {"database": "penguins-test.db"}
    config_file.write_text(json.dumps(config_data))
    return config_file


def test_model_initializes_with_no_backend():
    model = Model()
    assert model.backend is None


def test_process_input(model):
    model.features_transformer.transform = Mock(
        return_value=np.array([[0.1, 0.2]]),
    )
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    # Ensure the transform method is called with the input data.
    model.features_transformer.transform.assert_called_once_with(input_data)

    # The function should return the transformed data.
    assert np.array_equal(result, np.array([[0.1, 0.2]]))


def test_process_input_return_none_on_exception(model):
    model.features_transformer.transform = Mock(side_effect=Exception("Invalid input"))
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    # We want to make sure that the transform method is called with the input data.
    model.features_transformer.transform.assert_called_once_with(input_data)

    # Since there was an exception, the function should return None.
    assert result is None


def test_process_output(model):
    output = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    result = model.process_output(output)

    assert result == [
        {"prediction": "Adelie", "confidence": 0.6},
        {"prediction": "Chinstrap", "confidence": 0.7},
    ]


def test_process_output_return_empty_list_on_none(model):
    assert model.process_output(None) == []


def test_predict_returns_empty_list_if_input_is_empty(model):
    assert model.predict(None, pd.DataFrame()) == []


def test_predict_returns_empty_list_if_input_is_none(model):
    assert model.predict(None, None) == []


def test_predict_return_empty_list_on_invalid_input(model, monkeypatch):
    mock_process_input = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict_returns_empty_list_on_invalid_prediction(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    model.model.predict = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


@pytest.mark.parametrize(
    ("model_input", "samples"),
    [
        (pd.DataFrame([{"island": "Torgersen", "culmen_length_mm": 39.1}]), 1),
        ([{"island": "Torgersen", "culmen_length_mm": 39.1}, {"island": "Biscoe"}], 2),
        ({"island": "Torgersen", "culmen_length_mm": 39.1}, 1),
    ],
)
def test_predict_supported_input_format(model, model_input, samples):
    model.process_input = Mock()
    model.predict(None, model_input)

    model.process_input.assert_called_once()
    assert len(model.process_input.call_args[0][0]) == samples


def test_predict(model):
    model_input = pd.DataFrame([{"island": "Torgersen", "culmen_length_mm": 39.1}])
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    mock_process_output = Mock(
        return_value=[{"prediction": "Adelie", "confidence": 0.6}],
    )
    model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    model.process_input = mock_process_input
    model.process_output = mock_process_output

    result = model.predict(None, model_input)

    assert result == [{"prediction": "Adelie", "confidence": 0.6}]
    mock_process_input.assert_called_once()
    mock_process_output.assert_called_once()
    model.model.predict.assert_called_once()


def test_predict_calls_backend_if_backend_exists(model):
    model.backend = Mock()
    model.predict(None, pd.DataFrame([{"island": "Torgersen"}]))
    model.backend.save.assert_called_once()


def test_predict_backend_receives_prediction(model):
    model.backend = Mock()
    model_input = pd.DataFrame([{"island": "Torgersen"}])
    model.predict(context=None, model_input=model_input)

    assert model.backend.save.call_args[0][1] == [
        {"prediction": "Adelie", "confidence": 0.6},
    ]


def test_predict_backend_receives_none_if_prediction_is_none(model):
    model.backend = Mock()
    model.process_output = Mock(return_value=None)
    model.predict(context=None, model_input=[{"island": "Torgersen"}])

    assert model.backend.save.call_args[0][1] is None


def test_backend_is_loaded_from_environment_variable():
    with (
        patch.dict(os.environ, {"MODEL_BACKEND": "module.Backend"}),
        patch("importlib.import_module") as mock_import,
    ):
        mock_module = Mock()
        mock_module.Backend = Mock
        mock_import.return_value = mock_module

        model = Model()
        model.load_context(context=None)

        assert isinstance(model.backend, Mock)
        mock_import.assert_called_once_with("module")


def test_backend_is_initialized_with_config_file(temp_config):
    with (
        patch.dict(
            os.environ,
            {
                "MODEL_BACKEND": "module.Backend",
                "MODEL_BACKEND_CONFIG": str(temp_config),
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        mock_backend = Mock()
        mock_backend_class = Mock(return_value=mock_backend)

        mock_module = Mock()
        mock_module.Backend = mock_backend_class
        mock_import.return_value = mock_module

        model = Model()
        model.load_context(context=None)

        assert isinstance(model.backend, Mock)
        mock_backend_class.assert_called_once_with(
            config={"database": "penguins-test.db"},
        )


def test_backend_is_initialized_with_invalid_config():
    with (
        patch.dict(
            os.environ,
            {
                "MODEL_BACKEND": "module.Backend",
                "MODEL_BACKEND_CONFIG": "invalid.json",
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        mock_backend = Mock()
        mock_backend_class = Mock(return_value=mock_backend)

        mock_module = Mock()
        mock_module.Backend = mock_backend_class
        mock_import.return_value = mock_module

        model = Model()
        model.load_context(context=None)

        assert isinstance(model.backend, Mock)
        mock_backend_class.assert_called_once_with(config=None)


def test_backend_is_set_to_none_if_class_doesnt_exist():
    with patch.dict(os.environ, {"MODEL_BACKEND": "invalid"}):
        model = Model()
        model.load_context(context=None)
        assert model.backend is None
