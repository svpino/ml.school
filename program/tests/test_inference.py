import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from metaflow.inference import Model


@pytest.fixture()
def mock_keras_model(monkeypatch):
    """Return a mock Keras model."""
    mock_model = Mock()
    mock_model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr("keras.saving.load_model", lambda _: mock_model)

    return mock_model


@pytest.fixture()
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


@pytest.fixture()
def model(mock_keras_model, mock_transformers):
    """Return a model instance."""
    directory = tempfile.mkdtemp()
    data_capture_file = Path(directory) / "database.db"

    model = Model(data_capture_file=data_capture_file, data_capture=False)

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


def fetch_data(model):
    connection = sqlite3.connect(model.data_capture_file)
    cursor = connection.cursor()
    cursor.execute("SELECT island, prediction, confidence FROM data;")
    data = cursor.fetchone()
    connection.close()
    return data


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


def test_predict_return_empty_list_on_invalid_input(model, monkeypatch):
    mock_process_input = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict_return_empty_list_on_invalid_prediction(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    model.model.predict = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    mock_process_output = Mock(
        return_value=[{"prediction": "Adelie", "confidence": 0.6}],
    )
    model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr(model, "process_input", mock_process_input)
    monkeypatch.setattr(model, "process_output", mock_process_output)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)

    assert result == [{"prediction": "Adelie", "confidence": 0.6}]
    mock_process_input.assert_called_once()
    mock_process_output.assert_called_once()
    model.model.predict.assert_called_once()


@pytest.mark.parametrize(
    ("default_data_capture", "request_data_capture", "database_exists"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_data_capture(
    model,
    default_data_capture,
    request_data_capture,
    database_exists,
):
    model.data_capture = default_data_capture
    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": request_data_capture},
    )

    assert Path(model.data_capture_file).exists() == database_exists


def test_capture_uses_environment_variable_to_specify_database_file(model):
    # Let's set the database file to a custom path using the environment
    # variable.
    data_capture_file = Path(tempfile.mkdtemp()) / "test.db"
    os.environ["MODEL_DATA_CAPTURE_FILE"] = data_capture_file.as_posix()

    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": True},
    )

    assert Path(data_capture_file).exists()
    Path(data_capture_file).unlink()

    # Remove the environment variable so it doesn't affect any of the other tests
    # in this suite.
    del os.environ["MODEL_DATA_CAPTURE_FILE"]


def test_capture_stores_data_in_database(model):
    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": True},
    )

    data = fetch_data(model)
    assert data == ("Torgersen", "Adelie", 0.6)


def test_capture_on_invalid_output(model, monkeypatch):
    mock_process_output = Mock(return_value=None)
    monkeypatch.setattr(model, "process_output", mock_process_output)

    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": True},
    )

    data = fetch_data(model)

    # The prediction and confidence columns should be None because the output
    # from the model was empty
    assert data == ("Torgersen", None, None)
