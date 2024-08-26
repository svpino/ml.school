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
    mock_model.predict = Mock()
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
    database = Path(directory) / "database.db"

    model = Model(database=database)

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


def test_process_input_should_transform_input(model):
    model.features_transformer.transform = Mock(
        return_value=np.array([[0.1, 0.2]]),
    )
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    # We want to make sure that the transform method is called with the input data.
    model.features_transformer.transform.assert_called_once_with(input_data)

    # The function should return the transformed data.
    assert np.array_equal(result, np.array([[0.1, 0.2]]))


def test_process_input_should_return_none_on_exception(model):
    model.features_transformer.transform = Mock(side_effect=Exception("Invalid input"))
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    # We want to make sure that the transform method is called with the input data.
    model.features_transformer.transform.assert_called_once_with(input_data)

    # Since there was an exception, the function should return None.
    assert result is None


def test_process_output_should_transform_output(model):
    output = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    result = model.process_output(output)

    assert result == [
        {"prediction": "Adelie", "confidence": 0.6},
        {"prediction": "Chinstrap", "confidence": 0.7},
    ]


def test_process_output_should_return_empty_list_on_none(model):
    result = model.process_output(None)
    assert result == []


def test_predict_should_return_empty_list_on_invalid_input(model, monkeypatch):
    mock_process_input = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict_should_return_empty_list_on_invalid_prediction(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    mock_process_output = Mock()
    model.model.predict = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)
    monkeypatch.setattr(model, "process_output", mock_process_output)

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


def test_predict_should_not_capture_data(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    mock_process_output = Mock(
        return_value=[{"prediction": "Adelie", "confidence": 0.6}],
    )
    model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr(model, "process_input", mock_process_input)
    monkeypatch.setattr(model, "process_output", mock_process_output)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]

    # We want to make sure that the capture parameter is set to False.
    model.predict(context=None, model_input=input_data, params={"capture": False})

    assert not Path(model.database).exists()


def test_predict_should_capture_data_if_requested(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    mock_process_output = Mock(
        return_value=[{"prediction": "Adelie", "confidence": 0.6}],
    )
    model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr(model, "process_input", mock_process_input)
    monkeypatch.setattr(model, "process_output", mock_process_output)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]

    # We want to make sure that the capture parameter is set to True.
    model.predict(context=None, model_input=input_data, params={"capture": True})

    assert Path(model.database).exists()

    # The database should have the table 'data' in it.
    connection = sqlite3.connect(model.database)
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data';")
    tables = cursor.fetchall()
    assert len(tables) == 1

    # The table should have the information we captured.
    cursor.execute("SELECT island, prediction, confidence FROM data;")
    result = cursor.fetchone()
    assert result == ("Torgersen", "Adelie", 0.6)

    connection.close()


def test_predict_should_capture_data_in_custom_database(model, monkeypatch):
    # Let's set the database to a custom path using the MODEL_DATABASE environment
    # variable.
    database = Path(tempfile.mkdtemp()) / "test.db"
    os.environ["MODEL_DATABASE"] = database.as_posix()

    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    mock_process_output = Mock(
        return_value=[{"prediction": "Adelie", "confidence": 0.6}],
    )
    model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr(model, "process_input", mock_process_input)
    monkeypatch.setattr(model, "process_output", mock_process_output)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]

    # We want to make sure that the capture parameter is set to True.
    model.predict(context=None, model_input=input_data, params={"capture": True})

    assert Path(database).exists()
    Path(database).unlink()
