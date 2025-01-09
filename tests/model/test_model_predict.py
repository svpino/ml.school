from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("model_input", "samples"),
    [
        (pd.DataFrame([{"island": "Torgersen", "culmen_length_mm": 39.1}]), 1),
        ([{"island": "Torgersen", "culmen_length_mm": 39.1}, {"island": "Biscoe"}], 2),
        ({"island": "Torgersen", "culmen_length_mm": 39.1}, 1),
    ],
)
def test_predict_supported_formats(model, model_input, samples):
    model.process_input = Mock()
    model.predict(None, model_input)

    model.process_input.assert_called_once()
    assert len(model.process_input.call_args[0][0]) == samples


def test_predict_returns_empty_list_if_input_is_empty(model):
    assert model.predict(None, pd.DataFrame()) == []


def test_predict_returns_empty_list_if_input_is_none(model):
    assert model.predict(None, None) == []


def test_predict_return_empty_list_on_invalid_input(model, monkeypatch):
    mock_process_input = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen"}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict_returns_empty_list_on_invalid_prediction(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    model.model.predict = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


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
