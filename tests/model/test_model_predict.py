from unittest.mock import Mock

import numpy as np
import pandas as pd


def test_predict_returns_empty_list_if_input_is_empty(model):
    assert model.predict(None, model_input=[]) == []


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
    model_input = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
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


def test_predict_backend_is_called(model):
    model.backend = Mock()
    model.predict(None, [{"island": "Torgersen"}])
    model.backend.save.assert_called_once()


def test_predict_backend_receives_model_input(model):
    model.backend = Mock()
    model_input = [{"island": "Torgersen"}, {"island": "Biscoe"}]
    model.predict(context=None, model_input=model_input)

    backend_input_arg = model.backend.save.call_args[0][0]
    assert backend_input_arg.island.iloc[0] == "Torgersen"
    assert backend_input_arg.island.iloc[1] == "Biscoe"


def test_predict_backend_receives_prediction(model):
    model.backend = Mock()
    model_input = [{"island": "Torgersen"}]
    model.predict(context=None, model_input=model_input)

    backend_output_arg = model.backend.save.call_args[0][1]
    assert backend_output_arg == [
        {"prediction": "Adelie", "confidence": 0.6},
    ]


def test_predict_backend_receives_prediction_none(model):
    model.backend = Mock()
    model.process_output = Mock(return_value=None)
    model.predict(context=None, model_input=[{"island": "Torgersen"}])

    backend_output_arg = model.backend.save.call_args[0][1]
    assert backend_output_arg is None


def test_process_input_should_transform_input_data(model):
    model.features_transformer.transform = Mock(
        return_value=np.array([[0.1, 0.2]]),
    )
    model_input = [{"island": ["Torgersen"]}]
    result = model.process_input(model_input)

    model.features_transformer.transform.assert_called_once_with(model_input)
    assert np.array_equal(result, np.array([[0.1, 0.2]]))


def test_process_input_returns_none_on_exception(model):
    model.features_transformer.transform = Mock(side_effect=Exception("Invalid input"))
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    model.features_transformer.transform.assert_called_once_with(input_data)

    assert result is None, (
        "Since there was an exception, the function should return None."
    )


def test_process_output_returns_json(model):
    output = np.array([[0.6, 0.3, 0.1]])
    result = model.process_output(output)
    assert isinstance(result[0], dict)


def test_process_output_returns_prediction_and_confidence(model):
    output = np.array([[0.6, 0.3, 0.1]])
    result = model.process_output(output)

    assert result[0].keys() == {"prediction", "confidence"}


def test_process_output_returns_species(model):
    output = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    result = model.process_output(output)

    assert result[0]["prediction"] == "Adelie"
    assert result[1]["prediction"] == "Chinstrap"
    assert result[2]["prediction"] == "Gentoo"


def test_process_output_returns_empty_list_if_it_receives_none(model):
    assert model.process_output(None) == []
