import numpy as np


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
