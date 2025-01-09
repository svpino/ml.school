from unittest.mock import Mock

import numpy as np
import pandas as pd


def test_process_input_should_transform_input_data(model):
    model.features_transformer.transform = Mock(
        return_value=np.array([[0.1, 0.2]]),
    )
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    model.features_transformer.transform.assert_called_once_with(input_data)
    assert np.array_equal(result, np.array([[0.1, 0.2]]))


def test_process_input_returns_none_on_exception(model):
    model.features_transformer.transform = Mock(side_effect=Exception("Invalid input"))
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    model.features_transformer.transform.assert_called_once_with(input_data)

    assert result is None, (
        "Since there was an exception, the function should return None."
    )
