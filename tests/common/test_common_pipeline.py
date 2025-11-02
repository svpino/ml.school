import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from metaflow import Run, Runner

from common.pipeline import mlflow, pipeline
from inference import backend


class MockStep:
    """A mock class representing a step in a Metaflow flow."""

    IGNORE = "ignore"

    def __init__(self) -> None:
        """Initialize the Step with a mock method."""
        self.add_decorator = MagicMock()


class MockMutableFlow:
    """A mock class representing a mutable Metaflow flow."""

    mlflow_tracking_uri = "http://test-uri:1234"

    def __init__(self) -> None:
        """Initialize the MutableFlow with a list of mock steps."""
        self.steps = [(f"step{i}", MockStep()) for i in range(3)]


@pytest.fixture(scope="module")
def metaflow_data():
    with Runner("tests/common/mock_pipeline.py", show_output=True).run() as running:
        return Run(running.run.pathspec).data


def test_dataset_decorator_loads_data(metaflow_data):
    assert len(metaflow_data.data) > 0


def test_backend_decorator_loads_backend(metaflow_data):
    assert isinstance(metaflow_data.backend_impl, backend.Local)


def test_dataset_decorator_cleans_sex_column(metaflow_data):
    sex_distribution = metaflow_data.data["sex"].value_counts()

    assert len(sex_distribution) == len(["MALE", "FEMALE"])
    assert sex_distribution.index[0] == "MALE"
    assert sex_distribution.index[1] == "FEMALE"


def test_pipeline_mutator_adds_logging_and_mlflow_decorators():
    mutator = pipeline()
    flow = MockMutableFlow()

    mutator.mutate(flow)

    for _, step in flow.steps:
        # Check that logging and mlflow decorators are added with correct arguments
        step.add_decorator.assert_has_calls(
            [
                call("logging", duplicates=step.IGNORE),
                call("mlflow", duplicates=step.IGNORE),
            ]
        )


def test_mlflow_decorator_sets_tracking_uri():
    flow = MockMutableFlow()
    fake_mlflow = SimpleNamespace(set_tracking_uri=MagicMock())

    # Ensure that `import mlflow` inside the decorator picks up our fake module
    with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
        decorator = mlflow()

        # The `pre_step` function executes the generator up to the `yield`,
        # where the call happens.
        decorator.pre_step("step", flow, inputs=None)

    fake_mlflow.set_tracking_uri.assert_called_once_with("http://test-uri:1234")
