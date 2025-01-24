import pytest
from metaflow import Runner


@pytest.mark.integration
def test_register_doesnt_register_if_accuracy_under_threshold(mlflow_directory):
    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=False,
    ).run(
        mlflow_tracking_uri=mlflow_directory,
        training_epochs=1,
        accuracy_threshold=0.9,
    ) as running:
        data = running.run["register"].task.data
        assert data.registered is False, "Model shouldn't have been registered"


@pytest.mark.integration
def test_register_registers_model_if_accuracy_above_threshold(mlflow_directory):
    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=False,
    ).run(
        mlflow_tracking_uri=mlflow_directory,
        training_epochs=1,
        accuracy_threshold=0.0001,
    ) as running:
        data = running.run["register"].task.data
        assert data.registered is True, "Model should have been registered"


def test_register_pip_requirements(training_run):
    data = training_run["register"].task.data

    assert isinstance(data.pip_requirements, list)
    assert len(data.pip_requirements) > 0
    assert "pandas" in data.pip_requirements


def test_register_artifacts(training_run):
    data = training_run["register"].task.data

    assert "model" in data.artifacts
    assert "features_transformer" in data.artifacts
    assert "target_transformer" in data.artifacts


def test_register_code_paths_includes_default_endpoint(training_run):
    data = training_run["register"].task.data

    assert data.code_paths[0].endswith("backend.py")
