import tempfile
from pathlib import Path

import pytest
from metaflow import Runner


@pytest.fixture(scope="session")
def mlflow_directory():
    temporal_directory = tempfile.gettempdir()
    return (Path(temporal_directory) / "mlflow").as_posix()


@pytest.fixture(scope="session")
def training_run(mlflow_directory):
    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=False,
    ).run(
        mlflow_tracking_uri=mlflow_directory,
        training_epochs=1,
        accuracy_threshold=0.1,
    ) as running:
        return running.run


@pytest.fixture(scope="session")
def monitoring_run():
    with Runner(
        "pipelines/monitoring.py",
        environment="conda",
        show_output=False,
    ).run(
        backend="backend.Mock",
    ) as running:
        return running.run
