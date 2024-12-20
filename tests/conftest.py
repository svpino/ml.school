import tempfile
from pathlib import Path

import pytest
from metaflow import Runner


@pytest.fixture(scope="session")
def training_run():
    temporal_directory = tempfile.gettempdir()
    mlflow_directory = Path(temporal_directory) / "mlflow"

    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=True,
    ).run(
        mlflow_tracking_uri=mlflow_directory.as_posix(),
        training_epochs=1,
        accuracy_threshold=0.1,
    ) as running:
        return running.run
