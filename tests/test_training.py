import tempfile
from pathlib import Path

import pytest
from metaflow import Runner


@pytest.fixture(scope="module")
def metaflow_run():
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


def test_start_loads_dataset(metaflow_run):
    data = metaflow_run["start"].task.data
    assert "species" in data.data.columns


def test_start_creates_mlflow_run(metaflow_run):
    data = metaflow_run["start"].task.data
    assert data.mlflow_run_id is not None


def test_cross_validation_creates_5_folds(metaflow_run):
    data = metaflow_run["cross_validation"].task.data
    assert len(data.folds) == 5


def test_transform_fold_sets_fold_index(metaflow_run):
    data = metaflow_run["transform_fold"].task.data
    assert data.fold in range(5)


def test_transform_fold_transforms_splits(metaflow_run):
    data = metaflow_run["transform_fold"].task.data

    train_size = len(data.train_indices)
    test_size = len(data.test_indices)

    assert data.x_train.shape == (train_size, 9)
    assert data.y_train.shape == (train_size, 1)

    assert data.x_test.shape == (test_size, 9)
    assert data.y_test.shape == (test_size, 1)


def test_train_fold_builds_model(metaflow_run):
    data = metaflow_run["train_fold"].task.data
    assert data.model is not None


def test_train_fold_creates_mlflow_nested_run(metaflow_run):
    data = metaflow_run["train_fold"].task.data
    assert data.mlflow_fold_run_id is not None


def test_evaluate_fold_computes_test_metrics(metaflow_run):
    data = metaflow_run["evaluate_fold"].task.data

    assert isinstance(data.test_accuracy, float)
    assert isinstance(data.test_loss, float)
