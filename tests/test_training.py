def test_start_loads_dataset(training_run):
    data = training_run["start"].task.data
    assert "species" in data.data.columns


def test_start_creates_mlflow_run(training_run):
    data = training_run["start"].task.data
    assert data.mlflow_run_id is not None
