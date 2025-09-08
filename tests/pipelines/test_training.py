def test_start_loads_dataset(training_run):
    data = training_run["start"].task.data
    assert len(data.data) > 0


def test_start_creates_mlflow_run(training_run):
    data = training_run["start"].task.data
    assert data.mlflow_run_id is not None
