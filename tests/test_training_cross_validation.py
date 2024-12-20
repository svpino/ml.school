def test_cross_validation_creates_5_folds(training_run):
    data = training_run["cross_validation"].task.data
    assert len(data.folds) == 5
