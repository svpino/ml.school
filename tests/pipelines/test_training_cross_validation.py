def test_cross_validation_creates_5_folds(training_run):
    data = training_run["cross_validation"].task.data
    cross_validation_folds = 5
    assert len(data.folds) == cross_validation_folds
