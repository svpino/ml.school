def test_evaluate_fold_computes_test_metrics(training_run):
    data = training_run["evaluate_fold"].task.data

    assert isinstance(data.test_accuracy, float)
    assert isinstance(data.test_loss, float)


def test_average_scores_computes_final_metrics(training_run):
    data = training_run["average_scores"].task.data

    assert isinstance(data.test_accuracy, float)
    assert isinstance(data.test_loss, float)
