# Monitoring Model Quality

The Monitoring pipeline will also monitor the model's predictions and performance.

This process will check the model's accuracy to ensure it's above a predefined threshold, look for drift in model predictions, compute several classification metrics, and plot different reports.

Here is a list of some of the tests and metrics the pipeline will run using the original dataset as the reference. You can find more information about each of these under the [Test Presets](https://docs.evidentlyai.com/reference/all-tests) and [Metric Presets](https://docs.evidentlyai.com/reference/all-metrics) in Evidently's documentation:

* `TestAccuracyScore`: This test will pass when the accuracy score of the model is greater than or equal to the specified threshold.

* `ColumnDriftMetric`: This metric calculates data drift for the target column. This metric is part of the `TargetDriftPreset`.

* `ColumnCorrelationsMetric`: This metric calculates the correlations between the target column and all the other columns in the dataset. This metric is part of the `TargetDriftPreset`.

* `ClassificationQualityMetric`: This metric calculates various classification performance metrics, including accuracy, precision, recall, and f1-score, among others. This metric is part of the `ClassificationPreset`.

* `ClassificationClassBalance`: This metric calculates the number of objects for each label and plots the histogram. This metric is part of the `ClassificationPreset`.

* `ClassificationConfusionMatrix`: This metric calculates the true-positive rate, true-negative rate, false-positive rate, and false-negative rate and plots the confusion matrix. This metric is part of the `ClassificationPreset`.

* `ClassificationQualityByClass`: This metric calculates the classification quality metrics for each class and plots the matrix. This metric is part of the `ClassificationPreset`.


