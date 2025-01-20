# Monitoring Data Quality

The first goal of the Monitoring pipeline is to keep an eye on the quality of the traffic processed by the model.

This process involves tracking the input data distribution, looking for unexpected correlations, missing values, outliers, and any anomalies indicating a drift from the reference dataset. 

We want to ensure that the data feeding into the hosted model remains consistent with the data we used to train the model.

The Monitoring pipeline uses [Evidently](https://github.com/evidentlyai/evidently), an open-source Python library that will help us evaluate, test, and monitor the input data into the hosted model.

Here is a list of some of the tests the pipeline will perform using the original dataset as the reference. You can find more information about each of these under the [Test Presets](https://docs.evidentlyai.com/reference/all-tests) and [Metric Presets](https://docs.evidentlyai.com/reference/all-metrics) in Evidently's documentation:

* `TestColumnsType`: This test will pass when the type of every column in the production data matches the corresponding column type in the reference data.

* `TestNumberOfColumns`: This test will pass when the number of columns in the production data equals the number of columns in the reference data.

* `TestNumberOfEmptyColumns`: This test will pass when the number of empty columns in the production data is under the number of empty columns in the reference data.

* `TestNumberOfEmptyRows`: This test will pass when the share of empty rows is within 10% of the share of empty rows in the reference data.

* `TestNumberOfDuplicatedColumns`: This test will pass when the number of duplicated columns in the production data is under the number of duplicated columns in the reference data.

* `TestNumberOfMissingValues`: This test will pass when the number of missing values in the production data is under 10% of the number of missing values in the reference data.

* `TestColumnValueMean`: These tests will pass if the mean value of the specified columns in the production data is within 10% of the mean value of the same columns in the reference data.

* `TestValueList`: This test will pass when the supplied column is one of the values specified in the list.

* `TestNumberOfDriftedColumns`: This test will pass when the number of drifted columns from the specified list equals the specified threshold.

* `DatasetSummaryMetric`: This metric calculates descriptive dataset statistics, including the number of columns by type, number of rows, missing values, empty columns, and duplicated columns, among others. This metric is part of the `DataQualityPreset`.

*  `ColumnSummaryMetric`: This metric calculates various descriptive statistics for the numerical and categorical columns in the data, including their count, minimum value, maximum value, mean, standard deviation, quantiles, unique value share, most common value share, missing value share, and new and missing categories among others. This metric is part of the `DataQualityPreset`.

* `DatasetMissingValuesMetric`: This metric calculates the number and share of missing values in the dataset and displays the number of missing values per column. This metric is part of the `DataQualityPreset`.

* `DataDriftTable`: This metric calculates data drift for the specified columns, returns drift detection results, and visualizes distributions for the columns in a table. This metric is part of the `DataDriftPreset`.

* `DatasetDriftMetric`: This metric calculates the number and share of drifted features in the dataset. This metric is part of the `DataDriftPreset`.

                

