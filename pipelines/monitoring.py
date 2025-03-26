
from common import PYTHON, DatasetMixin, Pipeline, packages
from inference.backend import BackendMixin
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    conda_base,
    project,
    step,
)


@project(name="penguins")
@conda_base(
    python=PYTHON,
    packages=packages("mlflow", "evidently", "pandas", "boto3"),
)
class Monitoring(FlowSpec, Pipeline, DatasetMixin, BackendMixin):
    """A monitoring pipeline to monitor the performance of a hosted model.

    This pipeline runs a series of tests and generates several reports using the
    data captured by the hosted model and a reference dataset.
    """

    limit = Parameter(
        "samples",
        help=(
            "The maximum number of samples that will be loaded from the production "
            "datastore to run the monitoring tests and reports. The flow will load "
            "the most recent samples."
        ),
        default=500,
    )

    @card
    @step
    def start(self):
        """Start the monitoring pipeline."""
        from evidently import ColumnMapping

        logger = self.configure_logging()

        self.reference_data = self.load_dataset(logger)
        self.backend_impl = self.load_backend(logger)

        # When running some of the tests and reports, we need to have a prediction
        # column in the reference data to match the production dataset.
        self.reference_data["prediction"] = self.reference_data["species"]
        self.reference_data = self.reference_data.rename(
            columns={"species": "ground_truth"},
        )
        self.current_data = self.backend_impl.load(self.limit)

        # Some of the tests and reports require labeled data, so we need to filter out
        # the samples that don't have ground truth labels.
        self.current_data_labeled = self.current_data[
            self.current_data["ground_truth"].notna()
        ]

        self.column_mapping = ColumnMapping(
            target="ground_truth",
            prediction="prediction",
        )

        self.next(self.test_suite)

    @card(type="html")
    @step
    def test_suite(self):
        """Run a test suite of pre-built tests.

        This test suite will run a group of pre-built tests to perform data checks.
        """
        from evidently.test_suite import TestSuite
        from evidently.tests import (
            TestColumnsType,
            TestColumnValueMean,
            TestNumberOfColumns,
            TestNumberOfDriftedColumns,
            TestNumberOfDuplicatedColumns,
            TestNumberOfEmptyColumns,
            TestNumberOfEmptyRows,
            TestNumberOfMissingValues,
            TestValueList,
        )

        test_suite = TestSuite(
            tests=[
                TestColumnsType(),
                TestNumberOfColumns(),
                TestNumberOfEmptyColumns(),
                TestNumberOfEmptyRows(),
                TestNumberOfDuplicatedColumns(),
                TestNumberOfMissingValues(),
                TestColumnValueMean(column_name="culmen_length_mm"),
                TestColumnValueMean(column_name="culmen_depth_mm"),
                TestColumnValueMean(column_name="flipper_length_mm"),
                TestColumnValueMean(column_name="body_mass_g"),
                TestValueList(
                    column_name="island",
                    values=["Biscoe", "Dream", "Torgersen"],
                ),
                TestNumberOfDriftedColumns(
                    columns=[
                        "culmen_length_mm",
                        "culmen_depth_mm",
                        "flipper_length_mm",
                        "body_mass_g",
                    ],
                    eq=0,
                ),
            ],
        )

        # We don't want to include the prediction and ground truth columns in any of
        # these tests, so let's remove them from the reference and production datasets.
        columns = ["prediction", "ground_truth"]
        reference_data = self.reference_data.copy().drop(columns=columns)
        current_data = self.current_data.copy().drop(columns=columns)

        test_suite.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        self.html = test_suite.get_html()

        self.next(self.data_quality_report)

    @card(type="html")
    @step
    def data_quality_report(self):
        """Generate a report about the quality of the data and any data drift.

        This report will provide detailed feature statistics, feature behavior
        overview of the data, and an evaluation of data drift with respect to the
        reference data. It will perform a side-by-side comparison between the
        reference and the production data.
        """
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report

        report = Report(
            metrics=[
                DataQualityPreset(),
                # We want to report dataset drift as long as one of the columns has
                # drifted. We can accomplish this by specifying that the share of
                # drifting columns in the production dataset must stay under 10% (one
                # column drifting out of 8 columns represents 12.5%).
                DataDriftPreset(drift_share=0.1),
            ],
        )

        # We don't want to compute data drift in the ground truth column, so we need to
        # remove it from the reference and production datasets.
        reference_data = self.reference_data.copy().drop(
            columns=["ground_truth"])
        current_data = self.current_data.copy().drop(columns=["ground_truth"])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        self.html = report.get_html()
        self.next(self.test_accuracy_score)

    @card(type="html")
    @step
    def test_accuracy_score(self):
        """Run a test to check the accuracy score of the model."""
        from evidently.test_suite import TestSuite
        from evidently.tests import (
            TestAccuracyScore,
        )

        test_suite = TestSuite(
            tests=[
                # This test will pass when the accuracy score of the model is
                # greater than or equal to the specified threshold.
                TestAccuracyScore(gte=0.9),
            ],
        )

        if not self.current_data_labeled.empty:
            test_suite.run(
                reference_data=self.reference_data,
                current_data=self.current_data_labeled,
                column_mapping=self.column_mapping,
            )

            self.html = test_suite.get_html()
        else:
            self._message("No labeled production data.")

        self.next(self.target_drift_report)

    @card(type="html")
    @step
    def target_drift_report(self):
        """Generate a Target Drift report."""
        from evidently import ColumnMapping
        from evidently.metric_preset import TargetDriftPreset
        from evidently.report import Report

        report = Report(
            metrics=[
                # This preset evaluates the prediction or target drift. It will show
                # any changes in model predictions with respect to the reference data.
                # This will help us understand if the distribution of model predictions
                # is different from the distribution of the target in the reference
                # dataset.
                TargetDriftPreset(),
            ],
        )

        if not self.current_data_labeled.empty:
            report.run(
                reference_data=self.reference_data,
                current_data=self.current_data_labeled,
                # We only want to compute drift for the prediction column, so we need to
                # specify a column mapping without the target column.
                column_mapping=ColumnMapping(prediction="prediction"),
            )

            self.html = report.get_html()
        else:
            self._message("No labeled production data.")

        self.next(self.classification_report)

    @card(type="html")
    @step
    def classification_report(self):
        """Generate a Classification report."""
        from evidently.metric_preset import ClassificationPreset
        from evidently.report import Report

        logger = self.configure_logging()

        report = Report(
            metrics=[
                # This preset evaluates the quality of a classification model.
                ClassificationPreset(),
            ],
        )

        if not self.current_data_labeled.empty:
            report.run(
                # The reference data is using the same target column as the prediction, so
                # we don't want to compute the metrics for the reference data to compare
                # them with the production data.
                reference_data=None,
                current_data=self.current_data_labeled,
                column_mapping=self.column_mapping,
            )
            try:
                self.html = report.get_html()
            except Exception:
                logger.exception("Error generating report.")
        else:
            self._message("No labeled production data.")

        self.next(self.end)

    @step
    def end(self):
        """Finish the monitoring flow."""
        logger = self.configure_logging()
        logger.info("Finishing monitoring flow.")

    def _message(self, message):
        """Display a message in the HTML card associated to a step."""
        self.html = message
        logger = self.configure_logging()
        logger.info(message)


if __name__ == "__main__":
    Monitoring()
