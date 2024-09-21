import logging
import sqlite3
import sys

from common import PYTHON, FlowMixin, get_boto3_client
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    project,
    pypi_base,
    step,
)
from sagemaker import load_labeled_data

logger = logging.getLogger(__name__)


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages={
        "evidently": "0.4.33",
        "pandas": "2.2.2",
        "boto3": "1.35.15",
    },
)
class MonitoringFlow(FlowSpec, FlowMixin):
    datastore_uri = Parameter(
        "datastore-uri",
        help=(
            "The location where the production data is stored. The pipeline supports "
            "loading the data from a SQLite database or from an S3 location that "
            "follows SageMaker's format for capturing data."
        ),
        required=True,
    )

    ground_truth_uri = Parameter(
        "ground-truth-uri",
        help=(
            "The S3 location where the ground truth labels associated with the "
            "endpoint's collected data is stored. The content of this S3 location must "
            "follow SageMaker's format for storing ground truth data."
        ),
        required=False,
    )

    assume_role = Parameter(
        "assume-role",
        help=(
            "The role the pipeline will assume to access the production data in S3. "
            "This parameter is required when the pipeline is running under a set of "
            "credentials that don't have access to the S3 location where the "
            "production data is stored."
        ),
        required=False,
    )

    limit = Parameter(
        "limit",
        help=(
            "The maximum number of samples that will be loaded from the production "
            "datastore to run the monitoring tests and reports. The flow will load "
            "the most recent samples."
        ),
        default=200,
    )

    @step
    def start(self):
        from evidently import ColumnMapping

        self.reference_data = self.load_dataset()

        # When running some of the tests and reports, we need to have a prediction
        # column in the reference data to match the production dataset.
        self.reference_data["prediction"] = self.reference_data["species"]
        self.current_data = self._load_production_datastore()

        # Some of the tests and reports require labeled data, so we need to filter out
        # the samples that don't have ground truth labels.
        self.current_data_labeled = self.current_data[
            self.current_data["species"].notna()
        ]

        # TODO: Explain
        self.column_mapping = ColumnMapping(
            target="species",
            prediction="prediction",
        )

        self.next(self.test_suite)

    @card(type="html")
    @step
    def test_suite(self):
        """Run a test suite of pre-built tests.

        This test suite will run a group of pre-built tests to perform structured data
        and model checks.
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
            TestShareOfMissingValues,
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
                TestShareOfMissingValues(),
                TestColumnValueMean("culmen_length_mm"),
                TestColumnValueMean("culmen_depth_mm"),
                TestColumnValueMean("flipper_length_mm"),
                TestColumnValueMean("body_mass_g"),
                # This test will pass only when the island column is one of the
                # specified values.
                TestValueList(
                    column_name="island",
                    values=["Biscoe", "Dream", "Torgersen"],
                ),
                # This test will pass only when the number of drifted columns from the
                # specified list is equal to the specified threshold.
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

        test_suite.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping,
        )

        self.html = test_suite.get_html()

        self.next(self.test_accuracy_score)

    @card(type="html")
    @step
    def test_accuracy_score(self):
        from evidently.test_suite import TestSuite
        from evidently.tests import (
            TestAccuracyScore,
        )

        test_suite = TestSuite(
            tests=[
                # This test will pass only when the accuracy score is greater than or
                # equal to the specified threshold.
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
            logger.info("No labeled production data.")

        self.next(self.data_quality_report)

    @card(type="html")
    @step
    def data_quality_report(self):
        """Generate a Data Quality report.

        This report will provide detailed feature statistics and a feature behavior
        overview of the data. It will perform a side-by-side comparison between the
        reference and the production data.
        """
        from evidently.metric_preset import DataQualityPreset
        from evidently.report import Report

        report = Report(
            metrics=[
                DataQualityPreset(),
            ],
        )

        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping,
        )

        self.html = report.get_html()
        self.next(self.data_drift_report)

    @card(type="html")
    @step
    def data_drift_report(self):
        """Generate a Data Drift report.

        This report will evaluate data drift in all the production dataset columns
        with respect to the reference data.
        """
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(
            metrics=[
                # We want to report dataset drift as long as one of the columns has
                # drifted. We can accomplish this by specifying that the share of
                # drifting columns in the production dataset must stay under 10% (one
                # column drifting out of 8 columns represents 12.5%).
                DataDriftPreset(drift_share=0.1),
            ],
        )

        # We don't want to compute data drift in the ground truth column, so we need to
        # remove it from the reference and production datasets.
        reference_data = self.reference_data.copy().drop(columns=["species"])
        current_data = self.current_data.copy().drop(columns=["species"])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        self.html = report.get_html()

        self.next(self.target_drift_report)

    @card(type="html")
    @step
    def target_drift_report(self):
        """Generate a Target Drift report.

        This report will explore any changes in model predictions with respect to the
        reference data. This will help us understand if the distribution of model
        predictions is different from the distribution of the target in the reference
        dataset.
        """
        from evidently import ColumnMapping
        from evidently.metric_preset import TargetDriftPreset
        from evidently.report import Report

        report = Report(
            metrics=[
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
            logger.info("No labeled production data.")

        self.next(self.classification_report)

    @card(type="html")
    @step
    def classification_report(self):
        """Generate a Classification report.

        This report will evaluate the quality of a classification model.
        """
        from evidently.metric_preset import ClassificationPreset
        from evidently.report import Report

        report = Report(
            metrics=[
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
            logger.info("No labeled production data.")

        self.next(self.end)

    @step
    def end(self):
        """Finish the monitoring flow."""
        logger.info("Finishing monitoring flow.")

    def _load_production_datastore(self):
        data = None
        if self.datastore_uri.startswith("s3://"):
            data = self._load_production_data_from_s3()
        elif self.datastore_uri.startswith("sqlite://"):
            data = self._load_production_data_from_sqlite()
        else:
            message = (
                "Invalid datastore location. Must be an S3 location in the "
                "format `s3://bucket/prefix` or a SQLite database file in the format "
                "`sqlite:///path/to/database.db`"
            )
            raise ValueError(message)

        logger.info("Loaded %d samples from the production dataset.", len(data))

        return data

    def _load_production_data_from_s3(self):
        if self.ground_truth_uri is None:
            message = (
                'The "groundtruth-uri" parameter is required when loading the '
                "production data from S3."
            )
            raise RuntimeError(message)

        s3_client = get_boto3_client(service="s3", assume_role=self.assume_role)

        return load_labeled_data(
            s3_client,
            data_uri=self.datastore_uri,
            ground_truth_uri=self.ground_truth_uri,
        )

    def _load_production_data_from_sqlite(self):
        import pandas as pd

        connection = sqlite3.connect(self.datastore_uri)

        query = (
            "SELECT island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, "
            "body_mass_g, prediction, species FROM data "
            "ORDER BY date DESC LIMIT ?;"
        )

        # Notice that we are using the `samples` parameter to limit the number of
        # samples we are loading from the database.
        data = pd.read_sql_query(query, connection, params=(self.limit,))

        connection.close()

        return data


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    MonitoringFlow()
