import logging
import sqlite3
import sys

from common import load_dataset

from metaflow import (
    FlowSpec,
    IncludeFile,
    Parameter,
    card,
    current,
    project,
    pypi_base,
    step,
)

logger = logging.getLogger(__name__)


@project(name="penguins")
@pypi_base(
    python="3.10.14",
    packages={
        "evidently": "0.4.33",
        "pandas": "2.2.2",
    },
)
class MonitoringFlow(FlowSpec):
    dataset = IncludeFile(
        "penguins",
        is_text=True,
        help=(
            "Local copy of the penguins dataset. This file will be included in the "
            "flow and will be used whenever the flow is executed in development mode."
        ),
        default="../penguins.csv",
    )

    samples = Parameter(
        "samples",
        help=(
            "The number of most recent samples that will be loaded from the "
            "production dataset to run the monitoring tests and reports."
        ),
        default=200,
    )

    @step
    def start(self):
        import pandas as pd
        from evidently import ColumnMapping

        self.mode = "production" if current.is_production else "development"
        logger.info("Running flow in %s mode.", self.mode)

        self.reference_data = load_dataset(
            self.dataset,
            is_production=current.is_production,
        )

        # When running some of the tests and reports, we need to have a prediction
        # column in the reference data to match the production dataset.
        self.reference_data["prediction"] = self.reference_data["species"]

        # TODO: Need to load database from parameter.
        # TODO: Need to query a specific number of records from the database.

        connection = sqlite3.connect("penguins.db")

        # We need to make sure we are only loading the data that has ground truth
        # labels (species column is not null).
        query = (
            "SELECT island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, "
            "body_mass_g, prediction, species FROM data WHERE species IS NOT NULL "
            "ORDER BY date DESC LIMIT ?;"
        )

        self.current_data = pd.read_sql_query(query, connection, params=(self.samples,))
        logger.info(
            "Loaded %d out of %d samples requested from the production dataset.",
            len(self.current_data),
            self.samples,
        )

        connection.close()

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
            TestAccuracyScore,
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
                # This test will pass only when the accuracy score is greater than or
                # equal to the specified threshold.
                TestAccuracyScore(gte=0.9),
            ],
        )

        test_suite.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping,
        )

        self.html = test_suite.get_html()

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

        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
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

        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            # We only want to compute drift for the prediction column, so we need to
            # specify a column mapping without the target column.
            column_mapping=ColumnMapping(prediction="prediction"),
        )

        self.html = report.get_html()

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

        report.run(
            # The reference data is using the same target column as the prediction, so
            # we don't want to compute the metrics for the reference data to compare
            # them with the production data.
            reference_data=None,
            current_data=self.current_data,
            column_mapping=self.column_mapping,
        )

        self.html = report.get_html()

        self.next(self.end)

    @step
    def end(self):
        """Finish the monitoring flow."""
        logger.info("Finishing monitoring flow.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    MonitoringFlow()
