import sqlite3

from metaflow import FlowSpec, IncludeFile, card, project, pypi_base, step


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

    @step
    def start(self):
        from io import StringIO

        import pandas as pd
        from evidently import ColumnMapping

        print("Start")

        # TODO: Use the common load function?
        self.reference_data = pd.read_csv(StringIO(self.dataset))

        # When running some of the tests and reports, we need to have a prediction
        # column in the reference data to match the production dataset.
        self.reference_data["prediction"] = self.reference_data["species"]

        # TODO: Need to load database from parameter.
        # TODO: Need to query a specific number of records from the database.
        connection = sqlite3.connect("penguins.db")
        query = (
            "SELECT island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, "
            "body_mass_g, prediction, species FROM data;"
        )
        self.production_data = pd.read_sql_query(query, connection)
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
            current_data=self.production_data,
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
            current_data=self.production_data,
            column_mapping=self.column_mapping,
        )

        self.html = report.get_html()
        self.next(self.data_drift_report)

    @card(type="html")
    @step
    def data_drift_report(self):
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(
            metrics=[
                DataDriftPreset(drift_share=0.01),
            ],
        )

        report.run(
            reference_data=self.reference_data,
            current_data=self.production_data,
            column_mapping=self.column_mapping,
        )

        self.html = report.get_html()

        self.next(self.target_drift_report)

    @card(type="html")
    @step
    def target_drift_report(self):
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
            current_data=self.production_data,
            # We only want to compute drift for the prediction column, so we need to
            # specify a column mapping without the target column.
            column_mapping=ColumnMapping(prediction="prediction"),
        )

        self.html = report.get_html()

        self.next(self.classification_report)

    @card(type="html")
    @step
    def classification_report(self):
        from evidently.metric_preset import ClassificationPreset
        from evidently.report import Report

        report = Report(
            metrics=[
                ClassificationPreset(),
            ],
        )

        report.run(
            reference_data=None,
            current_data=self.production_data,
            column_mapping=self.column_mapping,
        )

        self.html = report.get_html()

        self.next(self.end)

    @step
    def end(self):
        print("the end")


if __name__ == "__main__":
    MonitoringFlow()
