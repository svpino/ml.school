from common import DatasetMixin, Pipeline
from inference.backend import BackendMixin
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    project,
    step,
)


@project(name="penguins")
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
        from evidently import DataDefinition, Dataset, MulticlassClassification

        logger = self.logger()
        self.backend_impl = self.load_backend(logger)

        # Let's load the reference data. When running some of the tests and reports,
        # we need to have a prediction column in the reference data to match the
        # production dataset.
        reference_data = self.load_dataset(logger)
        reference_data["prediction"] = reference_data["species"]
        reference_data = reference_data.rename(
            columns={"species": "target"},
        )

        data_definition = DataDefinition(
            classification=[MulticlassClassification(
                target="target",
                prediction_labels="prediction",
            )]
        )

        self.reference_dataset = Dataset.from_pandas(
            reference_data,
            data_definition=data_definition
        )

        # Let's now load the production data. We need to filter out the samples that
        # don't have ground truth labels.
        current_data = self.backend_impl.load(self.limit)
        current_data = current_data[current_data["target"].notna(
        )] if current_data is not None and not current_data.empty else None

        # We want to make sure there's production data available to run the reports.
        # If there's no production data, we'll skip the reports that need it.
        self.current_dataset = None
        if current_data is not None and not current_data.empty:
            self.current_dataset = Dataset.from_pandas(
                current_data,
                data_definition=data_definition
            )

        self.next(self.data_summary_report)

    @card(type="html")
    @step
    def data_summary_report(self):
        """Generate a report with descriptive statistics for each column.

        This report will provide detailed feature statistics, and will run a few tests
        to check for missing values, duplicated rows, and other data quality issues.
        """
        from evidently import Report
        from evidently.metrics import DuplicatedRowCount
        from evidently.presets import ValueStats

        report = Report([
            # These will generate statistics for each individual column.
            ValueStats(column="island", row_count_tests=[]),
            ValueStats(column="sex", row_count_tests=[]),
            ValueStats(column="culmen_length_mm", row_count_tests=[]),
            ValueStats(column="culmen_depth_mm", row_count_tests=[]),
            ValueStats(column="flipper_length_mm", row_count_tests=[]),
            ValueStats(column="body_mass_g", row_count_tests=[]),
            # This will check for duplicated rows in the dataset. Having duplicated
            # rows is not a problem for the model, but it might indicate an issue
            # with the data pipeline.
            DuplicatedRowCount(),
        ], include_tests=True)

        # We only want to run the report if there's production data available.
        if self.current_dataset:
            result = report.run(
                current_data=self.current_dataset,
                reference_data=self.reference_dataset,
            )
            self.html = result._repr_html_()
        else:
            self._message("No production data.")

        self.next(self.data_drift_report)

    @card(type="html")
    @step
    def data_drift_report(self):
        """Generate a report visualizing data distribution and drift.

        This report will generate a visualization of the data distribution of every
        column and determine if there's any drift in the data.
        """
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(
            [
                # We want to report dataset drift as long as one of the columns has
                # drifted. We can accomplish this by specifying that the share of
                # drifting columns in the production dataset must stay under 10% (one
                # column drifting out of 6 columns represents 16.66%).
                DataDriftPreset(
                    columns=["island", "sex", "culmen_length_mm", "culmen_depth_mm",
                             "flipper_length_mm", "body_mass_g"],
                    drift_share=0.1),
            ],
            include_tests=True
        )

        # We only want to run the report if there's production data available.
        if self.current_dataset:
            result = report.run(
                reference_data=self.reference_dataset,
                current_data=self.current_dataset,
            )
            self.html = result._repr_html_()
        else:
            self._message("No production data.")

        self.next(self.classification_report)

    @card(type="html")
    @step
    def classification_report(self):
        """Generate a Classification report.

        This report will evaluate the quality of the multi-class classification model.
        """
        from evidently import Report
        from evidently.presets import ClassificationPreset

        report = Report(
            [
                # This preset evaluates the quality of the classification model.
                ClassificationPreset(),
            ],
            include_tests=True
        )

        # We only want to run the report if there's production data available.
        if self.current_dataset:
            result = report.run(
                # The reference data is using the same target column as the prediction, so
                # we don't want to compute the metrics for the reference data to compare
                # them with the production data.
                current_data=self.current_dataset,
                reference_data=self.reference_dataset,
            )
            self.html = result._repr_html_()
        else:
            self._message("No production data.")

        self.next(self.end)

    @step
    def end(self):
        """Finish the monitoring flow."""
        self.logger().info("Finishing monitoring flow.")

    def _message(self, message):
        """Display a message in the HTML card associated to a step."""
        self.html = message
        self.logger().info(message)


if __name__ == "__main__":
    Monitoring()
