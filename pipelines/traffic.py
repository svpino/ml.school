
from common import DatasetMixin, logging
from inference.backend import BackendMixin
from metaflow import (
    FlowSpec,
    Parameter,
    project,
    step,
)


@project(name="penguins")
class Traffic(FlowSpec, DatasetMixin, BackendMixin):
    """A pipeline for sending fake traffic to a hosted model."""

    samples = Parameter(
        "samples",
        help="The number of samples that will be sent to the hosted model.",
        default=200,
        required=False,
    )

    drift = Parameter(
        "drift",
        help=(
            "Whether to introduce drift in the samples sent to the model. This is "
            "useful for testing the monitoring process."
        ),
        default=False,
        required=False,
    )

    @logging
    @step
    def start(self):
        """Start the pipeline and load the dataset."""
        self.backend_impl = self.load_backend(self.logger)
        self.data = self.load_dataset(self.logger)

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """Prepare the data and introduce drift before submitting it to the model."""
        import numpy as np

        # We don't need to send the target column to the model.
        self.data.pop("species")
        self.data = self.data.dropna()

        # If we want to introduce drift, we can add random noise to one of the
        # numerical features in the data.
        if self.drift:
            rng = np.random.default_rng()
            self.data["body_mass_g"] += rng.uniform(
                1,
                3 * self.data["body_mass_g"].std(),
                size=len(self.data),
            )

        self.next(self.generate_traffic)

    @step
    def generate_traffic(self):
        """Prepare the payload and send traffic to the hosted model."""
        import pandas as pd

        self.dispatched_samples = 0

        while self.dispatched_samples < self.samples:
            batch = min(self.samples - self.dispatched_samples, 10)
            payload = {}

            batch = self.data.sample(n=batch)

            payload = [
                {k: (None if pd.isna(v) else v)
                 for k, v in row.to_dict().items()}
                for _, row in batch.iterrows()
            ]

            predictions = self.backend_impl.invoke(payload)
            if predictions is None:
                self.logger().error(
                    "Failed to get predictions from the hosted model.")
                break

            self.dispatched_samples += len(batch)

        self.next(self.end)

    @logging
    @step
    def end(self):
        """End of the pipeline."""
        self.logger.info("Sent %s samples to the hosted model.",
                         self.dispatched_samples)


if __name__ == "__main__":
    Traffic()
